import argparse
import os
from copy import deepcopy
from dataclasses import dataclass
from threading import Lock
from typing import Any

import torch
from flask import Flask, jsonify, request
from torch.utils.data import DataLoader

from olmo.config import DistributedStrategy, TrainConfig
from olmo.data.collator import DataCollator
from olmo.model import OLMo
from olmo.optim import build_optimizer, build_scheduler
from olmo.tokenizer import Tokenizer
from olmo.torch_util import SingleAccelerator, move_to_device
from olmo.train import Trainer


@dataclass
class LoadedModel:
    trainer: Trainer
    cfg: TrainConfig


class InfluenceService:
    """Serve text influence as loss_before - loss_after using two OLMo checkpoints."""

    def __init__(
        self,
        before_checkpoint: str,
        after_checkpoint: str,
        config_path: str | None = None,
        batch_size: int = 8,
        max_seq_len: int = 2048,
    ):
        self.before_checkpoint = before_checkpoint
        self.after_checkpoint = after_checkpoint
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        cfg_path = config_path or os.path.join(before_checkpoint, "config.yaml")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(
                f"Config file not found: {cfg_path}. "
                "Pass --config explicitly with the exact training config used for the checkpoints."
            )
        self.base_cfg = TrainConfig.load(cfg_path, [])

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = Tokenizer.from_train_config(self.base_cfg)

        self.before = self._load_checkpoint(self.before_checkpoint)
        self.after = self._load_checkpoint(self.after_checkpoint)

    def _dummy_loader(self, cfg: TrainConfig) -> DataLoader:
        # Trainer requires a train loader even for eval-only usage; provide one tiny dummy batch.
        collator = DataCollator.from_train_config(cfg)
        dummy = [{"input_ids": [cfg.model.eos_token_id, cfg.model.eos_token_id]}]
        return DataLoader(dummy, batch_size=1, collate_fn=collator)

    def _normalize_cfg(self, cfg: TrainConfig) -> TrainConfig:
        cfg.distributed_strategy = DistributedStrategy.single
        cfg.model.init_device = str(self.device)
        cfg.model.precision = cfg.precision
        cfg.device_eval_batch_size = self.batch_size
        cfg.device_train_batch_size = 1
        cfg.device_train_microbatch_size = 1
        cfg.device_train_grad_accum = 1

        # Avoid noisy or expensive data loader setup for the placeholder train loader.
        cfg.data.num_workers = 0
        cfg.data.pin_memory = False
        cfg.data.prefetch_factor = None
        cfg.data.persistent_workers = False
        cfg.data.timeout = 0
        return cfg

    def _load_checkpoint(self, checkpoint_path: str) -> LoadedModel:
        cfg = self._normalize_cfg(deepcopy(self.base_cfg))

        model = OLMo(cfg.model).to(self.device)
        dist_model = SingleAccelerator(model)
        optim = build_optimizer(cfg, dist_model)
        scheduler = build_scheduler(cfg)
        train_loader = self._dummy_loader(cfg)

        trainer = Trainer(
            cfg=cfg,
            model=model,
            dist_model=dist_model,
            optim=optim,
            scheduler=scheduler,
            train_loader=train_loader,
            device=self.device,
            evaluators=[],
        )
        trainer.__enter__()
        try:
            trainer.restore_checkpoint(
                checkpoint_path,
                load_optimizer_state=False,
                load_trainer_state=False,
                sharded_checkpointer=cfg.load_path_sharded_checkpointer,
            )
        except RuntimeError as error:
            raise RuntimeError(
                "Checkpoint/model shape mismatch while loading influence model. "
                f"Current config builds d_model={cfg.model.d_model}, n_layers={cfg.model.n_layers}, "
                f"n_heads={cfg.model.n_heads}, vocab_size={cfg.model.vocab_size}, "
                f"embedding_size={cfg.model.embedding_size}. "
                f"Failed checkpoint: {checkpoint_path}. "
                "Make sure --config matches the checkpoint architecture exactly."
            ) from error
        trainer.dist_model.eval()
        return LoadedModel(trainer=trainer, cfg=cfg)

    def close(self) -> None:
        for loaded in (self.before, self.after):
            try:
                loaded.trainer.__exit__(None, None, None)
            except Exception:
                pass

    def _extract_losses(self, loaded: LoadedModel, texts: list[str]) -> list[float]:
        encoded = self.tokenizer.encode_batch(texts, add_special_tokens=True)
        encoded = [tokens[: self.max_seq_len] for tokens in encoded]

        losses: list[float] = []
        collator = DataCollator.from_train_config(loaded.cfg)

        for start in range(0, len(encoded), self.batch_size):
            batch_items = [
                {"input_ids": toks} for toks in encoded[start : start + self.batch_size]
            ]
            batch = collator(batch_items)
            batch = move_to_device(batch, self.device)
            with torch.no_grad():
                ce_loss, _ = loaded.trainer.eval_batch(batch)
            losses.extend(ce_loss.detach().float().cpu().tolist())

        return [float(x) for x in losses]

    def score(self, texts: list[str]) -> dict[str, Any]:
        if not texts:
            return {"loss_before": [], "loss_after": [], "influence_scores": []}

        loss_before = self._extract_losses(self.before, texts)
        loss_after = self._extract_losses(self.after, texts)
        influence = [lb - la for lb, la in zip(loss_before, loss_after)]
        return {
            "loss_before": loss_before,
            "loss_after": loss_after,
            "influence_scores": influence,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve OLMo checkpoint-difference influence scores"
    )
    parser.add_argument("--before-checkpoint", required=True)
    parser.add_argument("--after-checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=24775)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    service = InfluenceService(
        before_checkpoint=args.before_checkpoint,
        after_checkpoint=args.after_checkpoint,
        config_path=args.config,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )

    app = Flask(__name__)
    processing_lock = Lock()

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/get_influence")
    def get_influence():
        data = request.get_json(force=True, silent=True)
        if not data or not isinstance(data.get("texts"), list):
            return (
                jsonify({"error": "Expected JSON with field 'texts': [str,...]"}),
                400,
            )
        try:
            with processing_lock:
                result = service.score(data["texts"])
            return jsonify(result)
        except Exception as error:
            return jsonify({"error": str(error)}), 500

    try:
        print(f"Influence server running at http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        service.close()


if __name__ == "__main__":
    main()
