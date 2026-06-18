#!/usr/bin/env python3
"""Probe data influence via validation-updated model.

This script extends train.py to compute data influence by:
1. Loading a checkpoint (theta_hat)
2. Computing initial loss (loss0) on training data
3. Loading a validation-updated checkpoint (theta_hat - eta * grad_val)
4. Computing updated loss (loss1) on training data
5. Computing influence = loss0 - loss1

This script reuses train.py's entire infrastructure for FSDP compatibility.
"""

import logging
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from olmo.config import (
    DataConfig,
    DistributedStrategy,
    TrainConfig,
)
from olmo.data import build_memmap_dataset, build_train_dataloader
from olmo.data.collator import DataCollator
from olmo.exceptions import OLMoCliError, OLMoConfigurationError
from olmo.model import OLMo
from olmo.optim import build_optimizer, build_scheduler
from olmo.torch_util import (
    barrier,
    get_default_device,
    get_global_rank,
    get_local_rank,
    get_world_size,
    move_to_device,
    seed_all,
)
from olmo.train import Trainer
from olmo.util import (
    add_cached_path_clients,
    clean_opt,
    log_extra_field,
    prepare_cli_environment,
)

log = logging.getLogger("probe_influence")


class IndexedDataset(Dataset[Dict[str, Any]]):
    """Wrapper to add index to each dataset item for tracking."""

    def __init__(self, base: Dataset[Dict[str, Any]]):
        self._base = base

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self._base[idx]
        if isinstance(item, dict):
            out = dict(item)
        else:
            out = {"input_ids": item}
        out["index"] = idx
        return out


def build_indexed_dataloader(cfg: TrainConfig) -> DataLoader:
    """Build a DataLoader with index tracking for a single data file.

    Uses DistributedSampler to split data across ranks for efficient parallel evaluation.
    """
    from torch.utils.data.distributed import DistributedSampler

    data_cfg = DataConfig(
        paths=cfg.evaluators[0].data.paths,
        memmap_dtype=cfg.data.memmap_dtype,
        pad_direction=cfg.data.pad_direction,
        num_workers=cfg.data.num_workers,
        drop_last=False,  # Don't drop last batch for complete coverage
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        persistent_workers=cfg.data.persistent_workers if cfg.data.num_workers > 0 else False,
        timeout=cfg.data.timeout,
    )

    base_dataset = build_memmap_dataset(cfg, data_cfg, include_instance_metadata=False)
    dataset = IndexedDataset(base_dataset)
    collator = DataCollator.from_train_config(cfg)

    # Use DistributedSampler to split data across ranks
    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        shuffle=False,
        drop_last=False,
    )

    return DataLoader(
        dataset,
        batch_size=cfg.device_eval_batch_size,
        sampler=sampler,  # Use distributed sampler instead of shuffle
        collate_fn=collator,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        prefetch_factor=data_cfg.prefetch_factor,
        persistent_workers=data_cfg.persistent_workers,
        timeout=data_cfg.timeout,
    )


@torch.no_grad()
def compute_per_example_losses(
    trainer: Trainer,
    data_loader: DataLoader,
    device: torch.device,
    desc: str = "Computing losses",
) -> np.ndarray:
    """Compute per-example losses for all data in the loader.

    Each rank computes losses for its subset of data (via DistributedSampler).
    Results are combined across ranks via all-reduce.

    Returns:
        Array of shape (num_examples,) with per-example losses (complete on all ranks)
    """
    num_examples = len(data_loader.dataset)
    if get_global_rank() == 0:
        log.info(f"Total examples to evaluate: {num_examples:,d}")  # 276,229
        log.info(f"Examples per rank: ~{num_examples // get_world_size():,d}")

    # Initialize with zeros - each rank will fill in its subset
    losses = np.zeros(num_examples, dtype=np.float32)

    trainer.dist_model.eval()
    batch_count = 0
    for batch in tqdm(data_loader, desc=desc, disable=get_global_rank() != 0):
        batch = move_to_device(batch, device)

        # Get per-example losses (eval_batch returns ce_loss with shape [batch_size])
        ce_loss, _ = trainer.eval_batch(batch)

        # Get indices and store losses
        indices = batch["index"].detach().cpu().numpy().astype(np.int64)
        losses[indices] = ce_loss.detach().cpu().numpy().astype(np.float32)

        # Print sample data for sanity check (first batch only, rank 0 only)
        if batch_count == 0 and get_global_rank() == 0:
            input_ids = batch["input_ids"].detach().cpu().numpy()
            batch_indices = indices
            log.info("\n" + "=" * 60)
            log.info("SANITY CHECK: Sample data from first batch")
            log.info("=" * 60)
            for i in range(min(3, len(batch_indices))):
                log.info(f"Sample {i} (dataset index {batch_indices[i]}):")
                log.info(f"  Shape: {input_ids[i].shape}")
                log.info(f"  First 10 tokens: {input_ids[i][:10].tolist()}")
                log.info(f"  Last 10 tokens: {input_ids[i][-10:].tolist()}")
                log.info(f"  Token range: [{input_ids[i].min()}, {input_ids[i].max()}]")
            log.info("=" * 60 + "\n")
        batch_count += 1

    # All-reduce to combine results from all ranks
    # Each rank has different indices filled in, so sum will give us the complete result
    losses_tensor = torch.from_numpy(losses).to(device)
    dist.all_reduce(losses_tensor, op=dist.ReduceOp.SUM)
    losses = losses_tensor.cpu().numpy()

    return losses


def log_model_fingerprint(tag: str, model: torch.nn.Module) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param is None or param.numel() == 0:
                continue
            value = param.detach().float().sum().item()
            log.info(f"{tag} | {name} sum={value:.6e}")
            break


def compute_losses_with_checkpoint(
    cfg: TrainConfig,
    checkpoint_path: str,
    train_eval_loader: DataLoader,
    device: torch.device,
    desc: str = "Computing losses",
) -> np.ndarray:
    """Load a checkpoint and compute per-example losses.

    This creates a fresh trainer instance to avoid any cached state from FSDP.
    """
    log.info(f"Building trainer for checkpoint: {checkpoint_path}")

    # Build validation dataloader (for training)
    valid_train_loader = build_train_dataloader(cfg)

    # Initialize model
    olmo_model = OLMo(cfg.model)

    # Compile one block at a time
    if cfg.compile is not None:
        if cfg.model.block_group_size != 1:
            raise OLMoConfigurationError("Compile is only supported with block_group_size 1.")
        for block in olmo_model.transformer.blocks:
            block.compile(**cfg.compile.asdict())

    olmo_model.set_activation_checkpointing(cfg.activation_checkpointing)

    # Wrap model with FSDP/DDP
    if cfg.distributed_strategy == DistributedStrategy.fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
        from packaging import version

        assert cfg.fsdp is not None
        wrap_policy = olmo_model.get_fsdp_wrap_policy(cfg.fsdp.wrapping_strategy)

        if version.parse(torch.__version__) >= version.parse("2.1.0"):

            def dummy_init_fn(module: torch.nn.Module) -> None:
                module.to_empty(device=get_default_device())

            param_init_fn = dummy_init_fn
        else:
            param_init_fn = None

        # Handle hybrid sharding if needed
        device_mesh = None
        hybrid_sharding_fsdp_kwargs = {}
        if cfg.fsdp.sharding_strategy in (ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2):
            if version.parse(torch.__version__) < version.parse("2.2.0"):
                raise OLMoConfigurationError("Hybrid sharding requires torch >= 2.2.0")
            from torch.distributed.device_mesh import init_device_mesh
            from olmo.torch_util import get_local_world_size

            num_model_replicas = cfg.fsdp.hybrid_sharding_num_model_replicas or (
                get_world_size() // get_local_world_size()
            )
            if num_model_replicas <= 0:
                raise OLMoConfigurationError("hybrid_sharding_num_model_replicas must be positive")
            if get_world_size() % num_model_replicas != 0:
                raise OLMoConfigurationError("hybrid_sharding_num_model_replicas must divide world size")

            device_mesh = init_device_mesh("cuda", (num_model_replicas, get_world_size() // num_model_replicas))
            hybrid_sharding_fsdp_kwargs["device_mesh"] = device_mesh

        dist_model = FSDP(
            olmo_model,
            sharding_strategy=cfg.fsdp.sharding_strategy,
            mixed_precision=cfg.fsdp_precision,
            auto_wrap_policy=wrap_policy,
            use_orig_params=cfg.fsdp.use_orig_params,
            limit_all_gathers=True,
            device_id=get_local_rank(),
            param_init_fn=param_init_fn,
            **hybrid_sharding_fsdp_kwargs,
        )

        if param_init_fn is not None:
            olmo_model.reset_parameters()
    else:
        # Single GPU or DDP
        from olmo.torch_util import SingleAccelerator

        olmo_model = olmo_model.to(device)
        dist_model = SingleAccelerator(olmo_model)
        olmo_model.reset_parameters()

    # Build optimizer and scheduler
    optim = build_optimizer(cfg, dist_model)
    scheduler = build_scheduler(cfg)

    # Create Trainer and load checkpoint
    with Trainer(
        cfg=cfg,
        model=olmo_model,
        dist_model=dist_model,
        optim=optim,
        scheduler=scheduler,
        train_loader=valid_train_loader,
        device=device,
        evaluators=[],
    ) as trainer:
        log.info(f"Loading checkpoint from {checkpoint_path}...")
        trainer.restore_checkpoint(
            checkpoint_path,
            load_optimizer_state=False,  # We don't need optimizer state for evaluation
            load_trainer_state=False,  # We don't need trainer state for evaluation
            sharded_checkpointer=cfg.load_path_sharded_checkpointer,
        )
        log.info("Checkpoint successfully loaded")
        log_model_fingerprint(f"Checkpoint fingerprint ({checkpoint_path})", trainer.dist_model)

        # Compute losses
        losses = compute_per_example_losses(trainer, train_eval_loader, device, desc)

    # Clean up
    del trainer, dist_model, olmo_model, optim, scheduler, valid_train_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return losses


def main(cfg: TrainConfig) -> None:
    """Main influence computation logic.

    This follows train.py's structure but adds influence computation steps.
    """
    # Get influence-specific config from command line args
    output_dir = Path(cfg.save_folder)
    eval_batch_size = cfg.device_eval_batch_size

    log.info("=" * 60)
    log.info("Data Influence Computation")
    log.info("=" * 60)
    log.info(f"Initial checkpoint: {cfg.load_path}")
    log.info(f"Updated checkpoint: {cfg.load_path}/data_influence/latest-unsharded")
    log.info(f"Training data (for eval): {cfg.evaluators[0].data.paths}")
    log.info(f"Output directory: {output_dir}")
    log.info(f"Eval batch size: {eval_batch_size}")
    log.info("=" * 60)

    log_extra_field("run_name", cfg.run_name)

    barrier()

    # Set CUDA device (from train.py)
    if torch.cuda.is_available():
        torch.cuda.set_device(f"cuda:{get_local_rank()}")
        torch.cuda.empty_cache()
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Fill configuration options (from train.py)
    cfg.model.precision = cfg.precision
    cfg.device_train_batch_size = cfg.global_train_batch_size // get_world_size()
    assert cfg.device_train_batch_size is not None
    cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size

    barrier()

    # Set seed
    seed_all(cfg.seed)

    # Build training dataloader (for evaluation) - use indexed loader
    log.info(f"Building training dataloader (for evaluation)...")
    train_eval_loader = build_indexed_dataloader(cfg)

    barrier()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Compute initial loss (loss0) with checkpoint A
    log.info("=" * 60)
    log.info("Phase 1: Computing initial loss (loss0) with initial checkpoint")
    log.info("=" * 60)

    loss0 = compute_losses_with_checkpoint(cfg, cfg.load_path, train_eval_loader, device, "Computing loss0")

    if get_global_rank() == 0:
        log.info(f"Saving loss0 to {output_dir / 'loss_before.npy'}")
        np.save(output_dir / "loss_before.npy", loss0)
        log.info(f"Loss0 - Mean: {np.mean(loss0):.4f}, Std: {np.std(loss0):.4f}")

    barrier()

    # Phase 2: Compute updated loss (loss1) with checkpoint B
    log.info("=" * 60)
    log.info("Phase 2: Computing updated loss (loss1) with updated checkpoint")
    log.info("=" * 60)

    # Create a new config with different load_path
    cfg_after = deepcopy(cfg)
    cfg_after.load_path = cfg.load_path + "/data_influence/latest-unsharded"

    loss1 = compute_losses_with_checkpoint(cfg_after, cfg_after.load_path, train_eval_loader, device, "Computing loss1")

    if get_global_rank() == 0:
        log.info(f"Saving loss1 to {output_dir / 'loss_after.npy'}")
        np.save(output_dir / "loss_after.npy", loss1)
        log.info(f"Loss1 - Mean: {np.mean(loss1):.4f}, Std: {np.std(loss1):.4f}")

    # Compute influence
    if get_global_rank() == 0:
        influence = loss0 - loss1
        np.save(output_dir / "influence.npy", influence)
        log.info("=" * 60)
        log.info("Final Results")
        log.info("=" * 60)
        log.info(f"Loss0 - Mean: {np.mean(loss0):.4f}, Std: {np.std(loss0):.4f}")
        log.info(f"Loss1 - Mean: {np.mean(loss1):.4f}, Std: {np.std(loss1):.4f}")
        log.info(f"Influence - Mean: {np.mean(influence):.6f}, Std: {np.std(influence):.6f}")
        log.info(f"Min influence: {np.min(influence):.6f}, Max influence: {np.max(influence):.6f}")
        log.info("=" * 60)

    log.info("Influence computation complete!")


if __name__ == "__main__":
    # Set up multiprocessing and distributed training (from train.py)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"Failed to set multiprocessing start method: {e}")

    log.info(f"Multiprocessing start method set to '{mp.get_start_method()}'")

    if torch.cuda.is_available():
        torch.cuda.set_device(f"cuda:{get_local_rank()}")
        device_as_string = f"cuda:{get_local_rank()}"
        torch.cuda.set_device(device_as_string)
        dist.init_process_group(
            backend="nccl", timeout=timedelta(minutes=30), device_id=torch.device(device_as_string)
        )
    elif torch.backends.mps.is_available():
        if not os.getenv("RANK"):
            os.environ["RANK"] = "0"
        if not os.getenv("WORLD_SIZE"):
            os.environ["WORLD_SIZE"] = "1"
        if not os.getenv("MASTER_ADDR"):
            os.environ["MASTER_ADDR"] = "0.0.0.0"
        if not os.getenv("MASTER_PORT"):
            os.environ["MASTER_PORT"] = "24501"
        dist.init_process_group(backend="gloo", timeout=timedelta(minutes=30))
    else:
        dist.init_process_group(backend="gloo", timeout=timedelta(minutes=30))

    log.info("Process group initialized")

    prepare_cli_environment()
    log.info("CLI environment prepared")

    add_cached_path_clients()

    # Load config
    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OLMoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])

    # Adjust for MPS/CPU if needed
    if torch.backends.mps.is_available():
        log.info("Device is MPS. Updating config...")
        cfg.model.init_device = "mps"
        cfg.distributed_strategy = "single"  # type: ignore

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        log.info("Device is CPU. Updating config...")
        cfg.model.init_device = "cpu"
        cfg.distributed_strategy = "single"  # type: ignore

    main(cfg)
