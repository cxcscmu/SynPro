#!/usr/bin/env python3
"""Convert DCLM token .npy files to text JSONL chunks.

Reads data paths from a training config (e.g. configs/dclm/OLMo-400M_2x.yaml),
decodes token chunks with the configured tokenizer, and writes JSONL where each
line is {"text": "..."} for a fixed chunk size (default 2048).
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import yaml
from tqdm import tqdm

from olmo.tokenizer import Tokenizer
from olmo_data import get_data_path, is_data_file


def _replace_local_root(path_str: str, local_root: str | None) -> str:
    if local_root is None:
        return path_str
    return path_str.replace("${LOCAL_ROOT}", local_root)


def _load_paths_from_config(config_path: Path, local_root: str | None) -> List[Path]:
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    data = config.get("data", {})
    paths = data.get("paths", [])
    if not paths:
        raise ValueError(f"No data.paths found in {config_path}")

    out: List[Path] = []
    for path_str in paths:
        resolved = _replace_local_root(str(path_str), local_root)
        out.append(Path(resolved))
    return out


def _load_tokenizer_from_config(config_path: Path) -> Tokenizer:
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    tokenizer_cfg = config.get("tokenizer", {})
    model_cfg = config.get("model", {})
    identifier = tokenizer_cfg.get("identifier")
    if not identifier:
        raise ValueError(f"tokenizer.identifier not found in {config_path}")

    eos_token_id = model_cfg.get("eos_token_id")
    pad_token_id = model_cfg.get("pad_token_id")

    if Path(identifier).is_file():
        return Tokenizer.from_file(identifier, eos_token_id=eos_token_id, pad_token_id=pad_token_id)
    if is_data_file(identifier):
        with get_data_path(identifier) as tokenizer_path:
            return Tokenizer.from_file(tokenizer_path, eos_token_id=eos_token_id, pad_token_id=pad_token_id)
    return Tokenizer.from_pretrained(identifier, eos_token_id=eos_token_id, pad_token_id=pad_token_id)


def _default_output_path(input_path: Path, output_root: Path | None, output_ext: str) -> Path:
    if output_root is not None:
        # Preserve relative path after the tokenizer directory, if present.
        parts = list(input_path.parts)
        if "dolma2-tokenizer" in parts:
            idx = parts.index("dolma2-tokenizer")
            rel = Path(*parts[idx + 1 :])
        else:
            rel = input_path.name
            rel = Path(rel)
        out_path = output_root / rel
    else:
        # Replace tokenizer directory with "text".
        out_path = Path(str(input_path).replace("dolma2-tokenizer", "text"))

    return out_path.with_suffix(output_ext)


def _apply_suffix(path: Path, suffix: str) -> Path:
    if not suffix:
        return path
    if path.stem.endswith(suffix):
        return path
    return path.with_name(path.stem + suffix + path.suffix)


def _map_tokens_to_text_path(input_path: Path, output_root: Path | None, output_ext: str, suffix: str) -> Path:
    out_path = _default_output_path(input_path, output_root, output_ext)
    return _apply_suffix(out_path, suffix)


def _map_text_to_tokens_path(input_path: Path, output_root: Path | None, suffix: str) -> Path:
    if output_root is not None:
        parts = list(input_path.parts)
        if "text" in parts:
            idx = parts.index("text")
            rel = Path(*parts[idx + 1 :])
        else:
            rel = Path(input_path.name)
        out_path = output_root / rel
    else:
        out_path = Path(str(input_path).replace("text", "dolma2-tokenizer"))

    out_path = out_path.with_suffix(".npy")
    return _apply_suffix(out_path, suffix)


def _iter_chunks(data: np.memmap, chunk_size: int) -> Iterable[np.ndarray]:
    num_chunks = len(data) // chunk_size
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        yield data[start:end]


def convert_one(
    input_path: Path,
    output_path: Path,
    tokenizer: Tokenizer,
    chunk_size: int,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.memmap(input_path, dtype=np.uint32, mode="r")
    total_chunks = len(data) // chunk_size

    with output_path.open("w", encoding="utf-8") as f:
        for idx, chunk in enumerate(
            tqdm(
                _iter_chunks(data, chunk_size),
                total=total_chunks,
                desc=input_path.name,
                unit="chunk",
            )
        ):
            text = tokenizer.decode(chunk.tolist(), skip_special_tokens=False)
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"Wrote {total_chunks} chunks to {output_path}")


def _convert_one_worker(args: Tuple[str, str, str, int]) -> None:
    input_path_str, output_path_str, config_path_str, chunk_size = args
    tokenizer = _load_tokenizer_from_config(Path(config_path_str))
    convert_one(Path(input_path_str), Path(output_path_str), tokenizer, chunk_size)


def _encode_one_worker(args: Tuple[str, str, str]) -> None:
    input_jsonl_str, output_npy_str, config_path_str = args
    tokenizer = _load_tokenizer_from_config(Path(config_path_str))
    encode_jsonl_to_npy(Path(input_jsonl_str), Path(output_npy_str), tokenizer)


def encode_jsonl_to_npy(input_jsonl: Path, output_npy: Path, tokenizer: Tokenizer) -> None:
    if output_npy.exists():
        print(f"Output already exists, skipping: {output_npy}")
        return
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input not found: {input_jsonl}")
    output_npy.parent.mkdir(parents=True, exist_ok=True)

    with input_jsonl.open("r", encoding="utf-8") as reader, output_npy.open("wb") as writer:
        all_ids = []
        for line in tqdm(reader, desc=input_jsonl.name, unit="line"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_ids.append(ids)
        all_ids = np.concatenate(all_ids)
        np.array(all_ids, dtype=np.uint32).tofile(writer)

    print(f"Wrote tokens to {output_npy}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DCLM token files to text JSONL.")
    parser.add_argument(
        "--config",
        required=False,
        help="Training config with data.paths (e.g., configs/dclm/OLMo-400M_2x.yaml).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Chunk size in tokens (default: 2048).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output root. If set, paths are mirrored under this root.",
    )
    parser.add_argument(
        "--output-ext",
        default=".jsonl",
        help="Output file extension (default: .jsonl).",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix to append before file extension (e.g., _sr0.05_ri0.0_rs0.0_rd0.0_n1).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1).",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Optional limit on number of files to process.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Encode jsonl back to a token .npy (raw uint32).",
    )
    parser.add_argument("--input-jsonl", default=None, help="Input jsonl path for --reverse.")
    parser.add_argument("--output-npy", default=None, help="Output .npy path for --reverse.")
    args = parser.parse_args()

    if args.reverse:
        if not args.config:
            parser.error("--reverse requires --config")
        config_path = Path(args.config)
        tokenizer = _load_tokenizer_from_config(config_path)

        if args.input_jsonl:
            input_jsonl = Path(args.input_jsonl)
            output_npy = (
                Path(args.output_npy)
                if args.output_npy
                else _map_text_to_tokens_path(input_jsonl, args.output_root, args.suffix)
            )
            encode_jsonl_to_npy(input_jsonl, output_npy, tokenizer)
            return

        local_root = os.environ.get("LOCAL_ROOT")
        output_root = Path(args.output_root) if args.output_root else None
        paths = _load_paths_from_config(config_path, local_root)
        jobs_rev: List[Tuple[str, str, str]] = []
        for token_path in paths:
            input_jsonl = _map_tokens_to_text_path(token_path, output_root, ".jsonl", args.suffix)
            output_npy = _apply_suffix(token_path, args.suffix)
            jobs_rev.append((str(input_jsonl), str(output_npy), str(config_path)))

        if args.num_workers <= 1:
            for input_jsonl_str, output_npy_str, _config_path_str in jobs_rev:
                encode_jsonl_to_npy(Path(input_jsonl_str), Path(output_npy_str), tokenizer)
        else:
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                list(
                    tqdm(
                        executor.map(_encode_one_worker, jobs_rev),
                        total=len(jobs_rev),
                        desc="files",
                        unit="file",
                    )
                )
        return

    if not args.config:
        parser.error("--config is required unless --reverse is set")

    config_path = Path(args.config)
    local_root = os.environ.get("LOCAL_ROOT")
    output_root = Path(args.output_root) if args.output_root else None

    paths = _load_paths_from_config(config_path, local_root)
    if args.limit_files is not None:
        paths = paths[: args.limit_files]

    jobs: List[Tuple[str, str, str, int]] = []
    for input_path in paths:
        output_path = _map_tokens_to_text_path(input_path, output_root, args.output_ext, args.suffix)
        jobs.append((str(input_path), str(output_path), str(config_path), args.chunk_size))

    if args.num_workers <= 1:
        tokenizer = _load_tokenizer_from_config(config_path)
        for input_path_str, output_path_str, _config_path_str, chunk_size in jobs:
            convert_one(Path(input_path_str), Path(output_path_str), tokenizer, chunk_size)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            list(
                tqdm(
                    executor.map(_convert_one_worker, jobs),
                    total=len(jobs),
                    desc="files",
                    unit="file",
                )
            )


if __name__ == "__main__":
    main()
