#!/usr/bin/env python3
"""Extract training data by instance indices (e.g., from influence computation).

This script follows the exact same approach as probe_influence_v2.py:
- Loads from the same config file (probe-influence.yaml)
- Uses IndexedDataset wrapper for consistent indexing
- Extracts data that exactly matches what probe_influence_v2.py sees
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from olmo.config import DataConfig, TrainConfig
from olmo.data import build_memmap_dataset
from olmo.exceptions import OLMoConfigurationError
from olmo.util import prepare_cli_environment


class IndexedDataset(Dataset[Dict[str, Any]]):
    """Wrapper to add index to each dataset item for tracking.

    This is the same IndexedDataset class used in probe_influence_v2.py.
    """

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


def build_indexed_dataset(cfg: TrainConfig) -> IndexedDataset:
    """Build IndexedDataset using the same method as probe_influence_v2.py.

    This ensures exact consistency with probe_influence_v2.py's data loading.
    """
    # Use data config from evaluators (same as probe_influence_v2.py)
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

    return dataset


def count_positive_by_path(cfg: TrainConfig, metrics: np.ndarray):
    """Report how many positive-metric instances come from each source file path.

    Positive is defined as ``metrics > 0`` (same as the positive-only selection branch).
    """
    data_cfg = DataConfig(
        paths=cfg.evaluators[0].data.paths,
        memmap_dtype=cfg.data.memmap_dtype,
        pad_direction=cfg.data.pad_direction,
        num_workers=cfg.data.num_workers,
        drop_last=False,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        persistent_workers=cfg.data.persistent_workers if cfg.data.num_workers > 0 else False,
        timeout=cfg.data.timeout,
    )
    dataset = build_memmap_dataset(cfg, data_cfg, include_instance_metadata=False)

    if len(metrics) != len(dataset):
        raise ValueError(f"Metrics array size ({len(metrics)}) must match dataset size ({len(dataset)})")

    positive_mask = metrics > 0
    total_positive = int(positive_mask.sum())
    total_instances = len(dataset)
    total_positive_pct = (total_positive / total_instances * 100) if total_instances > 0 else 0.0

    rows = []
    for (start, end), path in zip(dataset.offsets, dataset._memmap_paths):
        file_metrics = metrics[start:end]
        n_instances = end - start
        n_positive = int(positive_mask[start:end].sum())
        pct_in_file = (n_positive / n_instances * 100) if n_instances > 0 else 0.0
        pct_of_all_pos = (n_positive / total_positive * 100) if total_positive > 0 else 0.0
        avg_influence = float(file_metrics.mean()) if n_instances > 0 else float("nan")
        rows.append(
            (
                n_positive,
                n_instances,
                pct_in_file,
                pct_of_all_pos,
                avg_influence,
                str(path),
            )
        )

    rows.sort(key=lambda x: (-x[0], x[5]))

    print("\nPositive count by source path (metrics > 0)")
    print("=" * 160)
    print(f"Total instances: {total_instances:,}")
    print(f"Total positives: {total_positive:,} ({total_positive_pct:.2f}%)")
    print("-" * 160)
    print(
        f"{'positives':>12}  {'instances':>12}  {'pos%_file':>10}  {'share_pos%':>10}  " f"{'avg_infl':>12}  path"
    )
    print("-" * 160)
    for n_positive, n_instances, pct_in_file, pct_of_all_pos, avg_influence, path_str in rows:
        print(
            f"{n_positive:12,}  {n_instances:12,}  {pct_in_file:10.2f}  {pct_of_all_pos:10.2f}  "
            f"{avg_influence:12.6e}  {path_str}"
        )
    print("-" * 160)
    print(f"Files listed: {len(rows)}")
    print("=" * 160)


def extract_data_by_indices(
    args,
    cfg: TrainConfig,
    output_path: str,
    indices: np.ndarray | None,
    metrics: np.ndarray | None = None,
    sample_ratio: float = 0.2,
):
    """Extract training instances by their indices using IndexedDataset.

    Args:
        cfg: Training config (loaded from probe-influence.yaml)
        indices: Array of instance indices to extract
        output_path: Path to save the extracted data
        sample_ratio: Ratio of data to sample if indices is None (default: 0.2 = 1/5)
        metrics: Optional array of metrics (e.g., influence scores) to rank instances.
                 If provided, selects top sample_ratio instances by metric value.
    """
    # Use uint32 dtype (from config)
    dtype = np.uint32

    print(f"\nBuilding IndexedDataset from {cfg.evaluators[0].data.paths} ...")
    dataset = build_indexed_dataset(cfg)

    print(f"Total instances in dataset: {len(dataset):,}")
    print(f"Chunk size: {cfg.model.max_sequence_length}")
    print(f"Total tokens in dataset: {len(dataset) * cfg.model.max_sequence_length:,}")

    # Validate indices
    if indices is None:
        if metrics is not None:
            if len(metrics) != len(dataset):
                raise ValueError(f"Metrics array size ({len(metrics)}) must match dataset size ({len(dataset)})")

            # Apply positive filter if sample_ratio is -1
            if sample_ratio == -1:
                print(f"Using positive-only selection strategy (metrics > 0)...")
                positive_mask = metrics > 0
                indices = np.where(positive_mask)[0]

                if args.gumbel:
                    metrics = (metrics - metrics.mean()) / metrics.std()
                    metrics = metrics / args.temp
                    # Gumbel-Top-$k$ algorithm
                    rng = np.random.default_rng(seed=42)
                    gumbel_noise = rng.gumbel(size=len(metrics))
                    metrics += gumbel_noise
                    selection_size = len(indices)
                    indices = np.argpartition(-metrics, selection_size)[:selection_size]

                print(
                    f"Selected {len(indices):,} instances with positive metrics ({len(indices)/len(dataset)*100:.2f}% of total)"
                )
            else:
                # Select indices based on metric ranking
                print(f"Selecting top {sample_ratio:.1%} instances by metric value...")
                # Standard top-k selection
                num_to_select = int(len(dataset) * sample_ratio)
                # argsort in descending order (highest metric values first)
                sorted_indices = np.argsort(metrics)[::-1]
                indices = sorted_indices[:num_to_select]
                print(f"Selected {len(indices):,} instances with highest metric values")
                print(f"  Metric range in selection: [{metrics[indices].min():.6f}, {metrics[indices].max():.6f}]")
        else:
            # Random sampling
            print(f"No indices or metrics provided, generating random sample ({sample_ratio:.1%} of data)...")
            np.random.seed(42)
            num_random_indices = int(len(dataset) * sample_ratio)
            indices = np.random.randint(0, len(dataset), size=(num_random_indices,))
            print(f"Generated {len(indices):,} random indices")
    if np.max(indices) >= len(dataset):
        raise ValueError(f"Index {np.max(indices)} is out of bounds for dataset of size {len(dataset)}")

    # Extract instances
    print(f"\nExtracting {len(indices):,} instances...")
    extracted_tokens = []

    # Sample positions for sanity check
    sample_positions = []
    if len(indices) > 0:
        sample_positions = [0]  # First
        if len(indices) > 1:
            sample_positions.append(len(indices) // 2)  # Middle
        if len(indices) > 2:
            sample_positions.append(len(indices) - 1)  # Last

    for i, idx in enumerate(indices):
        if i % 10000 == 0:
            print(f"  Processed {i:,}/{len(indices):,} instances...")

        # Get instance using IndexedDataset (same as probe_influence_v2.py)
        item = dataset[int(idx)]
        input_ids = item["input_ids"]

        # Convert to numpy if it's a tensor
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.cpu().numpy()
        elif hasattr(input_ids, "numpy"):
            input_ids = input_ids.numpy()

        # Print sample data for sanity check
        if i in sample_positions:
            print(f"\n  Sanity check for position {i} (dataset index {idx}):")
            print(f"    Shape: {input_ids.shape}")
            print(f"    First 10 tokens: {input_ids[:10].tolist()}")
            print(f"    Last 10 tokens: {input_ids[-10:].tolist()}")
            print(f"    Token range: [{input_ids.min()}, {input_ids.max()}]")

        extracted_tokens.append(input_ids)

    # Concatenate all tokens
    print("\nConcatenating extracted tokens...")
    all_tokens = np.concatenate(extracted_tokens).astype(dtype)

    print(f"Total extracted tokens: {len(all_tokens):,}")
    print(f"Output shape: {all_tokens.shape}")

    # Handle replay data if provided
    if args.replay_data_path:
        print(f"\nLoading replay tokens from: {args.replay_data_path}")
        # Use uint32 dtype (same as output)
        replay_tokens = np.memmap(args.replay_data_path, dtype=dtype, mode="r")
        print(f"  Total replay tokens: {len(replay_tokens):,}")

        # Calculate 1/5 of current selection's token count
        seq_len = cfg.model.max_sequence_length
        current_num_instances = len(all_tokens) // seq_len
        replay_num_instances = current_num_instances // 5
        replay_sample_size = replay_num_instances * seq_len

        print(f"  Current selection: {current_num_instances:,} instances ({len(all_tokens):,} tokens)")
        print(
            f"  Sampling {replay_num_instances:,} replay instances ({replay_sample_size:,} tokens, 1/5 of current)"
        )

        # Randomly sample complete instances from replay data
        max_instances = len(replay_tokens) // seq_len
        if max_instances < replay_num_instances:
            print(f"  WARNING: Replay data has only {max_instances:,} instances. Using all of them.")
            replay_num_instances = max_instances
            replay_sample_size = replay_num_instances * seq_len

        rng = np.random.default_rng(seed=42)
        sampled_instance_indices = rng.choice(max_instances, size=replay_num_instances, replace=False)
        sampled_instance_indices = np.sort(sampled_instance_indices)  # Sort for better memory access

        # Extract full sequences
        print(f"  Extracting {replay_num_instances:,} sampled replay instances...")
        replay_sampled_list = []
        for inst_idx in sampled_instance_indices:
            start = inst_idx * seq_len
            end = start + seq_len
            replay_sampled_list.append(replay_tokens[start:end].copy())  # Copy to load into memory

        replay_sampled = np.concatenate(replay_sampled_list)
        print(f"  Extracted {len(replay_sampled):,} replay tokens")

        # Merge with current tokens
        all_tokens = np.concatenate([all_tokens, replay_sampled]).astype(dtype)
        print(f"  After merging: {len(all_tokens):,} total tokens ({len(all_tokens)//seq_len:,} instances)")

    # Save as raw binary file (same format as input)
    print(f"\nSaving to: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        all_tokens.tofile(f)

    print(f"\nDone! Extracted data saved to {output_path}")
    print(f"File size: {output_path.stat().st_size / (1024**3):.2f} GB")

    # Verify the output
    print("\nVerifying output...")
    verify_data = np.memmap(output_path, dtype=dtype, mode="r")
    print(f"Verification: Output file has {len(verify_data):,} tokens")
    assert len(verify_data) == len(
        all_tokens
    ), f"Size mismatch: expected {len(all_tokens)}, got {len(verify_data)}"
    print("Verification passed!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Extract training data by instance indices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract using probe-influence.yaml config
  python scripts/nhird/extract_by_indices.py \\
    configs/nhird/probe-influence.yaml \\
    --output /path/to/extracted_data.npy \\
    --indices-file /path/to/influence_top_indices.npy
        """,
    )

    parser.add_argument("config_path", help="Path to config file (e.g., configs/nhird/probe-influence.yaml)")
    parser.add_argument("--output", required=False, help="Path to save extracted data")
    parser.add_argument("--indices-file", required=False, help="Path to .npy file containing indices to extract")
    parser.add_argument("--metrics-file", required=False, help="Path to .npy file containing metrics")
    parser.add_argument("--sample-ratio", type=float, default=0.2, help="Ratio of data to retain (default: 0.2)")
    parser.add_argument("--gumbel", action="store_true", help="Apply Gumbel noise for stochastic selection")
    parser.add_argument("--temp", type=float, default=0.5, help="Temperature for Gumbel noise (default: 0.5)")
    parser.add_argument("--replay-data-path", help="Path to previous phase's train_ids_olmo_gumbel.npy file for replay sampling")
    parser.add_argument("--count-positive-by-path", action="store_true", help="Only report positive counts per source file path")

    args = parser.parse_args()

    # Prepare environment (same as probe_influence_v2.py)
    prepare_cli_environment()

    # Load config (same as probe_influence_v2.py)
    print(f"Loading config from: {args.config_path}")
    cfg = TrainConfig.load(args.config_path, validate_paths=False)

    # Validate config
    if not cfg.evaluators or not cfg.evaluators[0].data.paths:
        raise OLMoConfigurationError("Config must have evaluators[0].data.paths set (data to extract from)")

    # Load indices
    if args.indices_file is None:
        indices = None
    else:
        print(f"Loading indices from: {args.indices_file}")
        indices = np.load(args.indices_file)
        print(f"Loaded {len(indices):,} indices")

    # Load metrics
    if args.metrics_file is None:
        metrics = None
    else:
        print(f"Loading metrics from: {args.metrics_file}")
        metrics = np.load(args.metrics_file)
        print(f"Loaded {len(metrics):,} metric values")

    if args.count_positive_by_path:
        if metrics is None:
            raise ValueError("--count-positive-by-path requires --metrics-file")
        count_positive_by_path(cfg=cfg, metrics=metrics)
        return

    if args.output is None:
        raise ValueError("--output is required unless --count-positive-by-path is used")

    # Extract data
    extract_data_by_indices(
        args,
        cfg=cfg,
        output_path=args.output,
        indices=indices,
        metrics=metrics,
        sample_ratio=args.sample_ratio,
    )


if __name__ == "__main__":
    main()
