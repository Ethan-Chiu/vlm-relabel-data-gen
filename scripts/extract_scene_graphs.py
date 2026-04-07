#!/usr/bin/env python
"""Extract scene graphs as an intermediate stage before VLM annotation.

Runs RAM++ → GroundingDINO → SAM → Depth Anything V2 over every image and
saves results to scene_graphs.parquet (inspectable with show_parquet.py).

The output is consumed by:
    uv run python scripts/annotate.py --pipeline scene-graph

Supports resuming: already-extracted rows are skipped automatically.

Usage:
    uv run python scripts/extract_scene_graphs.py
    uv run python scripts/extract_scene_graphs.py --config configs/scene_graph.toml
    uv run python scripts/extract_scene_graphs.py --limit 100       # first 100 rows only
    uv run python scripts/extract_scene_graphs.py --concurrency 2   # 2 GPUs
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import datagen.config as datagen_config
from datagen.scene_pipeline import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract scene graphs (pre-annotation stage)")
    parser.add_argument("--config", default="config.toml", help="Path to config TOML file")
    parser.add_argument(
        "--concurrency", type=int, default=None,
        help="Number of GPU workers (overrides config; default: use config value)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N rows of metadata (already-extracted rows are still skipped)",
    )
    parser.add_argument("--shard-id", type=int, default=0, help="Index of this shard (0-based)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    args = parser.parse_args()

    cfg = datagen_config.load(args.config)
    overrides = {}
    if args.concurrency is not None:
        overrides["concurrency"] = args.concurrency
    if args.limit is not None:
        overrides["annotate_limit"] = args.limit
    if overrides:
        cfg = cfg.model_copy(update=overrides)

    run(cfg, shard_id=args.shard_id, num_shards=args.num_shards)
