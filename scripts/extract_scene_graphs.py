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
    args = parser.parse_args()

    cfg = datagen_config.load(args.config)
    if args.concurrency is not None:
        cfg = cfg.model_copy(update={"concurrency": args.concurrency})

    run(cfg)
