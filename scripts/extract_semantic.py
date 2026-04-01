#!/usr/bin/env python
"""Extract semantic properties and relationships as an intermediate stage.

Reads scene_graphs.parquet, sends each image through a single SEMANTIC_EXTRACT
VLM call, and writes semantic_annotations.parquet.

The output is consumed by:
    uv run python scripts/annotate.py --pipeline semantic

Supports resuming: already-extracted rows are skipped automatically.

Usage:
    uv run python scripts/extract_semantic.py
    uv run python scripts/extract_semantic.py --config configs/scene_graph.toml
    uv run python scripts/extract_semantic.py --limit 200
    uv run python scripts/extract_semantic.py --concurrency 16
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import datagen.config as datagen_config
from datagen.semantic_pipeline import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract semantic annotations (pre-caption stage)")
    parser.add_argument("--config", default="config.toml", help="Path to config TOML file")
    parser.add_argument(
        "--concurrency", type=int, default=None,
        help="Number of API workers (overrides config)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N rows (already-extracted rows are still skipped)",
    )
    args = parser.parse_args()

    cfg = datagen_config.load(args.config)
    overrides = {}
    if args.concurrency is not None:
        overrides["concurrency"] = args.concurrency
    if args.limit is not None:
        overrides["annotate_limit"] = args.limit
    if overrides:
        cfg = cfg.model_copy(update=overrides)

    run(cfg)
