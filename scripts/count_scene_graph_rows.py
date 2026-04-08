#!/usr/bin/env python
"""Print total and valid row counts for a scene-graph parquet.

Usage:
    uv run python scripts/count_scene_graph_rows.py
    uv run python scripts/count_scene_graph_rows.py --config configs/semantic_10k.toml
    uv run python scripts/count_scene_graph_rows.py data/scene_graphs.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import datagen.config as datagen_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Count total and valid rows in a scene-graph parquet")
    parser.add_argument("--config", default="config.toml", help="Path to config TOML file")
    parser.add_argument(
        "path",
        nargs="?",
        help="Scene-graph parquet path (defaults to config scene_graph_path)",
    )
    args = parser.parse_args()

    cfg = datagen_config.load(args.config)
    path = Path(args.path) if args.path else cfg.scene_graph_path

    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(path)
    total_rows = len(df)

    if "valid" in df.columns:
        valid_rows = int(df["valid"].fillna(False).sum())
        missing_valid = int(df["valid"].isna().sum())
    else:
        valid_rows = total_rows
        missing_valid = 0

    print(f"File: {path}")
    print(f"Total rows: {total_rows}")
    print(f"Valid rows: {valid_rows}")
    if "valid" not in df.columns:
        print("Note: no valid column found, so all rows were treated as valid")
    elif missing_valid:
        print(f"Rows with missing valid flag: {missing_valid}")


if __name__ == "__main__":
    main()
