#!/usr/bin/env python
"""Inspect a Parquet file.

Usage:
    uv run python scripts/inspect.py                          # default: data/relabeled.parquet
    uv run python scripts/inspect.py data/metadata.parquet
    uv run python scripts/inspect.py data/relabeled.parquet --cols filename new_caption
    uv run python scripts/inspect.py data/relabeled.parquet --head 10
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a Parquet file")
    parser.add_argument("path", nargs="?", default="data/relabeled.parquet", help="Path to Parquet file")
    parser.add_argument("--cols", nargs="+", help="Only show these columns")
    parser.add_argument("--head", type=int, help="Show only first N rows")
    parser.add_argument("--info", action="store_true", help="Show schema and memory usage instead of rows")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(path)

    if args.info:
        print(f"File:    {path}")
        print(f"Rows:    {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
        print()
        df.info()
        return

    if args.cols:
        missing = [c for c in args.cols if c not in df.columns]
        if missing:
            print(f"Unknown columns: {missing}. Available: {df.columns.tolist()}", file=sys.stderr)
            sys.exit(1)
        df = df[args.cols]

    if args.head:
        df = df.head(args.head)

    pd.set_option("display.max_colwidth", 120)
    pd.set_option("display.max_rows", None)
    print(df.to_string())


if __name__ == "__main__":
    main()
