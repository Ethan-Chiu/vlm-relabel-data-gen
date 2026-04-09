#!/usr/bin/env python
"""Export annotated parquet to JSON.

Output fields per record: filename, original_caption, caption (the new annotation).

Usage:
    uv run python scripts/export_json.py
    uv run python scripts/export_json.py --input data/annotated.parquet --output out.json
    uv run python scripts/export_json.py --caption-col semantic_caption
    uv run python scripts/export_json.py --caption-col type_a
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Checked in priority order when --caption-col is not specified.
_CAPTION_COL_PRIORITY = [
    "new_caption", "semantic_caption", "spatial_caption",
    "type_a", "scene_type_a",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export annotated parquet to JSON")
    parser.add_argument(
        "--input", default="data/annotated.parquet",
        help="Path to annotated parquet (default: data/annotated.parquet)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: <input stem>.json)",
    )
    parser.add_argument(
        "--caption-col", default=None,
        help="Parquet column to use as the new caption (auto-detected if not specified)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(input_path)

    # Resolve new caption column.
    if args.caption_col:
        if args.caption_col not in df.columns:
            print(
                f"Error: column '{args.caption_col}' not found. "
                f"Available: {list(df.columns)}",
                file=sys.stderr,
            )
            sys.exit(1)
        caption_col = args.caption_col
    else:
        caption_col = next((c for c in _CAPTION_COL_PRIORITY if c in df.columns), None)
        if caption_col is None:
            print(
                f"Error: no caption column found. Use --caption-col to specify one. "
                f"Available: {list(df.columns)}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Auto-detected caption column: {caption_col}")

    original_col = "caption" if "caption" in df.columns else None

    records = []
    for _, row in df.iterrows():
        records.append({
            "filename": row["filename"],
            "original_caption": row[original_col] if original_col else None,
            "caption": row[caption_col] if pd.notna(row[caption_col]) else None,
        })

    output_path = Path(args.output) if args.output else input_path.with_suffix(".json")
    output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"Exported {len(records)} records → {output_path}")


if __name__ == "__main__":
    main()
