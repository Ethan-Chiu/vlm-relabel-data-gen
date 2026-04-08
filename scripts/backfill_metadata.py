#!/usr/bin/env python
"""Backfill download metadata fields for older metadata parquet files.

This script is intended for old ``metadata.parquet`` files that predate newer
download-time fields like ``valid``. It inspects the image file on disk and
fills only missing fields, leaving existing populated values unchanged.

Usage:
    uv run python scripts/backfill_metadata.py
    uv run python scripts/backfill_metadata.py --config configs/download.toml
    uv run python scripts/backfill_metadata.py --input old.parquet --output fixed.parquet
    uv run python scripts/backfill_metadata.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import datagen.config as datagen_config
from datagen.filter import FilterResult, check_image_validity
from datagen.storage import write_metadata

REQUIRED_COLUMNS = ("width", "height", "mode", "valid")


def _missing_value(value: object, column: str) -> bool:
    if pd.isna(value):
        return True
    if column == "mode":
        return value == ""
    if column in {"width", "height"}:
        return value == 0
    return False


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            if column == "mode":
                df[column] = ""
            else:
                df[column] = pd.NA
    return df


def _row_needs_backfill(row: pd.Series) -> bool:
    return any(_missing_value(row[column], column) for column in REQUIRED_COLUMNS)


def _inspect_image(img_path: Path, min_pixels: int) -> tuple[dict[str, object], FilterResult | None]:
    from PIL import Image, UnidentifiedImageError

    values: dict[str, object] = {"width": 0, "height": 0, "mode": "", "valid": False}

    try:
        with Image.open(img_path) as img:
            rgb = img.convert("RGB")
            values["width"] = rgb.width
            values["height"] = rgb.height
            values["mode"] = rgb.mode
    except (UnidentifiedImageError, OSError, Exception):
        pass

    invalid = check_image_validity(img_path, min_pixels)
    values["valid"] = invalid is None
    return values, invalid


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill missing fields in metadata parquet")
    parser.add_argument("--config", default="config.toml", help="Path to config TOML file")
    parser.add_argument("--input", help="Metadata parquet to read (defaults to config metadata_path)")
    parser.add_argument("--output", help="Where to write the updated parquet (defaults to overwrite input)")
    parser.add_argument(
        "--limit",
        type=int,
        help="Only process the first N rows that need backfill (useful for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect rows and print counts without writing changes",
    )
    args = parser.parse_args()

    cfg = datagen_config.load(args.config)
    input_path = Path(args.input) if args.input else cfg.metadata_path
    output_path = Path(args.output) if args.output else input_path

    df = pd.read_parquet(input_path)
    df = _ensure_columns(df)

    missing_mask = df.apply(_row_needs_backfill, axis=1)
    target_indices = df.index[missing_mask]
    if args.limit is not None:
        target_indices = target_indices[: args.limit]

    print(f"Loaded {len(df)} rows from {input_path}")
    print(f"Rows needing backfill: {len(target_indices)}")

    if len(target_indices) == 0:
        if not args.dry_run and output_path != input_path:
            df.to_parquet(output_path, index=False)
            print(f"No changes needed; copied schema-normalized parquet to {output_path}")
        return

    updated_rows = 0
    invalid_rows = 0
    missing_files = 0
    invalid_entries: list[dict[str, str]] = []

    for idx in target_indices:
        filename = df.at[idx, "filename"]
        img_path = cfg.output_dir / filename
        if not img_path.exists():
            missing_files += 1

        valid_was_missing = _missing_value(df.at[idx, "valid"], "valid")
        inspected, invalid = _inspect_image(img_path, cfg.min_image_pixels)
        changed = False

        for column, value in inspected.items():
            if _missing_value(df.at[idx, column], column):
                df.at[idx, column] = value
                changed = True

        if changed:
            updated_rows += 1
        if inspected["valid"] is False:
            invalid_rows += 1
            if valid_was_missing and invalid is not None:
                invalid_entries.append(
                    {
                        "filename": filename,
                        "filter_reason": f"{invalid.reason}: {invalid.details}",
                    }
                )

    print(f"Rows updated: {updated_rows}")
    print(f"Invalid rows found during backfill: {invalid_rows}")
    print(f"Missing image files encountered: {missing_files}")

    if args.dry_run:
        return

    df.to_parquet(output_path, index=False)
    print(f"Wrote updated metadata to {output_path}")

    if invalid_entries:
        if cfg.filtered_download_path.exists():
            existing_invalid = pd.read_parquet(cfg.filtered_download_path, columns=["filename"])
            known_invalid = set(existing_invalid["filename"].tolist())
            invalid_entries = [row for row in invalid_entries if row["filename"] not in known_invalid]

        if not invalid_entries:
            print("No new invalid audit rows needed")
            return

        write_metadata(invalid_entries, cfg.filtered_download_path)
        print(f"Appended {len(invalid_entries)} invalid entries to {cfg.filtered_download_path}")


if __name__ == "__main__":
    main()
