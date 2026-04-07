#!/usr/bin/env python
"""Merge staged shard parquet files into the canonical parquet file.

After running pipeline scripts with --num-shards > 1, each machine writes
results to data/_staging/. Run this script to merge them into the canonical
parquet and clean up the staging files.

Usage:
    uv run python scripts/merge.py --stage scene_graphs
    uv run python scripts/merge.py --stage metadata
    uv run python scripts/merge.py --all
    uv run python scripts/merge.py --all --data-dir /path/to/data

Stages (in pipeline order):
    metadata                 → metadata.parquet
    scene_graphs             → scene_graphs.parquet
    semantic_annotations     → semantic_annotations.parquet
    annotated                → annotated.parquet
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from loguru import logger

STAGES = ["metadata", "scene_graphs", "semantic_annotations", "annotated"]

# Column used to deduplicate rows across shards/runs.
_DEDUP_KEY = {
    "metadata": "source_url",
    "scene_graphs": "filename",
    "semantic_annotations": "filename",
    "annotated": "filename",
}


def merge_stage(stage: str, data_dir: Path) -> bool:
    """Merge all staging files for a stage into the canonical parquet.

    Returns True if any rows were merged, False if nothing to do.
    """
    canonical = data_dir / f"{stage}.parquet"
    staging_dir = data_dir / "_staging"
    staging_files = sorted(staging_dir.glob(f"{stage}_shard*_n*_b*.parquet")) if staging_dir.exists() else []

    if not staging_files:
        logger.info(f"{stage}: no staging files found, nothing to merge")
        return False

    logger.info(f"{stage}: found {len(staging_files)} staging file(s)")

    frames = []
    if canonical.exists():
        frames.append(pd.read_parquet(canonical))

    for f in staging_files:
        df = pd.read_parquet(f)
        logger.info(f"  {f.name}: {len(df)} rows")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Deduplicate: keep the last occurrence of each key (newest wins).
    key = _DEDUP_KEY.get(stage, "filename")
    if key in combined.columns:
        before = len(combined)
        combined = combined.drop_duplicates(subset=[key], keep="last")
        dupes = before - len(combined)
        if dupes:
            logger.info(f"  Deduplicated {dupes} duplicate rows by '{key}'")

    new_rows = len(combined) - (len(pd.read_parquet(canonical)) if canonical.exists() else 0)

    # Write atomically: write to temp file then rename.
    canonical.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mktemp(dir=canonical.parent, suffix=".parquet.tmp"))
    try:
        combined.to_parquet(tmp, index=False)
        tmp.replace(canonical)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    logger.info(f"{stage}: {new_rows:+d} rows → {canonical} ({len(combined)} total)")

    # Remove staging files only after successful write.
    for f in staging_files:
        f.unlink()
    logger.info(f"{stage}: removed {len(staging_files)} staging file(s)")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge shard staging files into canonical parquets")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--stage",
        choices=STAGES,
        help="Stage to merge",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Merge all stages in pipeline order",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing parquet files and _staging/ (default: data/)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    stages = STAGES if args.all else [args.stage]
    any_merged = False
    for stage in stages:
        try:
            merged = merge_stage(stage, data_dir)
            any_merged = any_merged or merged
        except Exception as e:
            logger.error(f"{stage}: merge failed — {e}")
            sys.exit(1)

    if not any_merged:
        logger.info("Nothing to merge.")
