"""Parquet-based metadata storage."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pandas as pd


def shard_dataframe(
    df: pd.DataFrame,
    shard_id: int,
    num_shards: int,
    key: str = "filename",
) -> pd.DataFrame:
    """Return only rows assigned to this shard by deterministic md5 hash.

    Uses hashlib.md5 (not Python hash()) so results are consistent across
    processes and machines regardless of PYTHONHASHSEED.
    """
    if num_shards <= 1:
        return df
    mask = df[key].map(
        lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % num_shards == shard_id
    )
    return df[mask]


def staging_path(canonical_path: Path, shard_id: int, num_shards: int) -> Path:
    """Return a unique staging file path for this shard run.

    Naming: {stem}_shard{shard_id}_n{num_shards}_b{timestamp}.parquet
    All staging files live in a _staging/ subdirectory next to the canonical file.
    merge.py scans this directory to detect and merge pending work.
    """
    staging_dir = canonical_path.parent / "_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    name = f"{canonical_path.stem}_shard{shard_id}_n{num_shards}_b{ts}.parquet"
    return staging_dir / name


def find_staging_files(canonical_path: Path, shard_id: int, num_shards: int) -> list[Path]:
    """Return existing staging files for this shard, sorted oldest-first.

    Matches files created by staging_path() with the same shard_id / num_shards.
    Returns an empty list if the staging directory does not exist or has no matches.
    """
    staging_dir = canonical_path.parent / "_staging"
    if not staging_dir.exists():
        return []
    pattern = f"{canonical_path.stem}_shard{shard_id}_n{num_shards}_b*.parquet"
    return sorted(staging_dir.glob(pattern))


def write_metadata(records: list[dict], path: Path) -> None:
    """Write records to a Parquet file, appending if the file already exists."""
    df_new = pd.DataFrame(records)
    if path.exists():
        df_existing = pd.read_parquet(path)
        df_new = pd.concat([df_existing, df_new], ignore_index=True)
    df_new.to_parquet(path, index=False)


def read_metadata(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
