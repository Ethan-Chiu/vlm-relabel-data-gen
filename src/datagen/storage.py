"""Parquet-based metadata storage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_metadata(records: list[dict], path: Path) -> None:
    """Write records to a Parquet file, appending if the file already exists."""
    df_new = pd.DataFrame(records)
    if path.exists():
        df_existing = pd.read_parquet(path)
        df_new = pd.concat([df_existing, df_new], ignore_index=True)
    df_new.to_parquet(path, index=False)


def read_metadata(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
