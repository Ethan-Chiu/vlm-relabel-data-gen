#!/usr/bin/env python
"""Re-sort scene_graph text in scene_graphs.parquet by depth_rank (nearest first).

Run this once after upgrading to the depth-sorted build_scene_graph(). It rebuilds
the scene_graph text column from the existing scene_detections JSON without
re-running the GPU pipeline.

Usage:
    uv run python scripts/rebuild_scene_graphs.py
    uv run python scripts/rebuild_scene_graphs.py data/scene_graphs.parquet
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Add src/ to path so datagen is importable without install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datagen.scene.geometry import build_scene_graph_from_dicts


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-sort scene_graph text by depth_rank")
    parser.add_argument(
        "path", nargs="?", default="data/scene_graphs.parquet", help="Path to parquet"
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} rows from {path}")

    def rebuild(row: pd.Series) -> str:
        try:
            dets = json.loads(row["scene_detections"])
            return build_scene_graph_from_dicts(dets)
        except Exception as e:
            print(f"  Warning: could not rebuild {row['filename']}: {e}")
            return row["scene_graph"]  # keep original on error

    df["scene_graph"] = df.apply(rebuild, axis=1)
    df.to_parquet(path, index=False)
    print(f"Done — scene_graph text re-sorted by depth_rank → {path}")


if __name__ == "__main__":
    main()
