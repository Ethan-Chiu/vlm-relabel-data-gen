#!/usr/bin/env python
"""Download images and build the metadata index.

Usage:
    uv run python scripts/download.py --config configs/download.toml
    uv run python scripts/download.py --config configs/download.toml --annotations /path/to/data.json
"""

import argparse
import itertools
import sys
from pathlib import Path

import ijson

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import datagen.config as datagen_config
from datagen.download import Annotation, run


def load_annotations(
    path: Path,
    limit: int | None,
    shard_id: int = 0,
    num_shards: int = 1,
) -> list[tuple[int, Annotation]]:
    """Stream-parse a JSON array, sharding by stream index.

    Returns a list of (global_index, annotation) pairs.  The global_index is
    the position of the annotation in the full (unsharded) JSON array and is
    used by run() to derive stable, collision-free filenames across shards.

    Shard assignment is by index (i % num_shards == shard_id) so each machine
    owns a deterministic, disjoint slice.  The limit is applied after sharding.
    """
    with path.open("rb") as f:
        items = ijson.items(f, "item")
        sharded = (
            (i, ann)
            for i, ann in enumerate(items)
            if i % num_shards == shard_id
        )
        if limit is not None:
            sharded = itertools.islice(sharded, limit)
        return list(sharded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download images and build metadata index")
    parser.add_argument("--config", default="config.toml", help="Path to config TOML file")
    parser.add_argument("--annotations", help="Path to annotations JSON file (overrides config)")
    parser.add_argument("--limit", type=int, help="Maximum number of annotations to download")
    parser.add_argument("--shard-id", type=int, default=0, help="Index of this shard (0-based)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument(
        "--skip-staging", action="store_true",
        help=(
            "Also skip URLs already present in this shard's existing staging files. "
            "Use when rerunning a failed shard to avoid re-downloading already-fetched images."
        ),
    )
    args = parser.parse_args()

    cfg = datagen_config.load(args.config)

    annotations_path = Path(args.annotations) if args.annotations else cfg.annotations_path
    limit = args.limit if args.limit is not None else cfg.download_limit

    annotations = load_annotations(annotations_path, limit, shard_id=args.shard_id, num_shards=args.num_shards)

    output_path = None
    if args.num_shards > 1:
        from datagen.storage import staging_path
        output_path = staging_path(cfg.metadata_path, args.shard_id, args.num_shards)

    additional_skip_urls = None
    if args.skip_staging and args.num_shards > 1:
        import pandas as pd
        staging_dir = cfg.metadata_path.parent / "_staging"
        pattern = f"{cfg.metadata_path.stem}_shard{args.shard_id}_n{args.num_shards}_b*.parquet"
        staging_files = sorted(staging_dir.glob(pattern)) if staging_dir.exists() else []
        if staging_files:
            additional_skip_urls = set()
            for f in staging_files:
                df = pd.read_parquet(f, columns=["source_url"])
                additional_skip_urls.update(df["source_url"].tolist())
            print(f"--skip-staging: {len(staging_files)} staging file(s), {len(additional_skip_urls)} URLs to skip")

    run(annotations, cfg, output_path=output_path, additional_skip_urls=additional_skip_urls)
