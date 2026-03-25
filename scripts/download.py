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


def load_annotations(path: Path, limit: int | None) -> list[Annotation]:
    """Stream-parse a JSON array, stopping early if limit is set."""
    with path.open("rb") as f:
        items = ijson.items(f, "item")
        if limit is not None:
            items = itertools.islice(items, limit)
        return list(items)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download images and build metadata index")
    parser.add_argument("--config", default="config.toml", help="Path to config TOML file")
    parser.add_argument("--annotations", help="Path to annotations JSON file (overrides config)")
    parser.add_argument("--limit", type=int, help="Maximum number of annotations to download")
    args = parser.parse_args()

    cfg = datagen_config.load(args.config)

    annotations_path = Path(args.annotations) if args.annotations else cfg.annotations_path
    limit = args.limit if args.limit is not None else cfg.download_limit

    annotations = load_annotations(annotations_path, limit)
    run(annotations, cfg)
