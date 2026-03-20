#!/usr/bin/env python
"""Download images and build the metadata index.

Usage:
    uv run python scripts/download.py
    uv run python scripts/download.py --config my_config.toml
"""

import argparse
import sys
from pathlib import Path

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import datagen.config as datagen_config
from datagen.download import Annotation, run

# ---------------------------------------------------------------------------
# Sample annotations — replace with your real data source
# (e.g. load from a JSON file, a database, or a HuggingFace dataset)
# ---------------------------------------------------------------------------
ANNOTATIONS: list[Annotation] = [
    {
        "caption": "the tiki ball logo is displayed on this apron",
        "url": "https://image.spreadshirtmedia.com/image-server/v1/products/P1016953357T1186A359PC1026849373PA2537PT17X1Y0S28/views/1,width=300,height=300,appearanceId=359,version=1497265829/z-tiki-bar-adjustable-apron.png",
    },
    {
        "caption": "the view of an aerial pool with palm trees",
        "url": "http://uberflip.cdntwrk.com/mediaproxy?url=http%3A%2F%2Fd22ir9aoo7cbf6.cloudfront.net%2Fwp-content%2Fuploads%2Fsites%2F4%2F2018%2F01%2FAyana1.jpg&size=1&version=1517393441&sig=e3e14f09e2b062e0fd144306d56abda7&default=hubs%2Ftilebg-blogs.jpg",
    },
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download images and build metadata index")
    parser.add_argument("--config", default="config.toml", help="Path to config TOML file")
    args = parser.parse_args()

    cfg = datagen_config.load(args.config)
    run(ANNOTATIONS, cfg)
