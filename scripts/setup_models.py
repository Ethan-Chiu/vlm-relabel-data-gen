#!/usr/bin/env python
"""Download model weights for the scene-graph pipeline.

Downloads:
  - RAM++ (ram_plus_swin_large_14m.pth)
  - GroundingDINO (groundingdino_swint_ogc.pth + config)
  - SAM vit_h (sam_vit_h_4b8939.pth)
  - Depth Anything V2 Large (via HuggingFace — cached automatically at runtime)

Usage:
    uv run python scripts/setup_models.py
    uv run python scripts/setup_models.py --models-dir /data/models
    uv run python scripts/setup_models.py --skip-depth   # skip HF cache step
"""

import argparse
import sys
import urllib.request
from pathlib import Path

MODELS = {
    "ram_plus": {
        "url": "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth",
        "filename": "ram_plus_swin_large_14m.pth",
    },
    "gdino_weights": {
        "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        "filename": "groundingdino_swint_ogc.pth",
    },
    "gdino_config": {
        "url": "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "filename": "GroundingDINO_SwinT_OGC.cfg.py",
    },
    "sam_vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "filename": "sam_vit_h_4b8939.pth",
    },
}

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return
    print(f"  Downloading {dest.name} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  [ok] {dest.name} ({dest.stat().st_size // 1_048_576} MB)")
    except Exception as e:
        print(f"  [error] {dest.name}: {e}", file=sys.stderr)


def _prefetch_depth(model_id: str) -> None:
    print(f"\nPre-fetching Depth Anything V2 from HuggingFace: {model_id}")
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        AutoImageProcessor.from_pretrained(model_id)
        AutoModelForDepthEstimation.from_pretrained(model_id)
        print("  [ok] Depth model cached")
    except Exception as e:
        print(f"  [error] Could not prefetch depth model: {e}", file=sys.stderr)
        print("  It will be downloaded automatically on first use.", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download scene-graph pipeline model weights")
    parser.add_argument(
        "--models-dir", default="models",
        help="Directory to save model weights (default: models/)",
    )
    parser.add_argument(
        "--skip-depth", action="store_true",
        help="Skip pre-fetching the Depth Anything V2 model (it will download on first use)",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading weights → {models_dir}/")
    for key, info in MODELS.items():
        _download(info["url"], models_dir / info["filename"])

    if not args.skip_depth:
        _prefetch_depth(DEPTH_MODEL)

    print("\nDone. Update config.toml if you used a custom --models-dir.")
    print("Example scene-graph config: configs/scene_graph.toml")


if __name__ == "__main__":
    main()
