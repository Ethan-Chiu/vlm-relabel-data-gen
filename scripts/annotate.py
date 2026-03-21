#!/usr/bin/env python
"""Run the multi-stage robotic annotation pipeline.

Produces Type A (spatial layout), Type B (object referring), and
Type C (action-conditioned) captions for each image, with optional
verification that discards spatially incorrect outputs.

Usage:
    uv run python scripts/annotate.py
    uv run python scripts/annotate.py --config configs/qwen_vllm.toml
    uv run python scripts/annotate.py --no-verify
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import datagen.config as datagen_config
from datagen.annotator import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-stage robotic VLM annotation")
    parser.add_argument("--config", default="config.toml", help="Path to config TOML file")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification step")
    args = parser.parse_args()

    cfg = datagen_config.load(args.config)
    if args.no_verify:
        cfg = cfg.model_copy(update={"verify": False})

    run(cfg)
