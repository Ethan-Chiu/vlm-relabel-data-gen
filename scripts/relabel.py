#!/usr/bin/env python
"""Relabel image captions using a VLM backend via Ray Data.

Usage:
    # Gemini (default)
    DATAGEN_GEMINI_API_KEY=sk-... uv run python scripts/relabel.py

    # Hosted vLLM
    DATAGEN_VLM_BACKEND=vllm DATAGEN_VLM_BASE_URL=http://gpu:8000/v1 uv run python scripts/relabel.py

    # Custom config file
    uv run python scripts/relabel.py --config configs/vllm.toml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import datagen.config as datagen_config
from datagen.pipeline import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relabel image captions with a VLM")
    parser.add_argument("--config", default="config.toml", help="Path to config TOML file")
    args = parser.parse_args()

    cfg = datagen_config.load(args.config)
    run(cfg)
