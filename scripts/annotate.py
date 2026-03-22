#!/usr/bin/env python
"""Run a robotic annotation pipeline.

Pipelines
─────────
  robotic   3 parallel VLM calls → Type A (spatial layout) + Type B (referring)
            + Type C (action-conditioned), each optionally verified.  [default]

  two-call  1 generate call → one spatial caption, optionally verified.

Usage:
    uv run python scripts/annotate.py
    uv run python scripts/annotate.py --pipeline two-call
    uv run python scripts/annotate.py --pipeline robotic --no-verify
    uv run python scripts/annotate.py --config configs/qwen_vllm.toml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import datagen.config as datagen_config
from datagen.annotator import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robotic VLM annotation pipeline")
    parser.add_argument("--config", default="config.toml", help="Path to config TOML file")
    parser.add_argument(
        "--pipeline", default="robotic", choices=["robotic", "two-call"],
        help="Pipeline to run (default: robotic)",
    )
    parser.add_argument("--no-verify", action="store_true", help="Skip verification step")
    args = parser.parse_args()

    cfg = datagen_config.load(args.config)
    if args.no_verify:
        cfg = cfg.model_copy(update={"verify": False})

    run(cfg, pipeline=args.pipeline)
