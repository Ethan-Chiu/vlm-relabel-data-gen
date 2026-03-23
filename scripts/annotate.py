#!/usr/bin/env python
"""Run a VLM annotation pipeline.

Pipelines
─────────
  relabel      1 VLM call with vlm_prompt from config → new_caption

  two-call     1 generate call → one spatial caption, optionally verified.

  robotic      3 parallel VLM calls → Type A (spatial layout) + Type B (referring)
               + Type C (action-conditioned), each optionally verified.  [default]

  scene-graph  Uses pre-computed scene_graphs.parquet → 3 scene-conditioned VLM calls.
               Run extract_scene_graphs.py first.

Usage:
    uv run python scripts/annotate.py
    uv run python scripts/annotate.py --pipeline relabel
    uv run python scripts/annotate.py --pipeline two-call
    uv run python scripts/annotate.py --pipeline scene-graph
    uv run python scripts/annotate.py --pipeline robotic --no-verify
    uv run python scripts/annotate.py --config configs/qwen_vllm.toml
    uv run python scripts/annotate.py --config configs/scene_graph.toml --pipeline scene-graph
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import datagen.config as datagen_config
from datagen.annotator import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM annotation pipeline")
    parser.add_argument("--config", default="config.toml", help="Path to config TOML file")
    parser.add_argument(
        "--pipeline", default="robotic",
        choices=["relabel", "two-call", "robotic", "scene-graph"],
        help="Pipeline to run (default: robotic)",
    )
    parser.add_argument("--no-verify", action="store_true", help="Skip verification step")
    args = parser.parse_args()

    cfg = datagen_config.load(args.config)
    if args.no_verify:
        cfg = cfg.model_copy(update={"verify": False})

    run(cfg, pipeline=args.pipeline)
