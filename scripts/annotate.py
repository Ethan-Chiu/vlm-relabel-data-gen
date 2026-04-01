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

  semantic     Uses pre-computed scene_graphs.parquet + semantic_annotations.parquet
               → 1 generation call → 1–3 focus-typed captions, optionally verified.
               Run extract_scene_graphs.py then extract_semantic.py first.

Usage:
    uv run python scripts/annotate.py
    uv run python scripts/annotate.py --pipeline relabel
    uv run python scripts/annotate.py --pipeline two-call
    uv run python scripts/annotate.py --pipeline scene-graph
    uv run python scripts/annotate.py --pipeline semantic
    uv run python scripts/annotate.py --pipeline semantic --verify
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
        choices=["relabel", "two-call", "robotic", "scene-graph", "semantic"],
        help="Pipeline to run (default: robotic)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Enable verification step (semantic pipeline: off by default; others: on by default)",
    )
    parser.add_argument("--no-verify", action="store_true", help="Skip verification step")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of rows to process in this run (after skipping already-done rows)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-process rows already present in annotated.parquet (default: skip them)",
    )
    args = parser.parse_args()

    cfg = datagen_config.load(args.config)
    overrides = {}
    if args.no_verify:
        overrides["verify"] = False
    elif args.verify:
        overrides["verify"] = True
    elif args.pipeline == "semantic":
        # semantic pipeline defaults to verify=False; user must opt in with --verify
        overrides["verify"] = False
    if args.limit is not None:
        overrides["annotate_limit"] = args.limit
    if args.overwrite:
        overrides["overwrite"] = True
    if overrides:
        cfg = cfg.model_copy(update=overrides)

    run(cfg, pipeline=args.pipeline)
