"""Annotation pipeline runner.

Orchestrates ProcessPoolExecutor workers, handles skip/overwrite/limit logic,
and writes results to annotated.parquet.

Annotator classes live in datagen/annotators/:
  simple.py    — SimpleAnnotator     (relabel)
  two_call.py  — TwoCallAnnotator    (two-call)
  robotic.py   — RoboticAnnotator    (robotic)
  scene.py     — CachedSceneAnnotator (scene-graph)
  semantic.py  — SemanticAnnotator   (semantic)
"""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from datagen.annotators import (
    CachedSceneAnnotator,
    RoboticAnnotator,
    SemanticAnnotator,
    SimpleAnnotator,
    TwoCallAnnotator,
)
from datagen.config import Config


# ── Worker process state ───────────────────────────────────────────────────────

_worker_annotator: (
    SimpleAnnotator | TwoCallAnnotator
    | RoboticAnnotator | CachedSceneAnnotator | SemanticAnnotator | None
) = None
_worker_img_dir: Path | None = None
_worker_pipeline: str = ""


def _worker_init(cfg: Config, pipeline: str) -> None:
    global _worker_annotator, _worker_img_dir, _worker_pipeline
    from datagen.vlm import get_backend
    backend = get_backend(cfg)
    _worker_img_dir = cfg.output_dir
    _worker_pipeline = pipeline
    if pipeline == "relabel":
        _worker_annotator = SimpleAnnotator(backend=backend, prompt=cfg.vlm_prompt)
    elif pipeline == "two-call":
        _worker_annotator = TwoCallAnnotator(backend=backend, verify=cfg.verify)
    elif pipeline == "scene-graph":
        _worker_annotator = CachedSceneAnnotator(backend=backend, verify=cfg.verify)
    elif pipeline == "semantic":
        _worker_annotator = SemanticAnnotator(backend=backend, verify=cfg.verify)
    else:  # "robotic"
        _worker_annotator = RoboticAnnotator(backend=backend, verify=cfg.verify)


def _annotate_row(row: dict) -> dict:
    try:
        img_bytes = (_worker_img_dir / row["filename"]).read_bytes()
        if _worker_pipeline == "scene-graph":
            annotations = _worker_annotator.annotate(img_bytes, row["caption"], row["scene_graph"])
        elif _worker_pipeline == "semantic":
            annotations = _worker_annotator.annotate(
                img_bytes,
                scene_graph=row["scene_graph"],
                semantic_props=row["semantic_props"],
                semantic_rels=row["semantic_rels"],
            )
        else:
            annotations = _worker_annotator.annotate(img_bytes, row["caption"])
        return {**row, **annotations}
    except Exception as e:
        # Re-raise as RuntimeError to guarantee picklability across process boundaries.
        raise RuntimeError(f"{type(e).__name__}: {e}") from None


# ── Pipeline runner ────────────────────────────────────────────────────────────

def run(cfg: Config, pipeline: str = "robotic") -> None:
    """Run the chosen annotation pipeline over metadata_path.

    pipeline: "relabel"     — single VLM call with vlm_prompt → new_caption
              "two-call"    — spatial_caption (2 VLM calls with verify)
              "robotic"     — Type A / B / C (6 VLM calls with verify)
              "scene-graph" — pre-computed scene_graph → scene Type A / B / C
              "semantic"    — pre-computed scene_graph + semantic_annotations
                              → 1-3 focus-typed captions, optionally verified
    """
    if not cfg.metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {cfg.metadata_path}. Run download.py first."
        )

    logger.info(
        f"Starting annotation | pipeline={pipeline} "
        f"| backend={cfg.vlm_backend} | concurrency={cfg.concurrency} | verify={cfg.verify}"
    )

    df = pd.read_parquet(cfg.metadata_path)
    logger.info(f"Dataset: {len(df)} rows")

    df = _join_prerequisites(df, cfg, pipeline)

    if cfg.annotate_limit is not None:
        df = df.head(cfg.annotate_limit)
        logger.info(f"Limiting to first {cfg.annotate_limit} rows")

    existing_df = _load_existing(df, cfg)

    if df.empty:
        logger.info("Nothing to process.")
        return

    records, error_counts = _run_workers(df, cfg, pipeline)

    new_df = pd.DataFrame(records)
    result_df = _merge_results(existing_df, new_df, cfg.overwrite)

    cfg.annotated_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(cfg.annotated_path, index=False)

    _log_summary(new_df, df, records, error_counts, pipeline, cfg.annotated_path)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _join_prerequisites(df: pd.DataFrame, cfg: Config, pipeline: str) -> pd.DataFrame:
    """Join metadata with scene graphs and/or semantic annotations as required."""
    if pipeline not in ("scene-graph", "semantic"):
        return df

    if not cfg.scene_graph_path.exists():
        raise FileNotFoundError(
            f"Scene graphs not found at {cfg.scene_graph_path}. "
            "Run extract_scene_graphs.py first."
        )
    sg_df = pd.read_parquet(cfg.scene_graph_path, columns=["filename", "scene_graph"])
    before = len(df)
    df = df.merge(sg_df, on="filename", how="inner")
    if len(df) < before:
        logger.warning(
            f"{before - len(df)} rows dropped — no scene graph. "
            "Re-run extract_scene_graphs.py to fill gaps."
        )

    if pipeline == "semantic":
        if not cfg.semantic_annotations_path.exists():
            raise FileNotFoundError(
                f"Semantic annotations not found at {cfg.semantic_annotations_path}. "
                "Run extract_semantic.py first."
            )
        sem_df = pd.read_parquet(
            cfg.semantic_annotations_path,
            columns=["filename", "semantic_props", "semantic_rels"],
        )
        before = len(df)
        df = df.merge(sem_df, on="filename", how="inner")
        if len(df) < before:
            logger.warning(
                f"{before - len(df)} rows dropped — no semantic annotations. "
                "Re-run extract_semantic.py to fill gaps."
            )

    logger.info(f"Joined prerequisites: {len(df)} rows ready")
    return df


def _load_existing(df: pd.DataFrame, cfg: Config) -> pd.DataFrame | None:
    """Load existing output and apply skip logic to df in-place."""
    if not cfg.annotated_path.exists():
        return None
    existing_df = pd.read_parquet(cfg.annotated_path)
    if not cfg.overwrite:
        before = len(df)
        df.drop(df[df["filename"].isin(set(existing_df["filename"]))].index, inplace=True)
        skipped = before - len(df)
        if skipped:
            logger.info(f"Skipping {skipped} already-annotated rows (overwrite=False)")
    return existing_df


def _run_workers(
    df: pd.DataFrame, cfg: Config, pipeline: str
) -> tuple[list[dict], dict[str, int]]:
    records: list[dict] = []
    error_counts: dict[str, int] = defaultdict(int)
    with ProcessPoolExecutor(
        max_workers=cfg.concurrency,
        initializer=_worker_init,
        initargs=(cfg, pipeline),
    ) as executor:
        futures = {
            executor.submit(_annotate_row, row.to_dict()): i
            for i, row in df.iterrows()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Annotating"):
            try:
                records.append(future.result())
            except Exception as e:
                error_counts[type(e).__name__] += 1
    return records, error_counts


def _merge_results(
    existing_df: pd.DataFrame | None, new_df: pd.DataFrame, overwrite: bool
) -> pd.DataFrame:
    if existing_df is None:
        return new_df
    if overwrite:
        existing_df = existing_df[
            ~existing_df["filename"].isin(set(new_df["filename"]))
        ]
    return pd.concat([existing_df, new_df], ignore_index=True)


_CAPTION_TYPES_BY_PIPELINE: dict[str, tuple[str, ...]] = {
    "relabel":     SimpleAnnotator.CAPTION_TYPES,
    "two-call":    TwoCallAnnotator.CAPTION_TYPES,
    "robotic":     RoboticAnnotator.CAPTION_TYPES,
    "scene-graph": CachedSceneAnnotator.CAPTION_TYPES,
    "semantic":    SemanticAnnotator.CAPTION_TYPES,
}


def _log_summary(
    new_df: pd.DataFrame,
    df: pd.DataFrame,
    records: list[dict],
    error_counts: dict[str, int],
    pipeline: str,
    out_path: Path,
) -> None:
    logger.info(f"Done: {len(records)}/{len(df)} succeeded → {out_path}")
    for col in _CAPTION_TYPES_BY_PIPELINE.get(pipeline, ()):
        if col not in new_df.columns:
            continue
        kept = new_df[col].notna().sum()
        logger.info(f"  {col}: {kept}/{len(new_df)} kept after verification")
    if error_counts:
        logger.warning(
            "Row errors: " + ", ".join(f"{k}: {v}" for k, v in error_counts.items())
        )
