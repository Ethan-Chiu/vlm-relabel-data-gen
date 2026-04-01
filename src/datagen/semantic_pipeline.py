"""Semantic extraction pipeline (Stage 1 of the semantic annotation pipeline).

Reads scene_graphs.parquet (produced by extract_scene_graphs.py), sends each
image + scene_graph + scene_detections through a single SEMANTIC_EXTRACT VLM
call, and writes the structured results to semantic_annotations.parquet.

Output columns
──────────────
  filename         str  — matches scene_graphs.parquet / metadata.parquet
  semantic_props   str  — JSON: {scene_context, objects}
  semantic_rels    str  — JSON: {relationships}

Execution model
───────────────
ProcessPoolExecutor — one process per API concurrency slot (same model as
annotator.py).  The VLM backend is initialised once per worker process via
the initializer so connections / model state are not re-created per row.

Usage (via extract_semantic.py)
────────────────────────────────
  uv run python scripts/extract_semantic.py
  uv run python scripts/extract_semantic.py --limit 200
  uv run python scripts/extract_semantic.py --concurrency 16
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from datagen.config import Config


# ── Worker process state ───────────────────────────────────────────────────────

_worker_extractor = None
_worker_img_dir: Path | None = None


def _worker_init(cfg: Config) -> None:
    global _worker_extractor, _worker_img_dir
    from datagen.semantic.extractor import SemanticExtractor
    from datagen.vlm import get_backend

    _worker_img_dir = cfg.output_dir
    _worker_extractor = SemanticExtractor(backend=get_backend(cfg))


def _extract_row(row: dict) -> dict:
    img_bytes = (_worker_img_dir / row["filename"]).read_bytes()
    try:
        annotation = _worker_extractor.extract(
            img_bytes,
            scene_graph=row["scene_graph"],
            scene_detections=row["scene_detections"],
        )
    except Exception as e:
        raise RuntimeError(f"{type(e).__name__}: {e}") from None

    return {
        "filename": row["filename"],
        "semantic_props": annotation.props_to_json(),
        "semantic_rels": annotation.rels_to_json(),
    }


# ── Entry point ────────────────────────────────────────────────────────────────

def run(cfg: Config) -> None:
    """Extract semantic annotations for all rows in scene_graphs.parquet.

    Writes semantic_annotations.parquet with columns:
      filename, semantic_props, semantic_rels
    Supports resuming: already-extracted rows are skipped automatically.
    """
    if not cfg.scene_graph_path.exists():
        raise FileNotFoundError(
            f"Scene graphs not found at {cfg.scene_graph_path}. "
            "Run extract_scene_graphs.py first."
        )

    logger.info(
        f"Starting semantic extraction | backend={cfg.vlm_backend} "
        f"| concurrency={cfg.concurrency}"
    )

    df = pd.read_parquet(cfg.scene_graph_path)
    logger.info(f"Scene graphs loaded: {len(df)} rows")

    if cfg.annotate_limit is not None:
        df = df.head(cfg.annotate_limit)
        logger.info(f"Limiting to first {cfg.annotate_limit} rows")

    out_path = cfg.semantic_annotations_path
    if out_path.exists():
        existing = pd.read_parquet(out_path, columns=["filename"])
        done = set(existing["filename"])
        df = df[~df["filename"].isin(done)]
        logger.info(f"Resuming: {len(done)} already done, {len(df)} remaining")

    if df.empty:
        logger.info("All rows already extracted.")
        return

    records: list[dict] = []
    error_counts: dict[str, int] = defaultdict(int)

    with ProcessPoolExecutor(
        max_workers=cfg.concurrency,
        initializer=_worker_init,
        initargs=(cfg,),
    ) as executor:
        futures = {
            executor.submit(_extract_row, row.to_dict()): i
            for i, row in df.iterrows()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting semantics"):
            try:
                records.append(future.result())
            except Exception as e:
                error_counts[type(e).__name__] += 1
                logger.debug(f"Row error: {e}")

    if not records:
        logger.warning("No records extracted.")
        return

    result_df = pd.DataFrame(records)

    if out_path.exists():
        existing_df = pd.read_parquet(out_path)
        result_df = pd.concat([existing_df, result_df], ignore_index=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(out_path, index=False)

    logger.info(f"Done: {len(records)} extracted → {out_path}")
    if error_counts:
        logger.warning(
            "Row errors: " + ", ".join(f"{k}: {v}" for k, v in error_counts.items())
        )
