"""Scene graph extraction pipeline.

Runs RAM++ → GroundingDINO → SAM → Depth Anything V2 over every image in
metadata.parquet and writes the results to scene_graphs.parquet.

Output columns
──────────────
  filename          str    — matches metadata.parquet
  scene_graph       str    — compact text scene graph for VLM prompts
  scene_detections  str    — JSON array of detection objects for inspection

Execution model
───────────────
ProcessPoolExecutor, one worker per GPU (concurrency = num_gpus in config).
Models are loaded once per worker process via the initializer.

Use concurrency = 1 if you have a single GPU.
"""

from __future__ import annotations

import json
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
    from datagen.scene.extractor import SceneExtractor

    _worker_img_dir = cfg.output_dir
    _worker_extractor = SceneExtractor(
        ram_weights=cfg.scene_ram_weights,
        gdino_config=cfg.scene_gdino_config,
        gdino_weights=cfg.scene_gdino_weights,
        sam_weights=cfg.scene_sam_weights,
        depth_model=cfg.scene_depth_model,
        sam_type=cfg.scene_sam_type,
        box_threshold=cfg.scene_box_threshold,
        text_threshold=cfg.scene_text_threshold,
        ram_image_size=cfg.scene_ram_image_size,
        device=cfg.scene_device,
    )


def _extract_row(row: dict) -> dict:
    img_bytes = (_worker_img_dir / row["filename"]).read_bytes()
    detections, scene_graph = _worker_extractor.extract(img_bytes)

    scene_detections = json.dumps([
        {
            "label": d.label,
            "bbox": list(d.bbox),
            "confidence": round(d.confidence, 4),
            "depth_value": round(d.depth_value, 4) if d.depth_value is not None else None,
            "position": d.position,
            "depth_rank": d.depth_rank,
            "area_rank": d.area_rank,
            "free_space": d.free_space,
        }
        for d in detections
    ])

    return {
        "filename": row["filename"],
        "scene_graph": scene_graph,
        "scene_detections": scene_detections,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

def run(cfg: Config) -> None:
    """Extract scene graphs for every image in metadata_path.

    Writes scene_graphs.parquet with columns:
      filename, scene_graph, scene_detections (JSON)
    """
    if not cfg.metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {cfg.metadata_path}. Run the download pipeline first."
        )

    logger.info(
        f"Starting scene graph extraction | device={cfg.scene_device} "
        f"| concurrency={cfg.concurrency}"
    )

    df = pd.read_parquet(cfg.metadata_path)
    logger.info(f"Dataset: {len(df)} rows")

    # Skip rows already extracted if output exists
    out_path = cfg.scene_graph_path
    if out_path.exists():
        existing = pd.read_parquet(out_path, columns=["filename"])
        done = set(existing["filename"])
        remaining = df[~df["filename"].isin(done)]
        logger.info(
            f"Resuming: {len(done)} already done, {len(remaining)} remaining"
        )
        df = remaining

    if df.empty:
        logger.info("All rows already extracted.")
        return

    records = []
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
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            try:
                records.append(future.result())
            except Exception as e:
                error_counts[type(e).__name__] += 1
                logger.debug(f"Row error: {type(e).__name__}: {e}")

    if not records:
        logger.warning("No records extracted.")
        return

    result_df = pd.DataFrame(records)

    # Append to existing output if resuming
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
