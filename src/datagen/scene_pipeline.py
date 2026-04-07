"""Scene graph extraction pipeline.

Runs RAM++ → GroundingDINO → SAM → Depth Anything V2 over every valid image in
metadata.parquet and writes the results to scene_graphs.parquet.

Output columns
──────────────
  filename          str   — matches metadata.parquet
  scene_graph       str   — compact text scene graph for VLM prompts
  scene_detections  str   — JSON array of detection objects for inspection
  n_objects         int   — unique object count after NMS deduplication
  valid             bool  — False if scene has too few objects for useful captioning

Filtering
─────────
  - Only metadata rows with valid=True are processed (or all rows if the valid
    column is absent for backward compatibility with old metadata.parquet files).
  - After extraction, scenes with n_objects < cfg.min_scene_objects are marked
    valid=False.  ALL rows (valid and invalid) are written to scene_graphs.parquet
    so the full extraction is preserved; filtered_scene.parquet additionally
    records the filenames and reasons for invalid scenes.

Execution model
───────────────
ProcessPoolExecutor, one worker per GPU (concurrency = num_gpus in config).
Models are loaded once per worker process via the initializer.

Use concurrency = 1 if you have a single GPU.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from datagen.config import Config
from datagen.scene.geometry import count_active_detections
from datagen.storage import shard_dataframe, staging_path, write_metadata


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

    dicts = [
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
    ]
    # Compute post-NMS unique object count using the same dicts already built above.
    # count_active_detections() wraps _nms_dicts() — no regex, no re-parsing.
    n_objects = count_active_detections(dicts)

    return {
        "filename": row["filename"],
        "scene_graph": scene_graph,
        "scene_detections": json.dumps(dicts),
        "n_objects": n_objects,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

def run(cfg: Config, shard_id: int = 0, num_shards: int = 1) -> None:
    """Extract scene graphs for every valid image in metadata_path.

    Writes scene_graphs.parquet with columns:
      filename, scene_graph, scene_detections (JSON), n_objects, valid

    When num_shards > 1, only rows assigned to shard_id (by md5 hash) are
    processed and results are written to a staging file. Run merge.py after
    all shards complete to combine into the canonical scene_graphs.parquet.
    """
    if not cfg.metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {cfg.metadata_path}. Run the download pipeline first."
        )

    logger.info(
        f"Starting scene graph extraction | device={cfg.scene_device} "
        f"| concurrency={cfg.concurrency}"
        + (f" | shard={shard_id}/{num_shards}" if num_shards > 1 else "")
    )

    df = pd.read_parquet(cfg.metadata_path)
    logger.info(f"Dataset: {len(df)} rows")

    # Skip images flagged as invalid during download (backward-compatible: if the
    # valid column is absent, assume all rows passed the download validity check).
    if "valid" in df.columns:
        n_before = len(df)
        df = df[df["valid"]]
        logger.info(f"Valid images: {len(df)}/{n_before}")

    if cfg.annotate_limit is not None:
        df = df.head(cfg.annotate_limit)
        logger.info(f"Limiting to first {cfg.annotate_limit} rows")

    # Skip rows already extracted — always reads canonical file.
    canonical_path = cfg.scene_graph_path
    if canonical_path.exists():
        existing = pd.read_parquet(canonical_path, columns=["filename"])
        done = set(existing["filename"])
        remaining = df[~df["filename"].isin(done)]
        logger.info(
            f"Resuming: {len(done)} already done, {len(remaining)} remaining"
        )
        df = remaining

    # Assign this shard's subset of the remaining work.
    df = shard_dataframe(df, shard_id, num_shards)
    if num_shards > 1:
        logger.info(f"Shard {shard_id}/{num_shards}: {len(df)} rows assigned")

    if df.empty:
        logger.info("All rows already extracted.")
        return

    records: list[dict] = []
    invalid_entries: list[dict] = []  # {filename, filter_reason} for filtered_scene.parquet
    error_counts: dict[str, int] = defaultdict(int)

    with ProcessPoolExecutor(
        max_workers=cfg.concurrency,
        initializer=_worker_init,
        initargs=(cfg,),
    ) as executor:
        futures = {
            executor.submit(_extract_row, row.to_dict()): row["filename"]
            for _, row in df.iterrows()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            filename = futures[future]
            try:
                result = future.result()
                n = result["n_objects"]
                if n < cfg.min_scene_objects:
                    filter_reason = (
                        "no_objects" if n == 0
                        else f"too_few_objects: {n} detected (threshold {cfg.min_scene_objects})"
                    )
                    result["valid"] = False
                    # filter_reason is NOT stored in scene_graphs.parquet — only in
                    # filtered_scene.parquet so the main output stays lean.
                    invalid_entries.append({
                        "filename": filename,
                        "filter_reason": filter_reason,
                    })
                else:
                    result["valid"] = True
                records.append(result)  # all rows (valid + invalid) go to scene_graphs.parquet
            except Exception as e:
                error_counts[type(e).__name__] += 1
                logger.debug(f"Row error ({filename}): {e}")

    # Write scene-complexity filter audit (filename + reason only).
    if invalid_entries:
        cfg.filtered_scene_path.parent.mkdir(parents=True, exist_ok=True)
        write_metadata(invalid_entries, cfg.filtered_scene_path)
        reason_counts = Counter(e["filter_reason"] for e in invalid_entries)
        for reason, count in sorted(reason_counts.items()):
            logger.info(f"  scene filter — {reason}: {count}")
        logger.info(
            f"Scene complexity: {len(invalid_entries)} filtered, "
            f"{len(records) - len(invalid_entries)} valid → {cfg.filtered_scene_path}"
        )

    if not records:
        logger.warning("No records extracted.")
        return

    result_df = pd.DataFrame(records)

    if num_shards > 1:
        # Write to staging; merge.py will combine into canonical later.
        out_path = staging_path(canonical_path, shard_id, num_shards)
    else:
        # Single-machine: append directly to canonical (original behavior).
        out_path = canonical_path
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
