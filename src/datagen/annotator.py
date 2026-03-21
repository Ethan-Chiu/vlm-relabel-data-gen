"""Multi-stage robotic annotation pipeline.

For each image the annotator runs three VLM calls in parallel:
  A — spatial scene description
  B — object-centric referring expressions
  C — action-conditioned caption

Optionally, a fourth verification call checks each output for spatial
errors and discards any caption that fails (sets it to None).

                     ┌─ call A ─┐
image + caption ─────┼─ call B ─┼──► [verify A, B, C in parallel] ──► result dict
                     └─ call C ─┘

The inner ThreadPoolExecutor is created once per annotator instance (i.e.
once per worker process) and reused across all rows — no per-row overhead.
"""

from __future__ import annotations

import textwrap
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from datagen import prompts
from datagen.config import Config
from datagen.vlm.base import VLMBackend

# Caption types produced by this pipeline
CAPTION_TYPES = ("type_a", "type_b", "type_c")

# ── Worker process state (module-level, initialized once per process) ─────────
_worker_annotator: "RoboticAnnotator | None" = None
_worker_img_dir: Path | None = None
_worker_cfg: Config | None = None


def _worker_init(cfg: Config) -> None:
    global _worker_annotator, _worker_img_dir, _worker_cfg
    from datagen.vlm import get_backend
    _worker_cfg = cfg
    _worker_img_dir = cfg.output_dir
    _worker_annotator = RoboticAnnotator(backend=get_backend(cfg), verify=cfg.verify)


def _annotate_row(row: dict) -> dict:
    img_bytes = (_worker_img_dir / row["filename"]).read_bytes()
    annotations = _worker_annotator.annotate(img_bytes, row["caption"])
    return {**row, **annotations}


# ── RoboticAnnotator ──────────────────────────────────────────────────────────

class RoboticAnnotator:
    """Runs the 3-stage annotation pipeline for a single image.

    Designed to be instantiated once per worker process and reused across rows.
    """

    def __init__(self, backend: VLMBackend, verify: bool = True) -> None:
        self._backend = backend
        self._verify = verify
        # Persistent pool: 3 threads for A/B/C calls, reused across all rows
        self._pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="vlm")

    def annotate(self, image_bytes: bytes, original_caption: str) -> dict:
        """Return dict with keys type_a, type_b, type_c (None if discarded)."""
        results = self._run_parallel({
            "type_a": prompts.TYPE_A.format(original_caption=original_caption),
            "type_b": prompts.TYPE_B.format(original_caption=original_caption),
            "type_c": prompts.TYPE_C.format(original_caption=original_caption),
        }, image_bytes)

        if self._verify:
            results = self._run_verification(image_bytes, results)

        return results

    def _run_parallel(self, prompt_map: dict[str, str], image_bytes: bytes) -> dict:
        """Submit all prompts concurrently, collect results."""
        futures = {
            key: self._pool.submit(self._backend.call, image_bytes, prompt)
            for key, prompt in prompt_map.items()
        }
        results = {}
        for key, future in futures.items():
            try:
                results[key] = future.result()
            except Exception as e:
                logger.warning(f"{key} failed: {type(e).__name__}: {e}")
                results[key] = None
        return results

    def _run_verification(self, image_bytes: bytes, results: dict) -> dict:
        """Run verification for all non-None captions in parallel.
        Discard (set to None) any caption the model flags as spatially wrong.
        """
        to_verify = {k: v for k, v in results.items() if v is not None}
        verify_futures = {
            key: self._pool.submit(
                self._backend.call,
                image_bytes,
                prompts.VERIFY.format(caption=caption),
            )
            for key, caption in to_verify.items()
        }
        for key, future in verify_futures.items():
            try:
                answer = future.result().strip().upper()
                if answer.startswith("YES"):
                    logger.debug(f"{key} discarded by verification")
                    results[key] = None
            except Exception as e:
                logger.warning(f"Verification for {key} failed: {type(e).__name__}: {e}")
        return results

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run(cfg: Config) -> None:
    """Annotate all rows in metadata_path and write to annotated_path."""
    if not cfg.metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {cfg.metadata_path}. Run the download pipeline first."
        )

    logger.info(
        f"Starting annotation pipeline | backend={cfg.vlm_backend} "
        f"| concurrency={cfg.concurrency} | verify={cfg.verify}"
    )

    df = pd.read_parquet(cfg.metadata_path)
    logger.info(f"Dataset: {len(df)} rows")

    records = []
    error_counts: dict[str, int] = defaultdict(int)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(
        max_workers=cfg.concurrency,
        initializer=_worker_init,
        initargs=(cfg,),
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

    result_df = pd.DataFrame(records)
    cfg.annotated_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(cfg.annotated_path, index=False)

    # Summary
    logger.info(f"Done: {len(records)}/{len(df)} succeeded → {cfg.annotated_path}")
    for caption_type in CAPTION_TYPES:
        if caption_type in result_df.columns:
            kept = result_df[caption_type].notna().sum()
            logger.info(f"  {caption_type}: {kept}/{len(result_df)} kept after verification")
    if error_counts:
        logger.warning(
            "Row errors: " + ", ".join(f"{k}: {v}" for k, v in error_counts.items())
        )
