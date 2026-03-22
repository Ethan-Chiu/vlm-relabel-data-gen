"""Robotic annotation pipelines.

Two pipelines are available:

  RoboticAnnotator  — 3 parallel VLM calls (Type A / B / C) + optional verification
  TwoCallAnnotator  — 1 generate call + optional verification

Both share the same verify_caption() primitive and the same outer
ProcessPoolExecutor infrastructure.

Execution model
───────────────
Outer parallelism  ProcessPoolExecutor — one process per CPU core, handles rows
Inner parallelism  ThreadPoolExecutor  — only used by RoboticAnnotator to run
                                         A/B/C and their verifications in parallel;
                                         TwoCallAnnotator needs no inner pool

                       ┌─ call A ─┐
Robotic:  img+cap ─────┼─ call B ─┼──► [verify A, B, C in parallel] ──► dict
                       └─ call C ─┘

Two-call: img+cap ─── call 1 (generate) ──► verify ──► spatial_caption
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from datagen import prompts
from datagen.config import Config
from datagen.vlm.base import VLMBackend


# ── Shared verification primitive ─────────────────────────────────────────────

def verify_caption(backend: VLMBackend, image_bytes: bytes, caption: str) -> bool:
    """Return True if the caption contains spatial errors (should be discarded).

    Reused by both RoboticAnnotator and TwoCallAnnotator.
    """
    answer = backend.call(image_bytes, prompts.VERIFY.format(caption=caption))
    return answer.strip().upper().startswith("YES")


# ── RoboticAnnotator (3-call parallel pipeline) ───────────────────────────────

class RoboticAnnotator:
    """Runs Type A / B / C calls in parallel, then verifies each in parallel.

    Instantiated once per worker process; the inner ThreadPoolExecutor is
    persistent and reused across all rows.
    """

    CAPTION_TYPES = ("type_a", "type_b", "type_c")

    def __init__(self, backend: VLMBackend, verify: bool = True) -> None:
        self._backend = backend
        self._verify = verify
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
        to_verify = {k: v for k, v in results.items() if v is not None}
        futures = {
            key: self._pool.submit(verify_caption, self._backend, image_bytes, caption)
            for key, caption in to_verify.items()
        }
        for key, future in futures.items():
            try:
                if future.result():
                    logger.debug(f"{key} discarded by verification")
                    results[key] = None
            except Exception as e:
                logger.warning(f"Verification for {key} failed: {type(e).__name__}: {e}")
        return results

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)


# ── TwoCallAnnotator (generate → verify) ──────────────────────────────────────

class TwoCallAnnotator:
    """Generate one spatial caption, then optionally verify it.

    No inner thread pool needed — the two calls are sequential by design.
    """

    CAPTION_TYPES = ("spatial_caption",)

    def __init__(self, backend: VLMBackend, verify: bool = True) -> None:
        self._backend = backend
        self._verify = verify

    def annotate(self, image_bytes: bytes, original_caption: str) -> dict:
        caption = self._backend.call(
            image_bytes,
            prompts.TWO_CALL_GENERATE.format(original_caption=original_caption),
        )
        if self._verify and verify_caption(self._backend, image_bytes, caption):
            logger.debug("spatial_caption discarded by verification")
            caption = None
        return {"spatial_caption": caption}


# ── Worker process state ───────────────────────────────────────────────────────
# Module-level globals, initialized once per worker process via the initializer.

_worker_annotator: RoboticAnnotator | TwoCallAnnotator | None = None
_worker_img_dir: Path | None = None


def _worker_init(cfg: Config, pipeline: str) -> None:
    global _worker_annotator, _worker_img_dir
    from datagen.vlm import get_backend
    backend = get_backend(cfg)
    _worker_img_dir = cfg.output_dir
    if pipeline == "two-call":
        _worker_annotator = TwoCallAnnotator(backend=backend, verify=cfg.verify)
    else:
        _worker_annotator = RoboticAnnotator(backend=backend, verify=cfg.verify)


def _annotate_row(row: dict) -> dict:
    img_bytes = (_worker_img_dir / row["filename"]).read_bytes()
    return {**row, **_worker_annotator.annotate(img_bytes, row["caption"])}


# ── Shared pipeline runner ─────────────────────────────────────────────────────

def run(cfg: Config, pipeline: str = "robotic") -> None:
    """Run the chosen annotation pipeline over metadata_path.

    pipeline: "robotic"  — Type A / B / C (6 VLM calls with verify)
              "two-call" — spatial_caption (2 VLM calls with verify)
    """
    if not cfg.metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {cfg.metadata_path}. Run the download pipeline first."
        )

    logger.info(
        f"Starting annotation pipeline | pipeline={pipeline} "
        f"| backend={cfg.vlm_backend} | concurrency={cfg.concurrency} | verify={cfg.verify}"
    )

    df = pd.read_parquet(cfg.metadata_path)
    logger.info(f"Dataset: {len(df)} rows")

    records = []
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

    result_df = pd.DataFrame(records)
    cfg.annotated_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(cfg.annotated_path, index=False)

    caption_types = (
        TwoCallAnnotator.CAPTION_TYPES if pipeline == "two-call"
        else RoboticAnnotator.CAPTION_TYPES
    )
    logger.info(f"Done: {len(records)}/{len(df)} succeeded → {cfg.annotated_path}")
    for col in caption_types:
        if col in result_df.columns:
            kept = result_df[col].notna().sum()
            logger.info(f"  {col}: {kept}/{len(result_df)} kept after verification")
    if error_counts:
        logger.warning(
            "Row errors: " + ", ".join(f"{k}: {v}" for k, v in error_counts.items())
        )
