"""All annotation pipelines.

Pipelines
─────────
  SimpleAnnotator     — 1 VLM call with a configurable prompt → new_caption  [relabel]
  TwoCallAnnotator    — 1 generate call + optional verification → spatial_caption
  RoboticAnnotator    — 3 parallel VLM calls (Type A / B / C) + optional verification
  CachedSceneAnnotator— reads pre-computed scene_graph → 3 VLM calls + optional verification

All share the same outer ProcessPoolExecutor infrastructure.

Execution model
───────────────
Outer parallelism  ProcessPoolExecutor — one process per CPU core, handles rows
Inner parallelism  ThreadPoolExecutor  — used by RoboticAnnotator / CachedSceneAnnotator
                                         to run A/B/C calls in parallel

Relabel:  img+cap ─── VLM(vlm_prompt) ──► new_caption

Two-call: img+cap ─── call 1 (generate) ──► verify ──► spatial_caption

                       ┌─ call A ─┐
Robotic:  img+cap ─────┼─ call B ─┼──► [verify in parallel] ──► dict
                       └─ call C ─┘

                                       ┌─ scene call A ─┐
Scene:    img+cap ─► scene_graph ──────┼─ scene call B ─┼──► [verify in parallel] ──► dict
          (pre-computed parquet)        └─ scene call C ─┘
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


# ── SimpleAnnotator (single configurable VLM call) ────────────────────────────

class SimpleAnnotator:
    """One VLM call per image using vlm_prompt from config. Outputs new_caption."""

    CAPTION_TYPES = ("new_caption",)

    def __init__(self, backend: VLMBackend, prompt: str) -> None:
        self._backend = backend
        self._prompt = prompt

    def annotate(self, image_bytes: bytes, original_caption: str) -> dict:
        caption = self._backend.call(image_bytes, self._prompt)
        return {"new_caption": caption}


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


# ── CachedSceneAnnotator (uses pre-computed scene_graph text) ─────────────────

class CachedSceneAnnotator:
    """Run 3 scene-conditioned VLM calls using a pre-computed scene_graph string.

    The scene_graph is produced by scripts/extract_scene_graphs.py and read from
    scene_graphs.parquet. No GPU models are loaded here — this annotator is just
    API calls, so you can use the normal concurrency setting.
    """

    CAPTION_TYPES = ("scene_type_a", "scene_type_b", "scene_type_c")

    def __init__(self, backend: VLMBackend, verify: bool = True) -> None:
        self._backend = backend
        self._verify = verify
        self._pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="scene_vlm")

    def annotate(self, image_bytes: bytes, original_caption: str, scene_graph: str) -> dict:
        """Return dict with keys scene_type_a, scene_type_b, scene_type_c."""
        results = self._run_parallel({
            "scene_type_a": prompts.SCENE_TYPE_A.format(
                original_caption=original_caption, scene_graph=scene_graph
            ),
            "scene_type_b": prompts.SCENE_TYPE_B.format(
                original_caption=original_caption, scene_graph=scene_graph
            ),
            "scene_type_c": prompts.SCENE_TYPE_C.format(
                original_caption=original_caption, scene_graph=scene_graph
            ),
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


# ── Worker process state ───────────────────────────────────────────────────────
# Module-level globals, initialized once per worker process via the initializer.

_worker_annotator: SimpleAnnotator | RoboticAnnotator | TwoCallAnnotator | CachedSceneAnnotator | None = None
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
    else:
        _worker_annotator = RoboticAnnotator(backend=backend, verify=cfg.verify)


def _annotate_row(row: dict) -> dict:
    try:
        img_bytes = (_worker_img_dir / row["filename"]).read_bytes()
        if _worker_pipeline == "scene-graph":
            annotations = _worker_annotator.annotate(img_bytes, row["caption"], row["scene_graph"])
        else:
            annotations = _worker_annotator.annotate(img_bytes, row["caption"])
        return {**row, **annotations}
    except Exception as e:
        # Re-raise as RuntimeError to guarantee picklability across process boundaries.
        # Some SDK exceptions (e.g. openai.APIStatusError) cannot be unpickled in the
        # parent process, which crashes the entire pool rather than recording a row error.
        raise RuntimeError(f"{type(e).__name__}: {e}") from None


# ── Shared pipeline runner ─────────────────────────────────────────────────────

def run(cfg: Config, pipeline: str = "robotic") -> None:
    """Run the chosen annotation pipeline over metadata_path.

    pipeline: "relabel"     — single VLM call with vlm_prompt → new_caption
              "two-call"    — spatial_caption (2 VLM calls with verify)
              "robotic"     — Type A / B / C (6 VLM calls with verify)
              "scene-graph" — pre-computed scene_graph → scene Type A / B / C
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

    if pipeline == "scene-graph":
        if not cfg.scene_graph_path.exists():
            raise FileNotFoundError(
                f"Scene graphs not found at {cfg.scene_graph_path}. "
                "Run the extraction stage first:\n"
                "  uv run python scripts/extract_scene_graphs.py "
                "--config configs/scene_graph.toml"
            )
        sg_df = pd.read_parquet(cfg.scene_graph_path, columns=["filename", "scene_graph"])
        before = len(df)
        df = df.merge(sg_df, on="filename", how="inner")
        if len(df) < before:
            logger.warning(
                f"{before - len(df)} rows dropped — no scene graph found for them. "
                "Re-run extract_scene_graphs.py to fill gaps."
            )
        logger.info(f"Joined with scene graphs: {len(df)} rows ready")

    if cfg.annotate_limit is not None:
        df = df.head(cfg.annotate_limit)
        logger.info(f"Limiting to first {cfg.annotate_limit} rows")

    # Load existing output to support skip/overwrite logic
    existing_df: pd.DataFrame | None = None
    if cfg.annotated_path.exists():
        existing_df = pd.read_parquet(cfg.annotated_path)
        existing_filenames = set(existing_df["filename"])
        if not cfg.overwrite:
            before = len(df)
            df = df[~df["filename"].isin(existing_filenames)]
            skipped = before - len(df)
            if skipped:
                logger.info(f"Skipping {skipped} already-annotated rows (overwrite=False)")

    if df.empty:
        logger.info("Nothing to process.")
        return

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

    new_df = pd.DataFrame(records)

    # Merge with existing output
    if existing_df is not None:
        if cfg.overwrite:
            # Drop stale records for rows we just re-processed
            reprocessed = {r["filename"] for r in records}
            existing_df = existing_df[~existing_df["filename"].isin(reprocessed)]
        result_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        result_df = new_df

    cfg.annotated_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(cfg.annotated_path, index=False)

    if pipeline == "relabel":
        caption_types = SimpleAnnotator.CAPTION_TYPES
    elif pipeline == "two-call":
        caption_types = TwoCallAnnotator.CAPTION_TYPES
    elif pipeline == "scene-graph":
        caption_types = CachedSceneAnnotator.CAPTION_TYPES
    else:
        caption_types = RoboticAnnotator.CAPTION_TYPES
    logger.info(f"Done: {len(records)}/{len(df)} succeeded → {cfg.annotated_path}")
    for col in caption_types:
        if col in new_df.columns:
            kept = new_df[col].notna().sum()
            logger.info(f"  {col}: {kept}/{len(new_df)} kept after verification")
    if error_counts:
        logger.warning(
            "Row errors: " + ", ".join(f"{k}: {v}" for k, v in error_counts.items())
        )
