"""VLM relabeling pipeline.

Executor is chosen automatically based on backend type:
  - API backends (openai, gemini, vllm server): ProcessPoolExecutor
    → true CPU parallelism per worker; each worker owns its own backend client,
      handles image loading, and can run additional CPU stages in the future.
  - Local model backends (qwen_local): Ray Data with ActorPoolStrategy
    → one Ray actor per GPU, model loaded once per worker.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from datagen.config import Config

# Backends that need a GPU per worker — use Ray.
# Everything else uses multiprocessing.
_RAY_BACKENDS = {"qwen_local"}

# --- Multiprocessing worker state (module-level so it's picklable) -----------
# Each worker process initializes this once via _worker_init().

_worker_backend = None
_worker_img_dir: Path | None = None
_worker_cfg: Config | None = None


def _worker_init(cfg: Config) -> None:
    """Called once per worker process. Initializes the backend and any shared state."""
    global _worker_backend, _worker_img_dir, _worker_cfg
    from datagen.vlm import get_backend
    _worker_cfg = cfg
    _worker_img_dir = cfg.output_dir
    _worker_backend = get_backend(cfg)


def _process_row(row: dict) -> dict:
    """Runs in a worker process. Add extra per-row stages here as the pipeline grows."""
    # Stage 1: load image
    img_bytes = (_worker_img_dir / row["filename"]).read_bytes()

    # Stage 2: VLM relabeling
    row["new_caption"] = _worker_backend.call(img_bytes, _worker_cfg.vlm_prompt)

    # Stage N: add more CPU stages here (resize, OCR, embedding, filtering…)

    return row


# -----------------------------------------------------------------------------

def _run_multiprocess(cfg: Config) -> None:
    """ProcessPoolExecutor for API-based backends."""
    df = pd.read_parquet(cfg.metadata_path)
    logger.info(f"Dataset: {len(df)} rows | workers: {cfg.concurrency}")

    records = []
    error_counts: dict[str, int] = defaultdict(int)

    with ProcessPoolExecutor(
        max_workers=cfg.concurrency,
        initializer=_worker_init,
        initargs=(cfg,),
    ) as executor:
        futures = {
            executor.submit(_process_row, row.to_dict()): i
            for i, row in df.iterrows()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Relabeling"):
            try:
                records.append(future.result())
            except Exception as e:
                error_counts[type(e).__name__] += 1

    result_df = pd.DataFrame(records)
    cfg.relabeled_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(cfg.relabeled_path, index=False)

    logger.info(f"Done: {len(records)}/{len(df)} succeeded → {cfg.relabeled_path}")
    if error_counts:
        logger.warning(
            "Error breakdown: " + ", ".join(f"{k}: {v}" for k, v in error_counts.items())
        )


def _run_ray(cfg: Config) -> None:
    """Ray actor pool for local GPU model backends (qwen_local)."""
    import ray
    import ray.data

    class _VLMRelabeler:
        def __init__(self, cfg: Config) -> None:
            from datagen.vlm import get_backend
            self.backend = get_backend(cfg)
            self.img_dir = cfg.output_dir

        def __call__(self, row: dict) -> dict:
            img_bytes = (self.img_dir / row["filename"]).read_bytes()
            row["new_caption"] = self.backend.call(img_bytes, cfg.vlm_prompt)
            return row

    _src = str(Path(__file__).parent.parent)
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"PYTHONPATH": _src}},
    )
    ctx = ray.data.DataContext.get_current()
    ctx.max_errored_blocks = 0

    ds = ray.data.read_parquet(str(cfg.metadata_path))
    logger.info(f"Dataset schema: {ds.schema()}")

    ds = ds.map(
        _VLMRelabeler,
        fn_constructor_args=[cfg],
        compute=ray.data.ActorPoolStrategy(size=cfg.concurrency),
        num_gpus=cfg.num_gpus_per_worker,
    )

    out_dir = str(cfg.relabeled_path.with_suffix(""))
    cfg.relabeled_path.parent.mkdir(parents=True, exist_ok=True)
    ds.write_parquet(out_dir)
    logger.info(f"Relabeled output → {out_dir}/")


def run(cfg: Config) -> None:
    if not cfg.metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {cfg.metadata_path}. Run the download pipeline first."
        )

    executor = "ray" if cfg.vlm_backend in _RAY_BACKENDS else "multiprocess"
    logger.info(
        f"Starting relabel pipeline | backend={cfg.vlm_backend} "
        f"| concurrency={cfg.concurrency} | executor={executor}"
    )

    if cfg.vlm_backend in _RAY_BACKENDS:
        _run_ray(cfg)
    else:
        _run_multiprocess(cfg)
