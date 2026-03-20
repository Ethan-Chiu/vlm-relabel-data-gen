"""Ray Data pipeline for large-scale VLM relabeling."""

from __future__ import annotations

from pathlib import Path

import ray
import ray.data
from loguru import logger

from datagen.config import Config


class _VLMRelabeler:
    """Stateful Ray worker that loads the VLM backend once per worker process."""

    def __init__(self, cfg: Config) -> None:
        from datagen.vlm import get_backend
        self.backend = get_backend(cfg)
        self.img_dir = cfg.output_dir

    def __call__(self, row: dict) -> dict:
        img_bytes = (self.img_dir / row["filename"]).read_bytes()
        row["new_caption"] = self.backend.relabel(img_bytes, row["caption"])
        return row


def run(cfg: Config) -> None:
    if not cfg.metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {cfg.metadata_path}. Run the download pipeline first."
        )

    logger.info(f"Starting relabel pipeline | backend={cfg.vlm_backend} | concurrency={cfg.concurrency}")
    ray.init(ignore_reinit_error=True)

    ds = ray.data.read_parquet(str(cfg.metadata_path))
    logger.info(f"Dataset: {ds.count()} rows")

    ds = ds.map(
        _VLMRelabeler,
        fn_constructor_args=[cfg],
        compute=ray.data.ActorPoolStrategy(size=cfg.concurrency),
    )

    cfg.relabeled_path.parent.mkdir(parents=True, exist_ok=True)
    ds.write_parquet(str(cfg.relabeled_path.with_suffix("")))  # Ray writes a sharded directory
    logger.info(f"Relabeled output → {cfg.relabeled_path.with_suffix('')}/")
