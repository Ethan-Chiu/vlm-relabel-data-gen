"""Shared verification primitive and base class for parallel annotators."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from datagen import prompts
from datagen.vlm.base import VLMBackend


def verify_caption(backend: VLMBackend, image_bytes: bytes, caption: str) -> bool:
    """Return True if the caption contains errors (should be discarded)."""
    answer = backend.call(image_bytes, prompts.VERIFY.format(caption=caption))
    return answer.strip().upper().startswith("YES")


class _ParallelVLMAnnotator:
    """Shared machinery for annotators that run multiple VLM calls in parallel.

    Provides a persistent ThreadPoolExecutor, parallel dispatch, and a
    dict-based verification helper. Subclasses only need to implement annotate().
    """

    def __init__(self, backend: VLMBackend, verify: bool, max_workers: int = 3) -> None:
        self._backend = backend
        self._verify = verify
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vlm")

    def _run_parallel(self, prompt_map: dict[str, str], image_bytes: bytes) -> dict:
        """Submit all prompts in parallel; return {key: result | None}."""
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

    def _verify_dict(self, image_bytes: bytes, results: dict) -> dict:
        """Verify each non-None caption in parallel; set to None if discarded."""
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
