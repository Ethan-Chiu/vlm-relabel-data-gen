from __future__ import annotations

from loguru import logger

from datagen import prompts
from datagen.annotators.base import verify_caption
from datagen.vlm.base import VLMBackend


class TwoCallAnnotator:
    """Generate one spatial caption, then optionally verify it sequentially."""

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
