from __future__ import annotations

from datagen import prompts
from datagen.annotators.base import _ParallelVLMAnnotator
from datagen.vlm.base import VLMBackend


class RoboticAnnotator(_ParallelVLMAnnotator):
    """Type A / B / C calls in parallel, then verify each in parallel."""

    CAPTION_TYPES = ("type_a", "type_b", "type_c")

    def annotate(self, image_bytes: bytes, original_caption: str) -> dict:
        results = self._run_parallel({
            "type_a": prompts.TYPE_A.format(original_caption=original_caption),
            "type_b": prompts.TYPE_B.format(original_caption=original_caption),
            "type_c": prompts.TYPE_C.format(original_caption=original_caption),
        }, image_bytes)
        if self._verify:
            results = self._verify_dict(image_bytes, results)
        return results
