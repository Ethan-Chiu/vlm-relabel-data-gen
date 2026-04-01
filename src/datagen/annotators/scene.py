from __future__ import annotations

from datagen import prompts
from datagen.annotators.base import _ParallelVLMAnnotator
from datagen.vlm.base import VLMBackend


class CachedSceneAnnotator(_ParallelVLMAnnotator):
    """3 scene-conditioned VLM calls using a pre-computed scene_graph string.

    No GPU models — pure API calls, so normal concurrency applies.
    Requires scene_graphs.parquet (produced by extract_scene_graphs.py).
    """

    CAPTION_TYPES = ("scene_type_a", "scene_type_b", "scene_type_c")

    def annotate(
        self, image_bytes: bytes, original_caption: str, scene_graph: str
    ) -> dict:
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
            results = self._verify_dict(image_bytes, results)
        return results
