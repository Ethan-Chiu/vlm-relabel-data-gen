from __future__ import annotations

import json

from loguru import logger

from datagen import prompts
from datagen.annotators.base import _ParallelVLMAnnotator
from datagen.vlm.base import VLMBackend


class SemanticAnnotator(_ParallelVLMAnnotator):
    """Generate a single grounded caption from pre-extracted semantic annotations.

    Requires both scene_graphs.parquet and semantic_annotations.parquet.
    Verification is off by default (opt in with --verify).

    Output column:
      semantic_caption — plain string, or None
    """

    CAPTION_TYPES = ("semantic_caption",)

    def __init__(self, backend: VLMBackend, verify: bool = False) -> None:
        super().__init__(backend, verify, max_workers=1)

    def annotate(
        self,
        image_bytes: bytes,
        scene_graph: str,
        semantic_props: str,
        semantic_rels: str,
    ) -> dict:
        try:
            props = json.loads(semantic_props)
            rels = json.loads(semantic_rels)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse semantic annotation JSON: {e}")
            return {"semantic_caption": None}

        semantic_annotation = json.dumps({**props, **rels}, indent=2)
        prompt = prompts.SEMANTIC_CAPTION.format(
            scene_graph=scene_graph,
            semantic_annotation=semantic_annotation,
        )

        try:
            caption = self._backend.call(image_bytes, prompt).strip()
        except Exception as e:
            logger.warning(f"Caption generation failed: {type(e).__name__}: {e}")
            return {"semantic_caption": None}

        if not caption:
            logger.debug("Empty caption returned by generation call")
            return {"semantic_caption": None}

        if self._verify:
            try:
                should_discard = self._verify_one(image_bytes, caption)
                if should_discard:
                    logger.debug("semantic_caption discarded by verification")
                    return {"semantic_caption": None}
            except Exception as e:
                logger.warning(f"Verification error: {e}")
                # keep on error to avoid data loss

        return {"semantic_caption": caption}

    def _verify_one(self, image_bytes: bytes, caption: str) -> bool:
        """Return True if caption should be discarded."""
        answer = self._backend.call(
            image_bytes, prompts.SEMANTIC_VERIFY.format(caption=caption)
        )
        return answer.strip().upper().startswith("YES")
