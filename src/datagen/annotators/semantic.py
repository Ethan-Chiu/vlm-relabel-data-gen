from __future__ import annotations

import json
import re

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

        # Reorder objects to match scene_graph label order (nearest-first after
        # build_scene_graph sorts by depth_rank; also works on old parquets).
        ordered_labels = _label_order_from_scene_graph(scene_graph)
        if ordered_labels:
            rank = {lbl: i for i, lbl in enumerate(ordered_labels)}
            props["objects"] = sorted(
                props.get("objects", []),
                key=lambda o: rank.get(o.get("label", "").strip().lower(), 9999),
            )

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


def _label_order_from_scene_graph(scene_graph: str) -> list[str]:
    """Return object labels in the order they appear in the scene_graph text.

    After build_scene_graph() sorts by depth_rank, this gives nearest-first order.
    Works correctly on both old (detection-ordered) and new (depth-ordered) scene_graphs.
    """
    labels: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"^\d+\.\s+([^\[]+)", scene_graph, re.MULTILINE):
        label = m.group(1).strip().lower()
        if label and label not in seen:
            labels.append(label)
            seen.add(label)
    return labels
