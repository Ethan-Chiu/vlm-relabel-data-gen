"""SemanticExtractor: single VLM call → parsed SemanticAnnotation.

Consumes a pre-computed scene_graph string and scene_detections JSON (both
produced by extract_scene_graphs.py) together with the raw image bytes.

The response is expected to be a JSON object matching the SEMANTIC_EXTRACT
schema.  We handle the two most common VLM deviations:
  - JSON wrapped in a markdown code fence (```json ... ```)
  - Trailing/leading whitespace or explanation text around the JSON block
"""

from __future__ import annotations

import json
import re

from loguru import logger

from datagen import prompts
from datagen.semantic.models import (
    ObjectProperties,
    Relationship,
    SceneContext,
    SemanticAnnotation,
)
from datagen.vlm.base import VLMBackend

# Valid values — used to filter out hallucinated entries
_VALID_AFFORDANCES = frozenset({
    "graspable", "fillable", "openable", "stackable",
    "pourable", "cuttable", "liftable",
})
_VALID_CONFIDENCE = frozenset({"high", "medium"})
_VALID_TEMPORAL_PHASE = frozenset({"setup", "in_progress", "completed"})


class SemanticExtractor:
    """Wraps the SEMANTIC_EXTRACT VLM call with parsing and light validation.

    Instantiated once per worker process; the VLMBackend is held for the
    lifetime of the process.
    """

    def __init__(self, backend: VLMBackend) -> None:
        self._backend = backend

    def extract(
        self,
        image_bytes: bytes,
        scene_graph: str,
        scene_detections: str,
    ) -> SemanticAnnotation:
        """Call the VLM and return a parsed SemanticAnnotation.

        Raises ValueError if the response cannot be parsed as valid JSON or
        does not contain the required top-level keys.
        """
        prompt = prompts.SEMANTIC_EXTRACT.format(
            scene_graph=scene_graph,
            scene_detections=scene_detections,
        )
        raw = self._backend.call(image_bytes, prompt)
        return _parse_response(raw, scene_graph)


# ── Parsing helpers ────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    # Match ```json ... ``` or ``` ... ```
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _extract_known_labels(scene_graph: str) -> set[str]:
    """Extract object labels from scene_graph text so we can validate references."""
    labels: set[str] = set()
    # Each detection line looks like: "1. cup [left-near] conf=..."
    for m in re.finditer(r"^\d+\.\s+([^\[]+)", scene_graph, re.MULTILINE):
        labels.add(m.group(1).strip().lower())
    return labels


def _parse_response(raw: str, scene_graph: str) -> SemanticAnnotation:
    """Parse raw VLM text into a SemanticAnnotation, applying light validation."""
    clean = _strip_fences(raw)

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as exc:
        raise ValueError(f"SEMANTIC_EXTRACT response is not valid JSON: {exc}\nRaw: {raw[:300]}") from exc

    for key in ("scene_context", "objects", "relationships"):
        if key not in data:
            raise ValueError(f"SEMANTIC_EXTRACT response missing key '{key}'")

    known_labels = _extract_known_labels(scene_graph)

    # ── scene_context ──────────────────────────────────────────────────────────
    sc = data["scene_context"]
    temporal_phase = sc.get("temporal_phase")
    if temporal_phase not in _VALID_TEMPORAL_PHASE:
        temporal_phase = None
    confidence = sc.get("confidence", "medium")
    if confidence not in _VALID_CONFIDENCE:
        confidence = "medium"

    scene_context = SceneContext(
        environment=str(sc.get("environment", "unknown")),
        apparent_activity=sc.get("apparent_activity") or None,
        temporal_phase=temporal_phase,
        confidence=confidence,
    )

    # ── objects ────────────────────────────────────────────────────────────────
    objects: list[ObjectProperties] = []
    for obj in data.get("objects", []):
        label = str(obj.get("label", "")).strip()
        if not label:
            continue

        app_conf = obj.get("appearance_confidence", "medium")
        state_conf = obj.get("state_confidence", "medium")

        raw_affordances = obj.get("affordances") or []
        affordances = [a for a in raw_affordances if a in _VALID_AFFORDANCES]

        objects.append(ObjectProperties(
            label=label,
            appearance=obj.get("appearance") or None,
            appearance_confidence=app_conf if app_conf in _VALID_CONFIDENCE else "medium",
            state=obj.get("state") or None,
            state_confidence=state_conf if state_conf in _VALID_CONFIDENCE else "medium",
            affordances=affordances,
        ))

    # ── relationships ──────────────────────────────────────────────────────────
    relationships: list[Relationship] = []
    for rel in data.get("relationships", []):
        subject = str(rel.get("subject", "")).strip().lower()
        obj_label = str(rel.get("object", "")).strip()
        predicate = str(rel.get("predicate", "")).strip()
        rel_type = str(rel.get("type", "")).strip()
        confidence = rel.get("confidence", "medium")
        evidence = str(rel.get("evidence", "")).strip()

        if not subject or not obj_label or not predicate:
            continue

        # Subject must reference a detected object
        if known_labels and subject not in known_labels:
            logger.debug(f"Dropping relationship with unknown subject '{subject}'")
            continue

        if confidence not in _VALID_CONFIDENCE:
            confidence = "medium"

        relationships.append(Relationship(
            subject=subject,
            predicate=predicate,
            object=obj_label,
            type=rel_type,
            confidence=confidence,
            evidence=evidence,
        ))

    return SemanticAnnotation(
        scene_context=scene_context,
        objects=objects,
        relationships=relationships,
    )
