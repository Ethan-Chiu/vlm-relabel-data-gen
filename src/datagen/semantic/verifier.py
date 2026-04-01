"""SemanticVerifier: per-object crop verification of appearance/state (Stage 1.5).

For each object in a SemanticAnnotation, crops the bounding box from the full
image (with padding), sends the crop to the VLM with a focused prompt, and
corrects appearance/state if the VLM disagrees with the Stage 1 extraction.

Only objects whose crop area exceeds MIN_CROP_AREA_FRACTION of the full image
are verified; tiny detections are skipped because crops are uninformative.

When multiple detections share the same label (e.g. "person × 5"), the
detection with the smallest depth_rank (nearest instance) is used for the crop.
"""

from __future__ import annotations

import io
import json
import re

from loguru import logger
from PIL import Image

from datagen import prompts
from datagen.semantic.models import ObjectProperties, SemanticAnnotation
from datagen.vlm.base import VLMBackend

# Skip crops whose area is less than this fraction of the full image area.
# 0.002 ≈ 0.2% — roughly a 50×50 px crop in a 1280×720 image.
MIN_CROP_AREA_FRACTION = 0.002

# Padding added to each side of the bbox as a fraction of the image dimension.
# 0.05 = 5% — adds a little context without losing crop resolution.
CROP_PADDING_FRACTION = 0.05

_VALID_CONFIDENCE = frozenset({"high", "medium"})


class SemanticVerifier:
    """Verify and correct per-object appearance/state using bounding-box crops.

    Instantiated once per worker process alongside SemanticExtractor.
    The same VLMBackend instance is shared (backends are stateless HTTP clients).
    """

    def __init__(self, backend: VLMBackend) -> None:
        self._backend = backend

    def verify_objects(
        self,
        image_bytes: bytes,
        detections_json: str,
        annotation: SemanticAnnotation,
    ) -> SemanticAnnotation:
        """Verify each object's appearance and state using a bounding-box crop.

        Parameters
        ----------
        image_bytes:
            Raw bytes of the full scene image (JPEG or PNG).
        detections_json:
            The scene_detections JSON string from scene_graphs.parquet.
            Each entry: {label, bbox [x1,y1,x2,y2] pixel coords, depth_rank, …}
        annotation:
            SemanticAnnotation produced by Stage 1. Mutated in place and returned.
        """
        try:
            detections = json.loads(detections_json)
        except json.JSONDecodeError:
            logger.warning("Could not parse scene_detections JSON; skipping verification")
            return annotation

        det_map = _detection_map(detections)

        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            logger.warning(f"Could not open image for crop verification: {e}")
            return annotation

        img_w, img_h = img.size
        full_area = img_w * img_h

        corrections = 0
        for obj in annotation.objects:
            label_key = obj.label.strip().lower()
            # Match by (label, position) first; fall back to nearest for that label
            det = det_map.get((label_key, obj.position)) or det_map.get((label_key, None))
            if det is None:
                logger.debug(f"No detection for '{obj.label}'; skipping crop verification")
                continue

            crop_bytes = _make_crop(img, det["bbox"], img_w, img_h, full_area)
            if crop_bytes is None:
                logger.debug(f"Crop for '{obj.label}' too small or malformed; skipping")
                continue

            if _verify_one_object(self._backend, crop_bytes, obj):
                corrections += 1

        if corrections:
            logger.debug(
                f"Crop verification: {corrections}/{len(annotation.objects)} objects corrected"
            )

        return annotation


# ── Helpers ────────────────────────────────────────────────────────────────────

def _detection_map(detections: list[dict]) -> dict[tuple, dict]:
    """Build a lookup map from (label_lower, position) and (label_lower, None).

    (label, position) gives an exact match for individual instances.
    (label, None) gives the nearest instance as a fallback for unpositioned lookups.
    """
    exact: dict[tuple, dict] = {}
    nearest: dict[str, dict] = {}  # label → nearest (min depth_rank)

    for det in detections:
        label = str(det.get("label", "")).strip().lower()
        if not label:
            continue
        pos = str(det.get("position", "")).strip() or None
        exact[(label, pos)] = det
        rank = det.get("depth_rank") or 9999
        if label not in nearest or rank < (nearest[label].get("depth_rank") or 9999):
            nearest[label] = det

    result: dict[tuple, dict] = dict(exact)
    for label, det in nearest.items():
        result[(label, None)] = det
    return result


def _make_crop(
    img: Image.Image,
    bbox: list[float],
    img_w: int,
    img_h: int,
    full_area: int,
) -> bytes | None:
    """Crop img to the padded bbox and return JPEG bytes, or None if too small.

    bbox is [x1, y1, x2, y2] in pixel coordinates (as stored in scene_detections).
    """
    try:
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    except (TypeError, IndexError, ValueError):
        return None

    pad_x = img_w * CROP_PADDING_FRACTION
    pad_y = img_h * CROP_PADDING_FRACTION

    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(img_w, int(x2 + pad_x))
    cy2 = min(img_h, int(y2 + pad_y))

    crop_w = cx2 - cx1
    crop_h = cy2 - cy1
    if crop_w <= 0 or crop_h <= 0:
        return None

    if (crop_w * crop_h) / full_area < MIN_CROP_AREA_FRACTION:
        return None

    crop_img = img.crop((cx1, cy1, cx2, cy2))
    buf = io.BytesIO()
    crop_img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _verify_one_object(
    backend: VLMBackend,
    crop_bytes: bytes,
    obj: ObjectProperties,
) -> bool:
    """Call the VLM on a crop; mutate obj in place. Returns True if any field corrected."""
    prompt = prompts.SEMANTIC_VERIFY_OBJECT.format(
        label=obj.label,
        appearance=obj.appearance if obj.appearance is not None else "unknown",
        state=obj.state if obj.state is not None else "unknown",
    )
    try:
        raw = backend.call(crop_bytes, prompt)
    except Exception as e:
        logger.debug(f"Crop VLM call failed for '{obj.label}': {e}")
        return False

    try:
        data = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as e:
        logger.debug(f"Could not parse crop verification response for '{obj.label}': {e}")
        return False

    corrected = False

    new_app = data.get("appearance")
    new_app_conf = data.get("appearance_confidence", "medium")
    if data.get("appearance_corrected") is True and new_app is not None:
        obj.appearance = new_app
        obj.appearance_confidence = new_app_conf if new_app_conf in _VALID_CONFIDENCE else "medium"
        corrected = True
    elif new_app_conf in _VALID_CONFIDENCE:
        obj.appearance_confidence = new_app_conf

    new_state = data.get("state")
    new_state_conf = data.get("state_confidence", "medium")
    if data.get("state_corrected") is True:
        obj.state = new_state  # may be None (cleared)
        obj.state_confidence = new_state_conf if new_state_conf in _VALID_CONFIDENCE else "medium"
        corrected = True
    elif new_state_conf in _VALID_CONFIDENCE:
        obj.state_confidence = new_state_conf

    return corrected
