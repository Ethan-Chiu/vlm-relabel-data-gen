"""Geometry utilities: assign spatial positions and build the scene_graph text.

Takes a list of Detection objects (already populated with depth_value) and:
  1. Assigns position_h (left / center / right) and position_v (near / mid / far)
     based on bbox center and depth_value.
  2. Assigns depth_rank (1 = nearest) and area_rank (1 = largest).
  3. Estimates a brief free_space string for each object.
  4. Serializes everything to a compact scene_graph text for VLM prompts.
"""

from __future__ import annotations

import numpy as np

from datagen.scene.models import Detection


# ── Position helpers ───────────────────────────────────────────────────────────

def _h_position(cx: float, W: float) -> str:
    frac = cx / W
    if frac < 0.35:
        return "left"
    if frac > 0.65:
        return "right"
    return "center"


def _v_position(depth: float) -> str:
    """depth in [0, 1] where 0 = closest."""
    if depth < 0.33:
        return "near"
    if depth < 0.67:
        return "mid"
    return "far"


# ── Free-space estimation ──────────────────────────────────────────────────────

def _estimate_free_space(det: Detection, all_dets: list[Detection], W: int, H: int) -> str:
    """Return a brief natural-language description of clear space around the object."""
    x1, y1, x2, y2 = det.bbox
    cx, cy = det.bbox_center

    directions: list[str] = []
    # Check each cardinal direction for clearance (25% of image dimension)
    margin_x = W * 0.25
    margin_y = H * 0.25

    if x1 > margin_x:
        directions.append("left")
    if x2 < W - margin_x:
        directions.append("right")
    if y1 > margin_y:
        directions.append("above")
    if y2 < H - margin_y:
        directions.append("below")

    # Remove directions blocked by another object's bbox
    blocked: set[str] = set()
    for other in all_dets:
        if other is det:
            continue
        ox1, oy1, ox2, oy2 = other.bbox
        ocx, ocy = other.bbox_center
        # Rough overlap check in each direction
        horizontal_aligned = abs(ocy - cy) < (y2 - y1)
        vertical_aligned = abs(ocx - cx) < (x2 - x1)
        if ocx < cx and horizontal_aligned and ox2 > x1 - margin_x:
            blocked.add("left")
        if ocx > cx and horizontal_aligned and ox1 < x2 + margin_x:
            blocked.add("right")
        if ocy < cy and vertical_aligned and oy2 > y1 - margin_y:
            blocked.add("above")
        if ocy > cy and vertical_aligned and oy1 < y2 + margin_y:
            blocked.add("below")

    clear = [d for d in directions if d not in blocked]
    if clear:
        return f"Clear space {', '.join(clear)} of the {det.label}."
    return f"Limited clear space immediately around the {det.label}."


# ── Main entry points ──────────────────────────────────────────────────────────

def assign_geometry(detections: list[Detection], image_shape: tuple[int, int]) -> None:
    """Mutate detections in-place to assign position, rank, and free_space fields.

    Parameters
    ----------
    detections:   list of Detection objects with depth_value already set
    image_shape:  (H, W) of the source image
    """
    H, W = image_shape

    # Depth ranks (1 = nearest = smallest depth_value)
    with_depth = [d for d in detections if d.depth_value is not None]
    sorted_by_depth = sorted(with_depth, key=lambda d: d.depth_value)
    for rank, det in enumerate(sorted_by_depth, start=1):
        det.depth_rank = rank

    # Area ranks (1 = largest mask)
    sorted_by_area = sorted(detections, key=lambda d: d.area, reverse=True)
    for rank, det in enumerate(sorted_by_area, start=1):
        det.area_rank = rank

    # Horizontal / vertical position and free space
    for det in detections:
        cx, cy = det.bbox_center
        det.position_h = _h_position(cx, W)
        det.position_v = _v_position(det.depth_value) if det.depth_value is not None else "unknown"
        det.free_space = _estimate_free_space(det, detections, W, H)


# ── IoU deduplication ─────────────────────────────────────────────────────────

# Detections whose bounding boxes overlap above this threshold with a larger
# detection are considered semantic near-duplicates (e.g. "hotel" vs "hotel
# resort" vs "resort") and are suppressed from the scene graph.
IOU_DEDUP_THRESHOLD = 0.5


def _iou(a, b) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _containment(small, large) -> float:
    """Fraction of `small` box's area that lies inside `large` box."""
    ix1, iy1 = max(small[0], large[0]), max(small[1], large[1])
    ix2, iy2 = min(small[2], large[2]), min(small[3], large[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_small = (small[2] - small[0]) * (small[3] - small[1])
    return inter / area_small if area_small > 0 else 0.0


# Suppress a detection if ≥80% of its area is inside a larger detection.
# This catches "resort" whose bbox is almost entirely within "hotel resort".
CONTAINMENT_THRESHOLD = 0.80


def _share_label_word(a: str, b: str) -> bool:
    """Return True if the two labels share at least one word (case-insensitive)."""
    return bool(set(a.lower().split()) & set(b.lower().split()))


def _nms_dicts(dets: list[dict]) -> list[dict]:
    """Suppress near-duplicate detections caused by overlapping GroundingDINO labels
    (e.g. "hotel" / "hotel resort" / "resort" all describing the same building).

    Two suppression criteria:
    - IoU > IOU_DEDUP_THRESHOLD: suppress regardless of label.
      High IoU means the boxes are nearly the same region (same object).
    - Containment > CONTAINMENT_THRESHOLD AND labels share a word: suppress.
      Catches "resort" (small box) inside "hotel resort" (large box).
      The shared-word guard prevents suppressing semantically different objects
      that happen to be spatially contained (e.g. "person" inside "hotel resort").

    Larger-area detections are processed first so the most encompassing label
    (e.g. "hotel resort") survives over contained sub-labels ("hotel", "resort").
    """
    def _area(d: dict) -> float:
        b = d.get("bbox") or []
        return (b[2] - b[0]) * (b[3] - b[1]) if len(b) == 4 else 0.0

    sorted_dets = sorted(dets, key=_area, reverse=True)
    kept: list[dict] = []
    for det in sorted_dets:
        bbox = det.get("bbox") or []
        if len(bbox) != 4:
            kept.append(det)
            continue
        suppressed = False
        for k in kept:
            kb = k.get("bbox") or []
            if len(kb) != 4:
                continue
            if _iou(bbox, kb) > IOU_DEDUP_THRESHOLD:
                suppressed = True
                break
            if (_containment(bbox, kb) > CONTAINMENT_THRESHOLD and
                    _share_label_word(det["label"], k["label"])):
                suppressed = True
                break
        if not suppressed:
            kept.append(det)
    return kept


# ── Text builder ──────────────────────────────────────────────────────────────

def _build_scene_graph_text(active: list[dict]) -> str:
    """Build scene_graph text from post-NMS detection dicts, sorted by depth_rank.

    Each detection becomes its own numbered line — same-label objects (e.g.
    multiple people) are listed individually so their spatial positions are
    preserved for the VLM.

    Dict keys: label, position, confidence, depth_value, depth_rank, area_rank,
               free_space.
    """
    ordered = sorted(active, key=lambda d: d.get("depth_rank") or 9999)
    lines = [f"SCENE GRAPH ({len(ordered)} objects detected):"]
    for i, d in enumerate(ordered, start=1):
        depth_val = d.get("depth_value")
        depth_str = f"{depth_val:.3f}" if depth_val is not None else "n/a"
        lines.append(
            f"{i}. {d['label']} [{d['position']}] "
            f"conf={d['confidence']:.2f} depth={depth_str} "
            f"depth_rank={d['depth_rank']} area_rank={d['area_rank']}"
        )
        if d.get("free_space"):
            lines.append(f"   {d['free_space']}")
    return "\n".join(lines)


def build_scene_graph_from_dicts(detections: list[dict]) -> str:
    """Rebuild scene_graph text from scene_detections JSON dicts.

    Applies IoU-based deduplication to remove overlapping near-duplicate labels
    (e.g. hotel / hotel resort / resort → just hotel resort), then lists every
    surviving detection individually sorted by depth_rank.

    Used by rebuild_scene_graphs.py to update existing parquets without
    re-running the full CV pipeline.
    """
    if not detections:
        return "SCENE GRAPH: No objects detected."
    return _build_scene_graph_text(_nms_dicts(detections))


def build_scene_graph(detections: list[Detection]) -> str:
    """Serialize detections to a compact text block for VLM prompts.

    Applies the same IoU deduplication as build_scene_graph_from_dicts(), then
    lists every surviving detection individually sorted by depth_rank.
    """
    if not detections:
        return "SCENE GRAPH: No objects detected."

    dicts = [
        {
            "label": d.label,
            "bbox": list(d.bbox),
            "confidence": d.confidence,
            "depth_value": d.depth_value,
            "position": d.position,
            "depth_rank": d.depth_rank,
            "area_rank": d.area_rank,
            "free_space": d.free_space or "",
        }
        for d in detections
    ]
    return _build_scene_graph_text(_nms_dicts(dicts))
