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


def build_scene_graph(detections: list[Detection]) -> str:
    """Serialize detections to a compact text block for VLM prompts.

    Example output::

        SCENE GRAPH (4 objects detected):
        1. cup [left-near] conf=0.82 depth_rank=1 area_rank=3
           Free space: Clear space left, above of the cup.
        2. plate [center-mid] conf=0.91 depth_rank=2 area_rank=1
           Free space: Clear space right, below of the plate.
        ...
    """
    if not detections:
        return "SCENE GRAPH: No objects detected."

    lines = [f"SCENE GRAPH ({len(detections)} objects detected):"]
    for i, det in enumerate(detections, start=1):
        depth_str = f"{det.depth_value:.3f}" if det.depth_value is not None else "n/a"
        lines.append(
            f"{i}. {det.label} [{det.position}] "
            f"conf={det.confidence:.2f} depth={depth_str} "
            f"depth_rank={det.depth_rank} area_rank={det.area_rank}"
        )
        if det.free_space:
            lines.append(f"   {det.free_space}")
    return "\n".join(lines)
