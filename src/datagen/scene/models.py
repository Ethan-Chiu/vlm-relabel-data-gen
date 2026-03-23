"""Data classes for scene understanding outputs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    """One detected object in the scene.

    Populated progressively as each stage of the pipeline runs:
      - label, bbox, confidence: from GroundingDINO
      - mask: from SAM
      - depth_value: from Depth Anything V2 (median within mask)
      - position, depth_rank, area_rank: computed by geometry.py
    """

    label: str
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (pixel coords)
    confidence: float
    mask: np.ndarray | None = field(default=None, repr=False)  # H×W bool

    # Filled in by geometry stage
    depth_value: float | None = None   # relative depth; smaller = closer
    position_h: str = ""               # "left" | "center" | "right"
    position_v: str = ""               # "near" | "mid" | "far"
    depth_rank: int = 0                # 1 = nearest object
    area_rank: int = 0                 # 1 = largest object (by mask area)
    free_space: str = ""               # brief description of surrounding free space

    @property
    def position(self) -> str:
        """Human-readable combined position, e.g. 'left-near'."""
        parts = [p for p in (self.position_h, self.position_v) if p]
        return "-".join(parts) if parts else "unknown"

    @property
    def area(self) -> int:
        """Mask area in pixels (0 if no mask)."""
        if self.mask is None:
            return 0
        return int(self.mask.sum())

    @property
    def bbox_center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
