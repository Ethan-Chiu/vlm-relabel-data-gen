"""SceneExtractor: orchestrates all scene understanding stages.

Pipeline
────────
image_bytes
  → GroundedSAM.detect()        → list[Detection]  (label, bbox, mask, confidence)
  → DepthEstimator.estimate()   → depth_map (H×W)
  → assign depth_value per detection (median depth in mask)
  → assign_geometry()           → position, ranks, free_space (in-place)
  → build_scene_graph()         → str

The resulting scene_graph string is passed to the VLM annotation prompts.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from datagen.scene.depth import DepthEstimator
from datagen.scene.geometry import assign_geometry, build_scene_graph
from datagen.scene.grounded_sam import GroundedSAM
from datagen.scene.models import Detection


class SceneExtractor:
    """Loads all models once and exposes extract() for per-image inference.

    Parameters
    ----------
    ram_weights:        path to RAM++ checkpoint
    gdino_config:       path to GroundingDINO config
    gdino_weights:      path to GroundingDINO checkpoint
    sam_weights:        path to SAM checkpoint
    depth_model:        HuggingFace model ID or local path for Depth Anything V2
    sam_type:           SAM variant — "vit_h" | "vit_l" | "vit_b"
    box_threshold:      GroundingDINO box confidence threshold
    text_threshold:     GroundingDINO text confidence threshold
    ram_image_size:     input resolution for RAM++
    device:             "cuda" or "cpu"
    """

    def __init__(
        self,
        ram_weights: str,
        gdino_config: str,
        gdino_weights: str,
        sam_weights: str,
        depth_model: str = "depth-anything/Depth-Anything-V2-Large-hf",
        sam_type: str = "vit_h",
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        ram_image_size: int = 384,
        device: str = "cuda",
    ) -> None:
        self._grounded_sam = GroundedSAM(
            ram_weights=ram_weights,
            gdino_config=gdino_config,
            gdino_weights=gdino_weights,
            sam_weights=sam_weights,
            sam_type=sam_type,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            ram_image_size=ram_image_size,
            device=device,
        )
        self._depth = DepthEstimator(depth_model, device=device)

    def extract(self, image_bytes: bytes) -> tuple[list[Detection], str]:
        """Run the full scene understanding pipeline.

        Returns
        -------
        detections : list[Detection]
            All detected objects with geometry fields populated.
        scene_graph : str
            Compact text representation suitable for VLM prompts.
        """
        # Detect objects
        detections = self._grounded_sam.detect(image_bytes)

        if not detections:
            return [], "SCENE GRAPH: No objects detected."

        # Estimate depth map
        depth_map = self._depth.estimate(image_bytes)

        # Assign per-object depth from mask
        for det in detections:
            det.depth_value = self._depth.object_depth(depth_map, det.mask)

        # Assign spatial positions, ranks, free space
        pil_img = Image.open(io.BytesIO(image_bytes))
        H, W = pil_img.size[1], pil_img.size[0]
        assign_geometry(detections, (H, W))

        scene_graph = build_scene_graph(detections)
        return detections, scene_graph
