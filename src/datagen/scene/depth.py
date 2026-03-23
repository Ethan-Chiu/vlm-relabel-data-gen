"""Depth Anything V2 stage.

Loads the model from HuggingFace (or a local path) and returns a normalized
depth map for an image.  Depth values are in [0, 1] where **lower = closer**
(disparity convention).

Usage::

    estimator = DepthEstimator("depth-anything/Depth-Anything-V2-Large-hf")
    depth_map = estimator.estimate(image_bytes)   # np.ndarray H×W float32
"""

from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image


class DepthEstimator:
    """Wraps Depth Anything V2 via HuggingFace transformers.

    Parameters
    ----------
    model_name_or_path:
        HuggingFace model ID or local directory, e.g.
        ``"depth-anything/Depth-Anything-V2-Large-hf"``
    device:
        ``"cuda"`` or ``"cpu"``
    """

    def __init__(self, model_name_or_path: str, device: str = "cuda") -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._device = device
        self._processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        self._model = (
            AutoModelForDepthEstimation.from_pretrained(model_name_or_path)
            .eval()
            .to(device)
        )

    def estimate(self, image_bytes: bytes) -> np.ndarray:
        """Return a normalized depth map (H×W float32, range [0, 1]).

        Convention: **lower value = closer to camera** (inverted disparity).
        This matches the "near = low depth rank" intuition used in geometry.py.
        """
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self._processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            # predicted_depth: [1, H, W] — raw disparity (higher = closer)
            predicted_depth = outputs.predicted_depth

        # Upsample to original image size
        H, W = pil_img.size[1], pil_img.size[0]
        depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_np = depth.cpu().numpy().astype(np.float32)

        # Normalize to [0, 1] and invert so that 0 = closest, 1 = farthest
        d_min, d_max = depth_np.min(), depth_np.max()
        if d_max > d_min:
            depth_np = (depth_np - d_min) / (d_max - d_min)
        else:
            depth_np = np.zeros_like(depth_np)
        depth_np = 1.0 - depth_np  # invert: low value = near

        return depth_np

    def object_depth(self, depth_map: np.ndarray, mask: np.ndarray | None) -> float:
        """Return the median depth value within a mask (or the image median if no mask)."""
        if mask is not None and mask.any():
            return float(np.median(depth_map[mask]))
        return float(np.median(depth_map))
