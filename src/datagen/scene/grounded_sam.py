"""RAM++ → GroundingDINO → SAM stage.

Outputs a list of Detection objects with label, bbox, confidence, and mask.

Model loading
─────────────
Models are loaded once at construction time and reused across calls.
On a GPU machine set device="cuda"; on CPU set device="cpu" (slow but functional).

Expected weights layout (configured via Config):
  models/
    ram_plus_swin_large_14m.pth       # RAM++ weights
    GroundingDINO_SwinT_OGC.cfg.py   # GroundingDINO config
    groundingdino_swint_ogc.pth       # GroundingDINO weights
    sam_vit_h_4b8939.pth              # SAM weights

Download helpers in scripts/setup_models.py.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from datagen.scene.models import Detection


def _to_pil(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


class GroundedSAM:
    """Wraps RAM++ + GroundingDINO + SAM into a single detect() call.

    Parameters
    ----------
    ram_weights:      path to RAM++ checkpoint (.pth)
    gdino_config:     path to GroundingDINO config (.cfg.py or .py)
    gdino_weights:    path to GroundingDINO checkpoint (.pth)
    sam_weights:      path to SAM checkpoint (.pth)
    sam_type:         SAM variant — "vit_h" | "vit_l" | "vit_b"
    box_threshold:    GroundingDINO box confidence threshold (default 0.30)
    text_threshold:   GroundingDINO text confidence threshold (default 0.25)
    ram_image_size:   input resolution for RAM++ (default 384)
    device:           "cuda" or "cpu"
    """

    def __init__(
        self,
        ram_weights: str | Path,
        gdino_config: str | Path,
        gdino_weights: str | Path,
        sam_weights: str | Path,
        sam_type: str = "vit_h",
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        ram_image_size: int = 384,
        device: str = "cuda",
    ) -> None:
        self._device = device
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold

        self._ram = self._load_ram(ram_weights, ram_image_size, device)
        self._gdino = self._load_gdino(gdino_config, gdino_weights, device)
        self._sam_predictor = self._load_sam(sam_weights, sam_type, device)

    # ── Model loading ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_ram(weights: str | Path, image_size: int, device: str):
        from ram import inference_ram
        from ram.models import ram_plus

        model = ram_plus(pretrained=str(weights), image_size=image_size, vit="swin_l")
        model = model.eval().to(device)
        return model

    @staticmethod
    def _load_gdino(config: str | Path, weights: str | Path, device: str):
        from groundingdino.util.inference import load_model

        model = load_model(str(config), str(weights))
        return model.to(device)

    @staticmethod
    def _load_sam(weights: str | Path, sam_type: str, device: str):
        from segment_anything import SamPredictor, sam_model_registry

        sam = sam_model_registry[sam_type](checkpoint=str(weights))
        sam = sam.to(device)
        return SamPredictor(sam)

    # ── Inference ──────────────────────────────────────────────────────────────

    def detect(self, image_bytes: bytes) -> list[Detection]:
        """Run the full RAM++ → GroundingDINO → SAM pipeline.

        Returns a list of Detection objects (one per detected object).
        """
        pil_img = _to_pil(image_bytes)
        img_array = np.array(pil_img)
        W, H = pil_img.size

        # Stage 1: RAM++ — generate open-vocabulary tags
        tags = self._run_ram(pil_img)
        if not tags:
            return []

        # Stage 2: GroundingDINO — ground tags to bounding boxes
        boxes, confidences, labels = self._run_gdino(pil_img, tags, W, H)
        if len(boxes) == 0:
            return []

        # Stage 3: SAM — segment each bounding box
        masks = self._run_sam(img_array, boxes)

        detections = []
        for i, (box, conf, label) in enumerate(zip(boxes, confidences, labels)):
            mask = masks[i] if i < len(masks) else None
            detections.append(Detection(
                label=label.strip(),
                bbox=tuple(float(v) for v in box),
                confidence=float(conf),
                mask=mask,
            ))
        return detections

    def _run_ram(self, pil_img: Image.Image) -> str:
        """Return comma-separated RAM++ tags."""
        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize((384, 384)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = transform(pil_img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            from ram import inference_ram
            tags, _ = inference_ram(img_tensor, self._ram)
        # inference_ram returns a single string like "apple | banana | table"
        # Normalize to comma-separated for GroundingDINO
        tags_str = tags.replace(" | ", " . ").replace("|", ".").strip(" .")
        return tags_str

    def _run_gdino(
        self, pil_img: Image.Image, tags: str, W: int, H: int
    ) -> tuple[list, list, list]:
        """Return (boxes_xyxy, confidences, labels) in pixel coordinates."""
        import torchvision.transforms as T
        from groundingdino.util.inference import predict

        transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img_tensor = transform(pil_img)

        boxes_cxcywh, logits, phrases = predict(
            model=self._gdino,
            image=img_tensor,
            caption=tags,
            box_threshold=self._box_threshold,
            text_threshold=self._text_threshold,
            device=self._device,
        )

        if len(boxes_cxcywh) == 0:
            return [], [], []

        # Convert from normalized cx,cy,w,h → pixel x1,y1,x2,y2
        boxes_xyxy = []
        for box in boxes_cxcywh:
            cx, cy, w, h = box
            x1 = float((cx - w / 2) * W)
            y1 = float((cy - h / 2) * H)
            x2 = float((cx + w / 2) * W)
            y2 = float((cy + h / 2) * H)
            boxes_xyxy.append((
                max(0.0, x1), max(0.0, y1),
                min(float(W), x2), min(float(H), y2),
            ))

        return boxes_xyxy, logits.tolist(), phrases

    def _run_sam(
        self, img_array: np.ndarray, boxes_xyxy: list[tuple]
    ) -> list[np.ndarray]:
        """Return a list of boolean H×W masks (one per box)."""
        self._sam_predictor.set_image(img_array)

        boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32, device=self._device)
        transformed_boxes = self._sam_predictor.transform.apply_boxes_torch(
            boxes_tensor, img_array.shape[:2]
        )

        with torch.no_grad():
            masks_tensor, _, _ = self._sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

        # masks_tensor: [N, 1, H, W] bool
        return [masks_tensor[i, 0].cpu().numpy().astype(bool) for i in range(len(masks_tensor))]
