"""Scene understanding package: RAM++ → GroundingDINO → SAM → Depth Anything V2."""

from datagen.scene.extractor import SceneExtractor
from datagen.scene.models import Detection

__all__ = ["SceneExtractor", "Detection"]
