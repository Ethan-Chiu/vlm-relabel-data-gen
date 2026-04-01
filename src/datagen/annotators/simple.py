from __future__ import annotations

from datagen.vlm.base import VLMBackend


class SimpleAnnotator:
    """One VLM call per image using vlm_prompt from config. Outputs new_caption."""

    CAPTION_TYPES = ("new_caption",)

    def __init__(self, backend: VLMBackend, prompt: str) -> None:
        self._backend = backend
        self._prompt = prompt

    def annotate(self, image_bytes: bytes, original_caption: str) -> dict:
        return {"new_caption": self._backend.call(image_bytes, self._prompt)}
