from __future__ import annotations

from google import genai
from google.genai import types

from datagen.vlm.base import VLMBackend


class GeminiBackend(VLMBackend):
    def __init__(self, cfg) -> None:
        self.client = genai.Client(api_key=cfg.gemini_api_key)
        self.model = cfg.vlm_model
        self.prompt = cfg.vlm_prompt

    def relabel(self, image_bytes: bytes, original_caption: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                self.prompt,
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            ],
        )
        return response.text.strip()
