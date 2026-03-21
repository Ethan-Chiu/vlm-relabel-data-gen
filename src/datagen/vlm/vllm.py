from __future__ import annotations

import base64

from openai import OpenAI

from datagen.vlm.base import VLMBackend


class VLLMBackend(VLMBackend):
    def __init__(self, cfg) -> None:
        self.client = OpenAI(base_url=cfg.vlm_base_url, api_key="none")
        self.model = cfg.vlm_model
        self.prompt = cfg.vlm_prompt

    def call(self, image_bytes: bytes, prompt: str) -> str:
        b64 = base64.b64encode(image_bytes).decode()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.choices[0].message.content.strip()
