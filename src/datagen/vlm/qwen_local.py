"""Local Qwen3-VL inference via transformers.

Install dependencies before using this backend:
    uv add transformers accelerate qwen-vl-utils
    # torch must be installed separately with the right CUDA version, e.g.:
    # uv add torch --index-url https://download.pytorch.org/whl/cu124

For higher throughput, prefer the 'vllm' backend with a vLLM server:
    vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000
then set vlm_backend = "vllm" and vlm_model = "Qwen/Qwen3-VL-8B-Instruct".
"""

from __future__ import annotations

import io

from datagen.vlm.base import VLMBackend


class QwenVLLocalBackend(VLMBackend):
    """Runs Qwen3-VL-8B-Instruct locally using transformers.

    Each Ray worker that instantiates this class loads the full model into GPU
    memory. Set concurrency = number of GPUs in config to avoid OOM.
    """

    def __init__(self, cfg) -> None:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prompt = cfg.vlm_prompt
        self.model_name = cfg.vlm_model  # e.g. "Qwen/Qwen3-VL-8B-Instruct"

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",  # requires flash-attn package
            device_map="auto",                        # spreads across available GPUs
            trust_remote_code=True,
        )
        self.model.eval()

    def call(self, image_bytes: bytes, prompt: str) -> str:
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        # Strip input tokens from output
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, output_ids)
        ]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
