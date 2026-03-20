from datagen.config import Config
from datagen.vlm.base import VLMBackend
from datagen.vlm.gemini import GeminiBackend
from datagen.vlm.openai_backend import OpenAIBackend
from datagen.vlm.vllm import VLLMBackend

_BACKENDS: dict[str, type[VLMBackend]] = {
    "gemini": GeminiBackend,
    "openai": OpenAIBackend,
    "vllm": VLLMBackend,
}


def get_backend(cfg: Config) -> VLMBackend:
    if cfg.vlm_backend not in _BACKENDS:
        raise ValueError(f"Unknown backend '{cfg.vlm_backend}'. Choose from: {list(_BACKENDS)}")
    return _BACKENDS[cfg.vlm_backend](cfg)
