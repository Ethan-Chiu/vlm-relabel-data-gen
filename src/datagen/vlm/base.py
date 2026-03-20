from abc import ABC, abstractmethod


class VLMBackend(ABC):
    @abstractmethod
    def relabel(self, image_bytes: bytes, original_caption: str) -> str:
        """Return a new caption for the given image bytes."""
        ...
