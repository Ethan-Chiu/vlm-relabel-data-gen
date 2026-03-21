from abc import ABC, abstractmethod


class VLMBackend(ABC):

    @abstractmethod
    def call(self, image_bytes: bytes, prompt: str) -> str:
        """Send image + prompt to the model, return the response text."""
        ...
