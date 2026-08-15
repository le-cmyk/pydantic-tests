"""Official ollama Python library backend."""

from typing import Any

import ollama

from ollama_extract.backends import OllamaBackend, BackendResponse


class OllamaLibraryBackend(OllamaBackend):
    name = "ollama"

    def generate(self, prompt: str, json_schema: dict[str, Any]) -> BackendResponse:
        resp = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            format=json_schema,
            stream=False,
        )
        return BackendResponse(body=resp, content=resp.get("response", ""))


__all__ = ["OllamaLibraryBackend"]
