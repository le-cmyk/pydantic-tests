"""requests library backend."""

import requests as requests_lib
from typing import Any

from ollama_extract.backends import OllamaBackend, BackendResponse


class RequestsBackend(OllamaBackend):
    name = "requests"

    def generate(self, prompt: str, json_schema: dict[str, Any]) -> BackendResponse:
        resp = requests_lib.post(
            self.ollama_url,
            json={"model": self.model_name, "prompt": prompt, "stream": False, "format": json_schema},
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        resp.raise_for_status()
        body = resp.json()
        return BackendResponse(body=body, content=body.get("response", ""))


__all__ = ["RequestsBackend"]
