"""Standard library urllib backend."""

import json
from urllib import request
from typing import Any

from ollama_extract.backends import OllamaBackend, BackendResponse


class UrllibBackend(OllamaBackend):
    name = "urllib"

    def generate(self, prompt: str, json_schema: dict[str, Any]) -> BackendResponse:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": json_schema,
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.ollama_url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return BackendResponse(body=body, content=body.get("response", ""))


__all__ = ["UrllibBackend"]
