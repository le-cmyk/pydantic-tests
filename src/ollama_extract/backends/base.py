"""Abstract base classes for Ollama backends."""

from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass

from pydantic import ValidationError

from ollama_extract.model import Movie

__all__ = ["BackendName", "BackendResponse", "OllamaBackend"]

BackendName = str


@dataclass
class BackendResponse:
    """Normalized response from any backend."""
    body: dict[str, Any]
    content: str

    @property
    def prompt_tokens(self) -> int | None:
        return self.body.get("prompt_eval_count")

    @property
    def eval_tokens(self) -> int | None:
        return self.body.get("eval_count")

    @property
    def total_duration_ms(self) -> int | None:
        td = self.body.get("total_duration")
        return td // 1_000_000 if td else None


class OllamaBackend(ABC):
    """Abstract base class for Ollama HTTP backends."""

    name: str

    def __init__(self, model_name: str, ollama_url: str) -> None:
        self.model_name = model_name
        self.ollama_url = ollama_url

    @abstractmethod
    def generate(self, prompt: str, json_schema: dict[str, Any]) -> BackendResponse:
        """Send a prompt to Ollama with structured output constraints."""

    def validate(self, content: str) -> Movie:
        """Parse and validate the LLM response against the Movie model."""
        content = content.strip()
        if not content:
            raise ValueError("Ollama returned an empty response")
        return Movie.model_validate_json(content)
