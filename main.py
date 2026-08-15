import json
import logging
from urllib import request
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationError,
    field_validator,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma4:12b"
MAX_RETRIES = 3


class Movie(BaseModel):
    """A movie extracted from natural-language text."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
    )

    title: str = Field(
        ...,
        description="The title of the movie",
        min_length=1,
        max_length=200,
        examples=["Inception"],
    )
    year: StrictInt = Field(
        ...,
        description="The release year of the movie",
        ge=1888,
        le=2100,
        examples=[2010],
    )
    genres: list[str] = Field(
        ...,
        description="A list of movie genres",
        min_length=1,
        max_length=5,
        examples=[["sci-fi", "heist"]],
    )
    summary: str = Field(
        ...,
        description="A short one-sentence summary of the movie",
        min_length=1,
        max_length=500,
        examples=["A thief who steals secrets through dreams."],
    )

    @field_validator("genres")
    @classmethod
    def _normalize_genres(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("At least one genre is required")
        return [g.strip() for g in v]


def build_prompt(raw_text: str, schema: dict[str, Any]) -> str:
    """Build a structured extraction prompt that includes the JSON schema."""
    return f"""You are a data extraction assistant.

Extract the movie information from the text below and return ONLY valid JSON that conforms to this JSON Schema:

{json.dumps(schema, indent=2)}

Rules:
1. Output ONLY valid JSON — no markdown fences, no preamble, no explanation.
2. Match the schema exactly; extra fields will cause validation to fail.
3. If a value is missing, use "Unknown" (for strings) or null (for integers).

Text to extract from:
---
{raw_text}
---"""


def call_ollama(prompt: str, json_schema: dict[str, Any]) -> str:
    """Send a request to Ollama's /api/generate endpoint and return the raw response text."""
    payload: dict[str, Any] = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": json_schema,
    }

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    with request.urlopen(req) as response:
        raw_response = json.loads(response.read().decode("utf-8"))
        return raw_response.get("response", "").strip()


def get_movie_data(raw_text: str) -> Movie:
    """Extract and validate a Movie from natural-language text via Ollama with retry."""
    json_schema = Movie.model_json_schema()
    prompt = build_prompt(raw_text, json_schema)

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            content = call_ollama(prompt, json_schema)
            if not content:
                raise ValueError("Ollama returned an empty response")

            return Movie.model_validate_json(content)

        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES} — parse error: {e}")
        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES} — unexpected error: {e}")

    raise RuntimeError(f"All {MAX_RETRIES} attempts failed. Last error: {last_error}")


def main() -> None:
    input_text = "I just watched 'Inception' - it's a sci-fi heist movie from 2010."

    count_success = 0
    count_failures = 0

    for i in range(10):
        try:
            movie = get_movie_data(input_text)
            count_success += 1
            print(f"[{i}] {movie.title} ({movie.year}) — {', '.join(movie.genres)}")
        except Exception as e:
            count_failures += 1
            print(f"[{i}] Failed: {e}")

    print(f"\nResults: {count_success} succeeded, {count_failures} failed")


if __name__ == "__main__":
    main()
