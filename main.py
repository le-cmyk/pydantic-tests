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
    test_inputs = [
        "I just watched 'Inception' - it's a sci-fi heist movie from 2010.",
        "Just saw 'The Godfather' — a crime drama from 1972 that every aspiring filmmaker should study.",
        "Last night I caught 'Parasite' on Netflix. It's a 2019 Korean thriller-comedy with incredible social commentary.",
        "'Pulp Fiction' from 1994 is a nonlinear crime film by Tarantino that changed indie cinema forever.",
        "Watched 'Spirited Away' yesterday — a 2001 Studio Ghibli animated fantasy masterpiece by Hayao Miyazaki.",
        "I saw 'Mad Max: Fury Road' last week; a high-octane 2015 action film directed by George Miller.",
        "'The Dark Knight' (2008) is a superhero film where Heath Ledger's Joker steals the show.",
        "Just finished 'Everything Everywhere All at Once' — a 2022 absurdist comedy sci-fi multiverse adventure.",
        "'Forrest Gump' is a 1994 American comedy-drama following a man with a low IQ through pivotal historical events.",
        "Watched 'Blade Runner 2049' from 2017, a neo-noir sci-fi sequel that's visually stunning.",
        "I finally saw 'Casablanca' — a 1942 wartime romantic drama that never gets old.",
        "'The Lord of the Rings: The Fellowship of the Ring' (2001) kicks off Jackson's epic fantasy trilogy.",
        "Just rewatched 'Fight Club' — David Fincher's 1999 psychological thriller about consumerism and identity.",
        "'Amélie' is a 2001 French romantic comedy that's whimsical and charming throughout.",
        "I caught 'Get Out' on a late-night stream. It's Jordan Peele's 2017 horror-comedy social thriller debut.",
        "'La La Land' (2016) is a modern musical romance that pays homage to classic Hollywood.",
        "Watched 'The Matrix' from 1999 — a groundbreaking cyberpunk sci-fi action film with innovative visuals.",
        "'Interstellar' (2014) is Christopher Nolan's space epic about love, time, and survival.",
        "Just saw 'Moonlight' — a 2016 coming-of-age drama told in three chapters of a young man's life.",
        "'Arrival' (2016) is a cerebral sci-fi film about linguistics, time, and first contact.",
    ]

    count_success = 0
    count_failures = 0

    for i, input_text in enumerate(test_inputs):
        try:
            movie = get_movie_data(input_text)
            count_success += 1
            print(f"[{i:2d}] {movie.title} ({movie.year}) — {', '.join(movie.genres)}")
        except Exception as e:
            count_failures += 1
            print(f"[{i:2d}] Failed: {e}")

    print(f"\nResults: {count_success} succeeded, {count_failures} failed")


if __name__ == "__main__":
    main()
