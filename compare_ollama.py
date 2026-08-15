"""Benchmark comparison: urllib vs requests vs ollama for calling Ollama API.

Each backend performs the same extraction task (Movie model) and supports
JSON-schema structured output.  Results are printed in a comparison table.
"""

import json
import time
import logging
from urllib import request
from typing import Any

import requests as requests_lib
import ollama
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationError,
    field_validator,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "gemma4:12b"
MAX_RETRIES = 3

TEST_INPUTS = [
    "I just watched 'Inception' — a 2010 sci-fi heist film about stealing secrets from dreams.",
    "'Pulp Fiction' (1994) is Quentin Tarantino's nonlinear crime film that changed indie cinema.",
    "That 2008 film 'WALL-E' from Pixar — a robot falls in love and saves Earth — animated post-apocalyptic romance.",
    "A 1997 fantasy adventure about a young wizard who gets a letter to attend magic school and faces a dark wizard.",
    "omg 'The Shining' 1980 jack nicholson axe murder hotel horror thriller.",
]


# ---------------------------------------------------------------------------
#  Pydantic model
# ---------------------------------------------------------------------------

class Movie(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_default=True)

    title: str = Field(..., description="The title of the movie", min_length=1, max_length=200)
    year: StrictInt = Field(..., description="The release year of the movie", ge=1888, le=2100)
    genres: list[str] = Field(..., description="A list of movie genres", min_length=1, max_length=5)
    summary: str = Field(..., description="A short one-sentence summary", min_length=1, max_length=500)

    @field_validator("genres")
    @classmethod
    def _normalize_genres(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("At least one genre is required")
        return [g.strip() for g in v]


def _build_prompt(raw_text: str, schema: dict[str, Any]) -> str:
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


# ---------------------------------------------------------------------------
#  Backend 1 — urllib (standard library)
# ---------------------------------------------------------------------------

def run_urllib(texts: list[str]) -> tuple[int, float]:
    schema = Movie.model_json_schema()
    prompt_template = _build_prompt

    success, total_time = 0, 0.0
    for t in texts:
        last_err: Exception | None = None
        for _ in range(MAX_RETRIES):
            try:
                start = time.perf_counter()
                payload = {"model": MODEL_NAME, "prompt": prompt_template(t, schema), "stream": False, "format": schema}
                data = json.dumps(payload).encode("utf-8")
                req = request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
                with request.urlopen(req) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                content = body.get("response", "").strip()
                Movie.model_validate_json(content)
                total_time += time.perf_counter() - start
                success += 1
                break
            except Exception as e:
                last_err = e
        else:
            logger.warning(f"urllib failed on: {t[:40]}... — {last_err}")

    return success, total_time


# ---------------------------------------------------------------------------
#  Backend 2 — requests
# ---------------------------------------------------------------------------

def run_requests(texts: list[str]) -> tuple[int, float]:
    schema = Movie.model_json_schema()

    success, total_time = 0, 0.0
    for t in texts:
        last_err: Exception | None = None
        for _ in range(MAX_RETRIES):
            try:
                start = time.perf_counter()
                resp = requests_lib.post(
                    OLLAMA_URL,
                    json={"model": MODEL_NAME, "prompt": _build_prompt(t, schema), "stream": False, "format": schema},
                    headers={"Content-Type": "application/json"},
                    timeout=120,
                )
                resp.raise_for_status()
                body = resp.json()
                content = body.get("response", "").strip()
                Movie.model_validate_json(content)
                total_time += time.perf_counter() - start
                success += 1
                break
            except Exception as e:
                last_err = e
        else:
            logger.warning(f"requests failed on: {t[:40]}... — {last_err}")

    return success, total_time


# ---------------------------------------------------------------------------
#  Backend 3 — ollama library
# ---------------------------------------------------------------------------

def run_ollama_lib(texts: list[str]) -> tuple[int, float]:
    schema = Movie.model_json_schema()

    success, total_time = 0, 0.0
    for t in texts:
        last_err: Exception | None = None
        for _ in range(MAX_RETRIES):
            try:
                start = time.perf_counter()
                resp = ollama.generate(
                    model=MODEL_NAME,
                    prompt=_build_prompt(t, schema),
                    format=schema,
                    stream=False,
                )
                content = resp.get("response", "").strip()
                Movie.model_validate_json(content)
                total_time += time.perf_counter() - start
                success += 1
                break
            except Exception as e:
                last_err = e
        else:
            logger.warning(f"ollama-lib failed on: {t[:40]}... — {last_err}")

    return success, total_time


# ---------------------------------------------------------------------------
#  Runner
# ---------------------------------------------------------------------------

def main() -> None:
    backends = [
        ("urllib        ", run_urllib),
        ("requests      ", run_requests),
        ("ollama lib    ", run_ollama_lib),
    ]

    print(f"Model: {MODEL_NAME}  |  Iterations per backend: {len(TEST_INPUTS)}")
    print(f"{'Backend':<16} {'Success':>8} {'Time (s)':>10} {'Avg (s)':>10}")
    print("-" * 46)

    results: dict[str, tuple[int, float]] = {}
    for label, fn in backends:
        s, t = fn(TEST_INPUTS)
        avg = t / s if s else float("inf")
        results[label.strip()] = (s, t)
        print(f"{label} {s:>8} {t:>10.2f} {avg:>10.2f}")

    total = sum(r[0] for r in results.values())
    print(f"\nTotal successful extractions across all backends: {total}/{len(TEST_INPUTS) * 3}")


if __name__ == "__main__":
    main()
