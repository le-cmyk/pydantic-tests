"""Benchmark comparison: urllib vs requests vs ollama for calling Ollama API.

Each backend performs the same extraction task (Movie model) and supports
JSON-schema structured output.  Results are printed in a comparison table.
"""

import json
import os
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
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:0.5b")
MAX_RETRIES = 3

TEST_INPUTS = [
    # 0 — title only in quotes, no genres
    "I saw 'Amélie' in 2001 — whimsical French film, no genres mentioned.",
    # 1 — director reference, must infer title
    "That Christopher Nolan 2010 film about dream invasion — you know, the spinning top ending.",
    # 2 — year as Roman numeral
    "The 1994 film 'Pulp Fiction' (MCMXCIV) — wait no that's not right, it's just 1994, a crime anthology.",
    # 3 — multiple movies, must pick the one described
    "Like 'Alien' from 1979 and 'Aliens' from 1992, 'The Shining' (1980) is Kubrick's horror masterpiece — not the other two.",
    # 4 — title implied through plot, never named
    "A 1999 sci-fi action film about a hacker who learns reality is an illusion and takes a red pill — you know the one.",
    # 5 — genre described as a feeling
    "That 2007 film 'There Will Be Blood' — it's oozing with greed, capitalism, and oil-soaked madness.",
    # 6 — very long compound title, no year given
    "The Lord of the Rings: The Return of the King concludes the trilogy — epic fantasy war from 2003.",
    # 7 — deliberate year misinformation
    "Everyone says 'Titanic' is from 1996, but James Cameron's epic disaster romance is actually 1997.",
    # 8 — title in a URL-like format
    "I found a file called 'the-matrix-1999-dvdrip' — that's the cyberpunk action classic right?",
    # 9 — text-speak / abbreviations
    "lmao 'Get Out' 2017 best horror social thriller everrr jordan peele killed it.",
    # 10 — non-English description, English title
    "이 영화 '인셉션' 2010 년 작품은 드림 속에서 비밀을 훔치는 팀을 보여줘요. sci-fi heist.",
    # 11 — genre as single-word emotions
    "That 2012 film 'Argo' feels like suspense, tension, and nervous anxiety wrapped in a CIA thriller.",
    # 12 — nothing but a year (impossible extraction)
    "1994.",
    # 13 — two movies mashed into one sentence
    "It's like 'Casablanca' (1942) meets 'The English Patient' (1996) — two wartime romances, make a 1942 romantic drama about a cynical bartender.",
    # 14 — title is a common word, quoted
    "'Her' (2013) — a man falls in love with his AI operating system in this Spike Jonze sci-fi romance.",
    # 15 — genres described with comma-separated string in narrative
    "The 1972 film 'The Godfather' is crime, drama, and family saga all rolled into one mafia epic.",
    # 16 — year spelled out
    "I saw a movie about a ring that gives you supernatural powers — it's 'The Ring' from two thousand and one, a 2001 supernatural horror thriller remake.",
    # 17 — title is just one letter (impossible)
    "'",  # deliberately invalid
    # 18 — input is a haiku
    "Dark knight rises once, Gotham burned and reborn, 2012 ends the trilogy.",
    # 19 — fake movie, must use null/unknown
    "I just watched 'The Zephyrian Protocol' from 2025 — a time-bending quantum espionage thriller that doesn't exist.",
    # 20 — title has typo in input
    "Just saw 'The Matirx' (1999) — a groundbreaking cyberpunk sci-fi action film with bullet-time.",
    # 21 — genre described through cinematography
    "That 1960 Hitchcock film 'Psycho' — shrieking violins, shower stabbing, psychological horror thriller at the Bates Motel.",
    # 22 — input is a movie review excerpt
    "As Roger Ebert said, 'Spirited Away' (2001) is 'an animated masterpiece where a shy girl navigates a spirit world' — pure magic from Miyazaki.",
    # 23 — title only in parentheses
    "That film from 2008 where the Joker says 'why so serious' — you know the Batman one.",
    # 24 — input mixes real and fictional
    "In 'The Lord of the Rings: The Fellowship of the Ring' (2001), Frodo must destroy the One Ring — fantasy adventure from 2013? No, 2001.",
    # 25 — genre through soundtrack description
    "That 1977 space opera 'Star Wars' with the epic orchestral score, lightsaber duels, and 'the force' — George Lucas's space fantasy epic.",
    # 26 — title in a foreign script with English subtitle
    "看过 2009 年的 'Inception'，这部 sci-fi 电影讲述了一个进入梦中的团队如何窃取秘密。",
    # 27 — nothing but a genre
    "Horror.",
    # 28 — title appears in both opening and closing
    "I just watched 'WALL-E'. That 2008 Pixar film 'WALL-E' — an animated post-apocalyptic romance comedy about a robot who finds love and saves Earth.",
    # 29 — contradictory years + no explicit title
    "That film from 2010, no wait 2011, the Christopher Nolan one about dreams within dreams with a spinning top — release year is 2010.",
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

def show_samples() -> None:
    """Run a few tricky inputs through urllib and print the extracted Movie objects."""
    schema = Movie.model_json_schema()
    tricky = [12, 17, 18, 19, 27]
    print(f"\nSample extractions (model={MODEL_NAME}, backend=urllib):")
    print("-" * 70)
    for idx in tricky:
        text = TEST_INPUTS[idx]
        try:
            payload = {
                "model": MODEL_NAME,
                "prompt": _build_prompt(text, schema),
                "stream": False,
                "format": schema,
            }
            data = json.dumps(payload).encode("utf-8")
            req = request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
            with request.urlopen(req) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            content = body.get("response", "").strip()
            movie = Movie.model_validate_json(content)
            print(f"[{idx:2d}] Input:  {text[:60]}")
            print(f"     Title: {movie.title}")
            print(f"     Year:  {movie.year}")
            print(f"     Genres: {', '.join(movie.genres)}")
            print(f"     Summary: {movie.summary[:80]}")
        except Exception as e:
            print(f"[{idx:2d}] Input:  {text[:60]}")
            print(f"     Error: {e}")
        print()


def main() -> None:
    backends = [
        ("urllib        ", run_urllib),
        ("requests      ", run_requests),
        ("ollama lib    ", run_ollama_lib),
    ]

    n = len(TEST_INPUTS)
    print(f"Model: {MODEL_NAME}  |  Iterations per backend: {n}")
    print(f"{'Backend':<16} {'Success':>8} {'Time (s)':>10} {'Avg (s)':>10}")
    print("-" * 46)

    results: dict[str, tuple[int, float]] = {}
    for label, fn in backends:
        s, t = fn(TEST_INPUTS)
        avg = t / s if s else float("inf")
        results[label.strip()] = (s, t)
        print(f"{label} {s:>8} {t:>10.2f} {avg:>10.2f}")

    total = sum(r[0] for r in results.values())
    print(f"\nTotal successful extractions across all backends: {total}/{n * 3}")

    if os.environ.get("SHOW_SAMPLES"):
        show_samples()


if __name__ == "__main__":
    main()
