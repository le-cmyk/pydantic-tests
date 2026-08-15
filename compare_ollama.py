"""Benchmark comparison: urllib vs requests vs ollama for calling Ollama API.

Each backend performs the same extraction task (Movie model) and supports
JSON-schema structured output.  Results are printed in comparison tables
with color-coded differences.

Usage:
    uv run compare_ollama.py                    # 30 hand-crafted edge-cases
    uv run compare_ollama.py --count 3000       # 3000 programmatically generated inputs
    OLLAMA_MODEL=gemma4:12b uv run compare_ollama.py --count 50
    SHOW_SAMPLES=1 uv run compare_ollama.py     # also show sample extractions
"""

import argparse
import json
import os
import time
import random
import logging
import statistics
from dataclasses import dataclass, field
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
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:0.5b")
MAX_RETRIES = 3
USE_COLOR = sys.stdout.isatty() if (sys := __import__("sys")) else False


# ---------------------------------------------------------------------------
#  Hand-crafted edge-case inputs (default test suite; replaced by generated
#  inputs when --count > len(TEST_INPUTS))
# ---------------------------------------------------------------------------

TEST_INPUTS = [
    #  0 — title only in quotes, no genres
    "I saw 'Amélie' in 2001 — whimsical French film, no genres mentioned.",
    #  1 — director reference, must infer title
    "That Christopher Nolan 2010 film about dream invasion — you know, the spinning top ending.",
    #  2 — year as Roman numeral
    "The 1994 film 'Pulp Fiction' (MCMXCIV) — wait no that's not right, it's just 1994, a crime anthology.",
    #  3 — multiple movies, must pick the one described
    "Like 'Alien' from 1979 and 'Aliens' from 1992, 'The Shining' (1980) is Kubrick's horror masterpiece — not the other two.",
    #  4 — title implied through plot, never named
    "A 1999 sci-fi action film about a hacker who learns reality is an illusion and takes a red pill — you know the one.",
    #  5 — genre described as a feeling
    "That 2007 film 'There Will Be Blood' — it's oozing with greed, capitalism, and oil-soaked madness.",
    #  6 — very long compound title, no year given
    "The Lord of the Rings: The Return of the King concludes the trilogy — epic fantasy war from 2003.",
    #  7 — deliberate year misinformation
    "Everyone says 'Titanic' is from 1996, but James Cameron's epic disaster romance is actually 1997.",
    #  8 — title in a URL-like format
    "I found a file called 'the-matrix-1999-dvdrip' — that's the cyberpunk action classic right?",
    #  9 — text-speak / abbreviations
    "lmao 'Get Out' 2017 best horror social thriller everrr jordan peele killed it.",
    # 10 — non-English description, English title
    "이 영화 '인셍' 2010 년 작품은 드림 속에서 비밀을 훔치는 팀을 보여줘요. sci-fi heist.",
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
#  ANSI colors
# ---------------------------------------------------------------------------

class C:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def _c(text: str, color: str) -> str:
    if not USE_COLOR:
        return text
    return f"{color}{text}{C.RESET}"


# ---------------------------------------------------------------------------
#  Test input generation
# ---------------------------------------------------------------------------

REAL_TITLES = [
    "Inception", "The Godfather", "Pulp Fiction", "The Dark Knight", "Fight Club",
    "Forrest Gump", "The Matrix", "Interstellar", "Spirited Away", "Amélie",
    "Casablanca", "The Lord of the Rings: The Fellowship of the Ring",
    "The Lord of the Rings: The Two Towers", "The Lord of the Rings: The Return of the King",
    "Mad Max: Fury Road", "The Shawshank Redemption", "The Prestige", "Memento",
    "The Departed", "There Will Be Blood", "No Country for Old Men", "There",
    "Her", "Moonlight", "Arrival", "Blade Runner 2049", "La La Land",
    "Get Out", "Us", "Black Panther", "Parasite", "1917", "Joker", "Dune",
    "Everything Everywhere All at Once", "The Whale", "Avatar: The Way of Water",
    "Top Gun: Maverick", "Elvis", "The Batman", "Doctor Strange in the Multiverse of Madness",
    "Spider-Man: No Way Home", "Dune: Part Two", "Oppenheimer", "Barbie",
    "Guardians of the Galaxy", "The Revenant", "The Social Network", "The King's Speech",
]

FAKE_TITLES = [
    "The Zephyrian Protocol", "Quantum Paradox", "The Crimson Void",
    "Echoes of Yesterday", "The Last Horizon", "Project Nebula",
    "Shadow of the Colossus", "The Eternal Circuit", "Neon Genesis",
]

PLOTS = [
    "a thief who steals secrets from dreams",
    "a botanist who discovers a mysterious plant with deadly consequences",
    "a detective who hunts a serial killer copycat",
    "a young lion prince who must embrace his destiny",
    "a hacker who learns reality is an illusion",
    "a scientist who invents a time machine",
    "a retired assassin who comes out of retirement",
    "a journalist who uncovers a conspiracy",
    "a teacher who discovers a student's dark secret",
    "a pilot who must deliver humanity's last hope",
    "a musician who loses his hearing and finds a new voice",
    "a mother who seeks justice for her family",
    "a scientist who creates artificial life",
    "a thief who steals memories",
    "a detective who can see the dead",
    "a young witch who must choose between love and duty",
]

GENRE_DESCRIPTIONS = [
    "sci-fi", "crime", "drama", "action", "horror", "comedy", "thriller",
    "fantasy", "romance", "mystery", "adventure", "animation", "western",
    "war", "musical", "documentary", "biography", "history",
]

YEAR_STRINGS = [
    "2010", "1994", "2001", "2017", "1997", "1977", "1999", "2008",
    "1942", "1972", "1985", "2012", "2013", "2016", "1996", "2014",
    "two thousand and one", "nineteen ninety-four", "MMX", "MCMLXIV",
]

ROMAN_NUMERALS = {
    "1942": "MCMXLII", "1972": "MCMLXXII", "1977": "MCMLXXVII", "1985": "MCMLXXXV",
    "1994": "MCMXCIV", "1996": "MCMXCVI", "1997": "MCMXCVII", "1999": "MCMXCIX",
    "2001": "MMI", "2010": "MMX", "2012": "MMXII", "2013": "MMXIII",
    "2014": "MMXIV", "2016": "MMXVI", "2017": "MMXVII",
}

DIRECTORS = [
    "Christopher Nolan", "Quentin Tarantino", "Steven Spielberg", "Martin Scorsese",
    "Ridley Scott", "James Cameron", "David Fincher", "The Coen Brothers",
    "Hayao Miyazaki", "Denis Villeneuve", "Greta Gerwig", "Jordan Peele",
    "Bong Joon-ho", "Chloé Zhao", "Taika Waititi",
]

CRITICS = ["Roger Ebert", "Peter Travers", "A.O. Scott", "Richard Roeper", "Owen Gleiberman", "Kenneth Turan"]


def _gen_standard(rng: random.Random) -> str:
    title = rng.choice(REAL_TITLES + FAKE_TITLES)
    year = rng.choice(YEAR_STRINGS)
    genre = rng.choice(GENRE_DESCRIPTIONS)
    plot = rng.choice(PLOTS)
    return f"I just watched '{title}' — it's a {year} {genre} film about {plot}."


def _gen_director(rng: random.Random) -> str:
    director = rng.choice(DIRECTORS)
    year = rng.choice(YEAR_STRINGS)
    plot = rng.choice(PLOTS)
    return f"That {director} {year} film where {plot} — you know the one."


def _gen_url_style(rng: random.Random) -> str:
    title = rng.choice(REAL_TITLES)
    year = rng.choice(YEAR_STRINGS)
    slug = title.lower().replace(" ", "-").replace(":", "").replace("'", "")
    return f"I found a file: {slug}-{year}-dvdrip.mkv — pretty sure that's the right movie?"


def _gen_text_speak(rng: random.Random) -> str:
    title = rng.choice(REAL_TITLES)
    year = rng.choice(YEAR_STRINGS)
    genre = rng.choice(GENRE_DESCRIPTIONS)
    return f"omg just saw '{title}' {year} so {genre} rn best movie ever lol"


def _gen_review(rng: random.Random) -> str:
    critic = rng.choice(CRITICS)
    title = rng.choice(REAL_TITLES + FAKE_TITLES)
    year = rng.choice(YEAR_STRINGS)
    genre = rng.choice(GENRE_DESCRIPTIONS)
    return f"As {critic} said, '{title}' ({year}) is a {genre} masterpiece."


def _gen_no_title(rng: random.Random) -> str:
    year = rng.choice(YEAR_STRINGS)
    genre = rng.choice(GENRE_DESCRIPTIONS)
    plot = rng.choice(PLOTS)
    return f"A {year} {genre} film about {plot}."


def _gen_roman_year(rng: random.Random) -> str:
    title = rng.choice(REAL_TITLES)
    year_key = rng.choice(list(ROMAN_NUMERALS.keys()))
    roman = ROMAN_NUMERALS[year_key]
    return f"That {year_key} film '{title}' ({roman}) — a must-see."


def _gen_contradictory(rng: random.Random) -> str:
    title = rng.choice(REAL_TITLES)
    y1 = rng.choice(YEAR_STRINGS)
    y2 = rng.choice(YEAR_STRINGS)
    while y2 == y1:
        y2 = rng.choice(YEAR_STRINGS)
    return f"I saw '{title}' in {y1}... no wait, it was actually {y2}."


def _gen_haiku(rng: random.Random) -> str:
    title = rng.choice(REAL_TITLES)
    year = rng.choice(YEAR_STRINGS)
    return f"{title[:5]} whispers in silence, {year} cinema defines art, masterpiece seen."


def _gen_minimal(rng: random.Random) -> str:
    option = rng.randint(0, 3)
    if option == 0:
        return rng.choice(YEAR_STRINGS) + "."
    if option == 1:
        return "'"
    if option == 2:
        return rng.choice(GENRE_DESCRIPTIONS) + "."
    title = rng.choice(REAL_TITLES)
    return title


def _gen_fake_movie(rng: random.Random) -> str:
    title = rng.choice(FAKE_TITLES)
    year = rng.choice(YEAR_STRINGS)
    genre = rng.choice(GENRE_DESCRIPTIONS)
    plot = rng.choice(PLOTS)
    return f"I just watched '{title}' from {year} — a {genre} film about {plot} that does not exist."


def _gen_multiple_movies(rng: random.Random) -> str:
    t1 = rng.choice(REAL_TITLES)
    t2 = rng.choice(REAL_TITLES)
    while t2 == t1:
        t2 = rng.choice(REAL_TITLES)
    target = rng.choice(REAL_TITLES + FAKE_TITLES)
    year = rng.choice(YEAR_STRINGS)
    return f"Like '{t1}' and '{t2}', '{target}' ({year}) is the one that stands out."


def _gen_foreign_mixed(rng: random.Random) -> str:
    title = rng.choice(REAL_TITLES)
    year = rng.choice(YEAR_STRINGS)
    phrases_ko = ["이 영화", "이제야 봤어", "정말 좋은 영화", "추천해요"]
    phrases_zh = ["看过", "真的很好看", "强烈推荐", "这是一部"]
    phrase = rng.choice(phrases_ko + phrases_zh)
    return f"{phrase} '{title}' {year} — a truly unforgettable film."


_GENERATORS = [
    _gen_standard,
    _gen_director,
    _gen_url_style,
    _gen_text_speak,
    _gen_review,
    _gen_no_title,
    _gen_roman_year,
    _gen_contradictory,
    _gen_haiku,
    _gen_minimal,
    _gen_fake_movie,
    _gen_multiple_movies,
    _gen_foreign_mixed,
]


def generate_inputs(n: int, seed: int = 42) -> list[str]:
    """Generate *n* diverse movie-extraction test inputs using a seeded RNG."""
    rng = random.Random(seed)
    inputs = []
    for _ in range(n):
        gen = rng.choice(_GENERATORS)
        inputs.append(gen(rng))
    return inputs


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
#  Stats collection
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    index: int
    success: bool
    elapsed: float
    retries: int = 0
    prompt_tokens: int | None = None
    eval_tokens: int | None = None
    total_duration_ms: int | None = None
    input_len_chars: int = 0
    error: str | None = None
    error_type: str | None = None
    title: str | None = None
    year: int | None = None
    genres: list[str] = field(default_factory=list)


def _categorize_error(e: Exception) -> str:
    if isinstance(e, json.JSONDecodeError):
        return "json_decode"
    if isinstance(e, ValidationError):
        return "pydantic_validation"
    if isinstance(e, Exception) and "HTTP" in type(e).__name__:
        return "http_error"
    if isinstance(e, OSError):
        return "connection_error"
    return type(e).__name__


def _extract_metrics(body: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    return (
        body.get("prompt_eval_count"),
        body.get("eval_count"),
        body.get("total_duration"),
    )


def _run_backend(
    texts: list[str],
    call_fn: Any,
    backend_name: str,
) -> list[ExtractionResult]:
    """Generic runner that collects per-input stats for any backend callable."""
    results: list[ExtractionResult] = []
    schema = Movie.model_json_schema()

    for idx, text in enumerate(texts):
        result = ExtractionResult(
            index=idx,
            success=False,
            elapsed=0.0,
            retries=0,
            input_len_chars=len(text),
        )

        for attempt in range(MAX_RETRIES):
            start = time.perf_counter()
            try:
                body, content = call_fn(text, schema)
                elapsed = time.perf_counter() - start
                result.elapsed += elapsed

                pt, et, td = _extract_metrics(body)
                result.prompt_tokens = pt
                result.eval_tokens = et
                result.total_duration_ms = td // 1_000_000 if td else None

                content = content.strip()
                if not content:
                    raise ValueError("Empty LLM response")

                movie = Movie.model_validate_json(content)
                result.success = True
                result.title = movie.title
                result.year = movie.year
                result.genres = movie.genres
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    result.retries += 1
                else:
                    result.error = str(e)
                    result.error_type = _categorize_error(e)
            finally:
                pass

        if not result.success:
            logger.warning(f"{backend_name} #[idx] failed — {result.error_type}: {result.error}")
        results.append(result)

    return results


# ---------------------------------------------------------------------------
#  Backend callables
# ---------------------------------------------------------------------------

def _call_urllib(text: str, schema: dict[str, Any]) -> tuple[dict[str, Any], str]:
    payload = {"model": MODEL_NAME, "prompt": _build_prompt(text, schema), "stream": False, "format": schema}
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
    with request.urlopen(req) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body, body.get("response", "")


def _call_requests(text: str, schema: dict[str, Any]) -> tuple[dict[str, Any], str]:
    resp = requests_lib.post(
        OLLAMA_URL,
        json={"model": MODEL_NAME, "prompt": _build_prompt(text, schema), "stream": False, "format": schema},
        headers={"Content-Type": "application/json"},
        timeout=120,
    )
    resp.raise_for_status()
    body = resp.json()
    return body, body.get("response", "")


def _call_ollama_lib(text: str, schema: dict[str, Any]) -> tuple[dict[str, Any], str]:
    resp = ollama.generate(
        model=MODEL_NAME,
        prompt=_build_prompt(text, schema),
        format=schema,
        stream=False,
    )
    return resp, resp.get("response", "")


# ---------------------------------------------------------------------------
#  Stats display
# ---------------------------------------------------------------------------

def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    return values_sorted[f] + (values_sorted[c] - values_sorted[f]) * (k - f)


def print_detailed_stats(results: list[ExtractionResult], label: str) -> None:
    n = len(results)
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]
    times = [r.elapsed for r in successes]
    pt_vals = [r.prompt_tokens for r in successes if r.prompt_tokens is not None]
    et_vals = [r.eval_tokens for r in successes if r.eval_tokens is not None]
    td_vals = [r.total_duration_ms for r in successes if r.total_duration_ms is not None]

    success_rate = len(successes) / n * 100
    first_attempt_rate = (n - sum(r.retries for r in successes)) / n * 100

    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(f"{'=' * 72}")
    print(f"  Inputs:              {n}")
    print(f"  Success:             {len(successes)}/{n} ({success_rate:.1f}%)")
    print(f"  Failures:            {len(failures)}/{n} ({100 - success_rate:.1f}%)")
    print(f"  First-attempt rate:  {first_attempt_rate:.1f}%")
    print()

    if times:
        print(f"  Latency (s):")
        print(f"    min:    {min(times):.3f}")
        print(f"    p50:    {_pct(times, 50):.3f}")
        print(f"    p90:    {_pct(times, 90):.3f}")
        print(f"    p99:    {_pct(times, 99):.3f}")
        print(f"    max:    {max(times):.3f}")
        print(f"    mean:   {statistics.mean(times):.3f}")
        if len(times) > 1:
            print(f"    stdev:  {statistics.stdev(times):.3f}")
        print()

    if pt_vals:
        print(f"  Token counts:")
        print(f"    prompt eval:       {sum(pt_vals)} total, mean {statistics.mean(pt_vals):.0f}")
        if et_vals:
            print(f"    eval:              {sum(et_vals)} total, mean {statistics.mean(et_vals):.0f}")
        if td_vals:
            print(f"    total duration:    {sum(td_vals)/1000:.1f}s total")
        print()

    if failures:
        print(f"  Failure breakdown:")
        error_types: dict[str, int] = {}
        for r in failures:
            et = r.error_type or "unknown"
            error_types[et] = error_types.get(et, 0) + 1
        for et, count in sorted(error_types.items()):
            print(f"    {et:25s} {count}")
        print()

    print(f"  Input char lengths:  min {min(r.input_len_chars for r in results)}, "
          f"max {max(r.input_len_chars for r in results)}, "
          f"mean {statistics.mean(r.input_len_chars for r in results):.0f}")

    if len(successes) > 1:
        throughput = len(successes) / sum(times)
        print(f"  Throughput:          {throughput:.2f} extractions/sec")
    print()


# ---------------------------------------------------------------------------
#  Cross-backend comparison tables
# ---------------------------------------------------------------------------

BACKEND_NAMES = ["urllib", "requests", "ollama lib"]


def print_title_year_comparison(all_results: dict[str, list[ExtractionResult]]) -> None:
    """Per-input table showing title + year from all three backends, color-coded on disagreement."""
    n = len(next(iter(all_results.values())))

    print(f"\n{'=' * 120}")
    print(f"  TITLE & YEAR COMPARISON (color = red when backends disagree)")
    print(f"{'=' * 120}")

    header = (
        f"  {'#':>3}  "
        f"{'urllib':<46}  "
        f"{'requests':<46}  "
        f"{'ollama lib':<46}"
    )
    print(_c(header, C.BOLD))
    print(f"  {'─'*3}  {'─'*46}  {'─'*46}  {'─'*46}")
    for i in range(n):
        cells = []
        for name in BACKEND_NAMES:
            r = all_results[name][i]
            if r.success:
                cell = f"{r.title} ({r.year})"
            else:
                cell = f"FAIL: {r.error_type}"
            cells.append(cell)

        titles = {c.split(" (")[0] for c in cells if "FAIL" not in c}
        years = set()
        for c in cells:
            if "FAIL" not in c and "(" in c:
                try:
                    years.add(int(c.split(" (")[1].rstrip(")")))
                except (ValueError, IndexError):
                    pass

        row_parts = []
        for name, cell in zip(BACKEND_NAMES, cells):
            has_disagreement = len(titles) > 1 or len(years) > 1
            if has_disagreement and "FAIL" not in cell:
                color = C.YELLOW
            elif "FAIL" in cell:
                color = C.RED
            else:
                color = ""
            if len(cell) > 44:
                cell = cell[:41] + "..."
            row_parts.append(_c(f"{cell:<46}", color))

        print(f"  {i:>3}  {'  '.join(row_parts)}")

    print()


def print_timing_comparison(all_results: dict[str, list[ExtractionResult]]) -> None:
    """Side-by-side timing comparison per input, color-coded on divergence."""
    n = len(next(iter(all_results.values())))

    print(f"\n{'=' * 90}")
    print(f"  LATENCY COMPARISON (color = red if >2x difference from fastest)")
    print(f"{'=' * 90}")

    header = f"  {'#':>3}  {'Input len':>9}  {'urllib':>8} {'requests':>9} {'ollama':>7}  {'fastest':>8}"
    print(_c(header, C.BOLD))
    print(f"  {'─'*3}  {'─'*9}  {'─'*8} {'─'*9} {'─'*7}  {'─'*8}")

    for i in range(n):
        times = {name: all_results[name][i].elapsed for name in BACKEND_NAMES}
        min_t = min(times.values())
        max_t = max(times.values())
        input_len = all_results[BACKEND_NAMES[0]][i].input_len_chars

        cells = []
        for name in BACKEND_NAMES:
            t = times[name]
            diff_ratio = t / min_t if min_t > 0 else 1
            # Red if >2x slower than fastest
            if diff_ratio > 2.0:
                color = C.RED
            elif diff_ratio > 1.5:
                color = C.YELLOW
            else:
                color = ""
            cells.append(_c(f"{t:>8.3f}", color))

        fastest_name = min(times, key=times.get)
        print(f"  {i:>3}  {input_len:>9}  {'  '.join(cells)}  {_c(fastest_name, C.CYAN):>8}")
    print()


def print_consistency_matrix(all_results: dict[str, list[ExtractionResult]]) -> None:
    """Agreement rate between each pair of backends on title + year."""
    n = len(next(iter(all_results.values())))

    print(f"\n{'=' * 60}")
    print(f"  CROSS-BACKEND CONSISTENCY MATRIX (title + year agreement)")
    print(f"{'=' * 60}")

    pairs = [
        ("urllib", "requests"),
        ("urllib", "ollama lib"),
        ("requests", "ollama lib"),
    ]

    for a, b in pairs:
        agreements = 0
        compared = 0
        for i in range(n):
            ra, rb = all_results[a][i], all_results[b][i]
            if ra.success and rb.success:
                compared += 1
                if ra.title == rb.title and ra.year == rb.year:
                    agreements += 1

        rate = agreements / compared * 100 if compared else 0
        color = C.GREEN if rate >= 95 else (C.YELLOW if rate >= 80 else C.RED)
        print(f"  {_c(a, C.BOLD):<12} vs {b:<12}  {agreements}/{compared} agree "
              f"({_c(f'{rate:.1f}%', color)})")
    print()


def print_error_summary(all_results: dict[str, list[ExtractionResult]]) -> None:
    """Summarize any failures across all backends."""
    n = len(next(iter(all_results.values())))
    total_errors = 0
    for name in BACKEND_NAMES:
        errors = [r for r in all_results[name] if not r.success]
        total_errors += len(errors)

    if total_errors == 0:
        print(f"\n  {_c(f'No errors across any backend — all {n * len(BACKEND_NAMES)} extractions succeeded.', C.GREEN)}")
        return

    print(f"\n  {_c(f'{total_errors} total errors across all backends:', C.RED)}")
    for name in BACKEND_NAMES:
        errors = [r for r in all_results[name] if not r.success]
        if errors:
            print(f"  {name:<12}:")
            for r in errors:
                print(f"    #{r.index} — {r.error_type}: {r.error}")


# ---------------------------------------------------------------------------
#  Runner
# ---------------------------------------------------------------------------

def show_samples(all_results: dict[str, list[ExtractionResult]], test_inputs: list[str]) -> None:
    n = len(test_inputs)
    if n <= 10:
        tricky = list(range(n))
    else:
        tricky = [12, 17, 18, 19, 27, 0, 5, n - 1]
    tricky = [i for i in tricky if i < n]
    print(f"\n{'=' * 70}")
    print(f"  Sample extractions (model={MODEL_NAME})")
    print(f"{'=' * 70}")
    for idx in tricky:
        text = test_inputs[idx]
        for name in BACKEND_NAMES:
            r = all_results[name][idx]
            if r.success:
                print(f"\n  [{idx:2d}] {name:>12} | Input: {text[:60]}")
                print(f"       Title:   {r.title}")
                print(f"       Year:    {r.year}")
                print(f"       Genres:  {', '.join(r.genres)}")
                print(f"       Tokens:  prompt={r.prompt_tokens}, eval={r.eval_tokens}")
            else:
                print(f"\n  [{idx:2d}] {name:>12} | Input: {text[:60]}")
                print(f"       Error:   {r.error_type}: {r.error}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark urllib vs requests vs ollama for structured LLM extraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  uv run compare_ollama.py                    # 30 hard edge-case inputs (default)
  uv run compare_ollama.py --count 3000      # 3000 programmatically generated inputs
  OLLAMA_MODEL=gemma4:12b uv run compare_ollama.py --count 50
  SHOW_SAMPLES=1 uv run compare_ollama.py    # also show sample extractions
""",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=len(TEST_INPUTS),
        help=f"Number of test inputs (default: {len(TEST_INPUTS)}). "
             f"Values > {len(TEST_INPUTS)} generate inputs programmatically.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generated inputs (default: 42)",
    )
    args = parser.parse_args()

    if args.count <= len(TEST_INPUTS):
        test_inputs = TEST_INPUTS[:args.count]
        source = "hand-crafted edge-cases"
    else:
        test_inputs = generate_inputs(args.count, seed=args.seed)
        source = f"programmatically generated (seed={args.seed})"

    n = len(test_inputs)
    backends = [
        ("urllib", _call_urllib),
        ("requests", _call_requests),
        ("ollama lib", _call_ollama_lib),
    ]

    print(f"Model: {MODEL_NAME}  |  Inputs: {n} ({source})  |  Max retries: {MAX_RETRIES}")
    print(f"Backends: {', '.join(b[0] for b in backends)}")

    all_results: dict[str, list[ExtractionResult]] = {}
    for label, fn in backends:
        print(f"  Running {label}...", flush=True)
        results = _run_backend(test_inputs, fn, label)
        all_results[label] = results

    # --- Summary table ---
    print(f"\n{'=' * 80}")
    print("  SUMMARY")
    print(f"{'=' * 80}")
    print(f"  {'Backend':<16} {'Success':>8} {'Total (s)':>11} {'Avg (s)':>9} {'p50 (s)':>9} {'p90 (s)':>9} {'Failures':>9} {'Throughput':>11}")
    print(f"  {'─'*16} {'─'*8} {'─'*11} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*11}")
    for label, _ in backends:
        results = all_results[label]
        s = sum(1 for r in results if r.success)
        t = sum(r.elapsed for r in results)
        f = n - s
        avg = t / s if s else float("inf")
        times = [r.elapsed for r in results if r.success]
        p50 = _pct(times, 50) if times else 0
        p90 = _pct(times, 90) if times else 0
        throughput = s / t if t > 0 else 0
        print(f"  {label:<16} {s:>8} {t:>11.2f} {avg:>9.3f} {p50:>9.3f} {p90:>9.3f} {f:>9} {throughput:>10.2f}/s")

    total_ok = sum(sum(1 for r in results if r.success) for results in all_results.values())
    print(f"\n  Total successful extractions: {total_ok}/{n * len(backends)}")

    # --- Detailed stats per backend ---
    for label, _ in backends:
        print_detailed_stats(all_results[label], label)

    # --- Cross-backend comparison tables (only for small counts) ---
    if n <= 100:
        print_title_year_comparison(all_results)
        print_timing_comparison(all_results)

    print_consistency_matrix(all_results)
    print_error_summary(all_results)

    if os.environ.get("SHOW_SAMPLES"):
        show_samples(all_results, test_inputs)


if __name__ == "__main__":
    main()
