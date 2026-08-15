"""Benchmark comparison: urllib vs requests vs ollama for calling Ollama API.

Each backend performs the same extraction task (Movie model) and supports
JSON-schema structured output.  Results are printed in comparison tables
with color-coded differences.
"""

import json
import os
import time
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
#  Test inputs — 30 hard edge-cases
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
        disagreements = []
        for name in BACKEND_NAMES:
            r = all_results[name][i]
            if r.success:
                cell = f"{r.title} ({r.year})"
            else:
                cell = f"FAIL: {r.error_type}"
            cells.append(cell)

        # Check disagreement across successful results
        titles = {c.split(" (")[0] for c in cells if "FAIL" not in c and cell is not None}
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
    total_errors = 0
    for name in BACKEND_NAMES:
        errors = [r for r in all_results[name] if not r.success]
        total_errors += len(errors)

    if total_errors == 0:
        print(f"\n  {_c('No errors across any backend — all 90 extractions succeeded.', C.GREEN)}")
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

def main() -> None:
    n = len(TEST_INPUTS)
    backends = [
        ("urllib", _call_urllib),
        ("requests", _call_requests),
        ("ollama lib", _call_ollama_lib),
    ]

    print(f"Model: {MODEL_NAME}  |  Inputs per backend: {n}  |  Max retries: {MAX_RETRIES}")

    all_results: dict[str, list[ExtractionResult]] = {}
    for label, fn in backends:
        results = _run_backend(TEST_INPUTS, fn, label)
        all_results[label] = results

    # --- Summary table ---
    print(f"\n{'=' * 72}")
    print("  SUMMARY")
    print(f"{'=' * 72}")
    print(f"  {'Backend':<16} {'Success':>8} {'Total (s)':>11} {'Avg (s)':>9} {'Failures':>9}")
    print(f"  {'─'*16} {'─'*8} {'─'*11} {'─'*9} {'─'*9}")
    for label, _ in backends:
        results = all_results[label]
        s = sum(1 for r in results if r.success)
        t = sum(r.elapsed for r in results)
        f = n - s
        avg = t / s if s else float("inf")
        print(f"  {label:<16} {s:>8} {t:>11.2f} {avg:>9.3f} {f:>9}")

    total_ok = sum(sum(1 for r in results if r.success) for results in all_results.values())
    print(f"\n  Total successful extractions: {total_ok}/{n * len(backends)}")

    # --- Detailed stats per backend ---
    for label, _ in backends:
        print_detailed_stats(all_results[label], label)

    # --- Cross-backend comparison tables ---
    print_title_year_comparison(all_results)
    print_timing_comparison(all_results)
    print_consistency_matrix(all_results)
    print_error_summary(all_results)

    if os.environ.get("SHOW_SAMPLES"):
        show_samples(all_results)


def show_samples(all_results: dict[str, list[ExtractionResult]]) -> None:
    schema = Movie.model_json_schema()
    tricky = [12, 17, 18, 19, 27]
    print(f"\n{'=' * 70}")
    print(f"  Sample extractions (model={MODEL_NAME})")
    print(f"{'=' * 70}")
    for idx in tricky:
        text = TEST_INPUTS[idx]
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


if __name__ == "__main__":
    main()
