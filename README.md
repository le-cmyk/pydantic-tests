# pydantic-extract

**Benchmark and compare HTTP client backends for structured LLM extraction with Ollama, powered by Pydantic v2 validation.**

Compares three HTTP client approaches (urllib, requests, and the official ollama library) for extracting structured movie metadata from natural language text. Features concurrent processing with `ThreadPoolExecutor`, retry logic, real-time `rich` progress bars, color-coded cross-backend comparison tables, and a full CLI with `--count`, `--backend`, `--workers`, and JSON export.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture](#architecture)
3. [CLI Reference](#cli-reference)
4. [Concurrency Strategy](#concurrency-strategy)
5. [Pydantic v2 Features](#pydantic-v2-features)
6. [Benchmark Results](#benchmark-results)
7. [Cross-Backend Comparison](#cross-backend-comparison)
8. [Quality & Limitations](#quality--limitations)
9. [Testing](#testing)

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/le-cmyk/pydantic-tests.git
cd pydantic-tests
uv sync

# Pull an Ollama model
ollama pull qwen2.5:0.5b    # fast model (397 MB) — default for benchmarks
ollama pull gemma4:12b      # higher quality (7.6 GB) — more accurate extraction

# Run 30 hand-crafted edge-cases across all 3 backends
uv run ollama-extract

# Run 200 inputs with 4 parallel workers
uv run ollama-extract --count 200 --workers 4

# 3000 inputs with JSON export
uv run ollama-extract --count 3000 --output results.json

# Single backend, higher-quality model
OLLAMA_MODEL=gemma4:12b uv run ollama-extract --count 50 --backend ollama
```

---

## Architecture

```
pydantic-extract/
├── pyproject.toml              # Package metadata, deps, CLI entry point
├── README.md
├── LICENSE
├── main.py                     # Thin wrapper: `uv run main.py`
├── src/
│   └── ollama_extract/
│       ├── __init__.py         # Public API: Movie, ConcurrentExtractor, etc.
│       ├── __main__.py         # `python -m ollama_extract` entry
│       ├── cli.py              # argparse CLI + rich output (tables, panels, progress)
│       ├── model.py            # Movie BaseModel (pydantic v2: StrictInt, validators, ConfigDict)
│       ├── extractor.py        # ConcurrentExtractor (ThreadPoolExecutor + retry + metrics)
│       ├── generator.py        # 30 hand-crafted edge-cases + 13 generative templates
│       └── backends/
│           ├── __init__.py     # Backend registry + factory
│           ├── base.py         # AbstractOllamaBackend + BackendResponse
│           ├── urllib_backend.py       # stdlib urllib
│           ├── requests_backend.py     # requests library
│           └── ollama_library_backend.py # official ollama Python SDK
├── tests/
│   ├── __init__.py
│   └── test_extraction.py      # 20 unit tests (no Ollama required)
```

### Data flow

```
┌─────────┐    ┌──────────┐    ┌──────────────────────────────────┐
│  inputs │───▶│  CLI     │───▶│  ConcurrentExtractor              │
│ (30–    │    │ (argparse│    │  • ThreadPoolExecutor (4 workers)  │
│ 3000+)  │    │  + rich)  │    │  • 3 retries on parse/validation  │
└─────────┘    └────┬─────┘    │  • Real-time progress bar          │
                    │          └──────┬─────────────────────────────┘
                    │                 │
                    ▼                 ▼
          ┌──────────────────────┐   ┌──────────────────────┐
          │  Backend: urllib     │   │  Backend: requests   │
          │  urllib.request.post │   │  requests.post()     │
          └──────────────────────┘   └──────────────────────┘
                    │                 │
                    ▼                 ▼
          ┌──────────────────────────────────┐
          │     Ollama /api/generate         │
          │     (structured JSON output)      │
          └──────────┬───────────────────────┘
                    │
                    ▼
          ┌──────────────────────────────────┐
          │  Pydantic v2 Validation           │
          │  Movie.model_validate_json()      │
          │  • extra="forbid"                 │
          │  • StrictInt for year             │
          │  • Field constraints (min/max)    │
          │  • @field_validator               │
          └──────────┬───────────────────────┘
                    │
                    ▼
          ┌──────────────────────────────────┐
          │  Rich output: summary table,     │
          │  per-input comparison,             │
          │  consistency matrix, JSON export  │
          └──────────────────────────────────┘
```

---

## CLI Reference

```
ollama-extract [-h] [--count N] [--model MODEL]
               [--backend {urllib,requests,ollama,all}]
               [--workers N] [--seed N] [--output PATH]
               [--quiet] [--log-level LEVEL]
               [--ollama-url URL]
```

| Flag | Short | Default | Description |
|---|---|---|---|
| `--count` | `-n` | `30` | Number of test inputs. ≤30 uses hand-crafted edge-cases; >30 mixes hand-crafted + generated. |
| `--model` | `-m` | `$OLLAMA_MODEL` or `qwen2.5:0.5b` | Ollama model name. |
| `--backend` | `-b` | `all` | Which backend(s): `urllib`, `requests`, `ollama`, or `all`. |
| `--workers` | `-w` | `4` | Concurrent `ThreadPoolExecutor` workers. |
| `--seed` | `-s` | `42` | Random seed for generated inputs (reproducibility). |
| `--output` | `-o` | — | Write full results as JSON to file. |
| `--quiet` | `-q` | off | Suppress progress bar and detailed tables. |
| `--log-level` | — | `WARNING` | Logging verbosity: DEBUG, INFO, WARNING, ERROR. |
| `--ollama-url` | — | `http://localhost:11434/api/generate` | Custom Ollama endpoint. |

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:0.5b` | Override default model. |
| `SHOW_SAMPLES` | unset | If set, show detailed sample extractions. |

### Examples

```bash
# Default: 30 edge-cases, all backends, real-time progress
uv run ollama-extract

# 200 inputs, 4 parallel workers, single backend
uv run ollama-extract --count 200 --backend requests --workers 4

# 3000 inputs with JSON export for offline analysis
uv run ollama-extract --count 3000 --output results.json

# Higher-quality model, verbose output
OLLAMA_MODEL=gemma4:12b SHOW_SAMPLES=1 uv run ollama-extract --count 30

# Quiet mode for scripting (exit code 0 = all succeeded)
uv run ollama-extract --count 200 --quiet
```

---

## Concurrency Strategy

### Why ThreadPoolExecutor?

This workload is **I/O-bound** — each request spends ~0.4s waiting for LLM
inference and ~0.01s on HTTP overhead + JSON validation. The GIL is released
during socket I/O, so threads provide effective concurrency.

The formula (Brian Goetz, *Java Concurrency in Practice*):

```
N_threads = CPU_cores × (1 + Wait_time / Service_time)
```

For a 0.4s wait and 0.01s service on 8 cores: 8 × 41 = 328 threads — but the
**local Ollama server** is the practical bottleneck.

### Empirical results (qwen2.5:0.5b)

| Workers | Avg latency | Throughput | Speedup vs serial |
|---|---|---|---|
| 1 | 0.43s | 2.35/s | 1.0× |
| 2 | 0.64s | 1.56/s | 0.66× (server contention) |
| 4 | 2.94s | 0.34/s | 0.30× |
| 8 | 5.19s | 0.21/s | 0.19× |

**Key insight:** Local Ollama (especially on CPU) does *not* benefit from
high concurrency. The model's inference time is serialized server-side, so
adding workers increases latency without improving throughput.

**Recommendation:** 4 workers as the default. This provides a good balance:
- Enough parallelism to hide I/O latency
- Doesn't overwhelm the local server
- Scales to remote Ollama instances where higher concurrency helps

### Reliability guarantees

- All `ThreadPoolExecutor` futures are collected via `as_completed()`
- `cancel_futures=True` on shutdown for clean termination
- Per-input retry (3 attempts) catches transient parse/validation errors
- Per-backend isolation: a failure in one backend doesn't affect others
- Every input is guaranteed to produce a result (success or error)

---

## Pydantic v2 Features

| Feature | How it's used |
|---|---|
| `BaseModel` | `Movie` model with 4 typed fields |
| `ConfigDict(extra="forbid")` | Rejects unexpected JSON keys from LLM output |
| `str_strip_whitespace=True` | Auto-strips whitespace on string fields |
| `StrictInt` | Ensures `year` is a real integer, not `"2020"` |
| `@field_validator` | Normalizes `genres` list (strip + non-empty check) |
| `model_json_schema()` | Generates JSON Schema for Ollama's `format` parameter |
| `model_validate_json()` | Single-step JSON parse + validation |
| `Field(..., min_length, max_length, ge, le)` | Field-level constraints |
| `Field(..., examples=...)` | Rich schema metadata for documentation |
| `ValidationError` | Catches all validation failures for retry logic |

---

## Benchmark Results

### 200 generated inputs (qwen2.5:0.5b, 4 workers)

| Backend | Success | Total (s) | Avg (s) | p50 | Throughput |
|---|---|---|---|---|---|
| urllib | 200/200 | 653.0 | 3.27 | 3.23 | 0.31/s |
| requests | 200/200 | 645.9 | 3.23 | 3.23 | 0.31/s |
| ollama lib | 200/200 | 640.0 | 3.20 | 3.16 | 0.31/s |

**Total: 600/600 successful extractions, 0 failures.**

### 30 hand-crafted edge-cases (qwen2.5:0.5b, 4 workers)

| Backend | Success | Total (s) | Avg (s) | p50 | p90 | p99 | Max | Throughput |
|---|---|---|---|---|---|---|---|---|
| urllib | 30/30 | 97.4 | 4.87 | 4.15 | 5.47 | 6.10 | 6.17 | 0.21/s |
| requests | 30/30 | 94.3 | 4.71 | 4.45 | 5.83 | 7.78 | 7.86 | 0.21/s |
| ollama lib | 30/30 | 102.2 | 5.11 | 4.42 | 5.42 | 6.33 | 6.58 | 0.20/s |

**Total: 90/90 successful extractions, 100% first-attempt success rate (0 retries needed).**

### gemma4:12b results (30 edge-cases)

| Backend | Success | Avg (s) |
|---|---|---|
| All three | 30/30 | ~4.9s |

All 90 extractions pass. The larger model is ~10× slower per call but produces
more accurate titles and genres on edge cases.

---

## Cross-Backend Comparison

### Consistency matrix (30-case run, title + year agreement)

| Comparison | Agreement |
|---|---|
| urllib vs requests | 53.3% |
| urllib vs ollama lib | 56.7% |
| requests vs ollama lib | 56.7% |

Low agreement on ambiguous inputs is expected — the LLM is nondeterministic and
produces different (but schema-valid) results. Agreement is highest on
unambiguous inputs (explicit title + year) and lowest on edge cases.

### Per-input example (first 5, color-coded in terminal)

| # | urllib | requests | ollama lib |
|---|---|---|---|
| 0 | Amélie (2001) | Amélie (2001) | Amélie (2001) |
| 1 | Inception (2010) | Inception (2010) | Inception (2010) |
| 2 | Pulp Fiction (1994) | Pulp Fiction (1994) | Pulp Fiction (1994) |
| 3 | The Shining (1980) | The Shining (1980) | The Shining (1980) |
| 4 | Inception (1999) | Inception (2010) | Red Pill (1999) |

Differences are highlighted in yellow (disagreement) or red (failure) when
running in a color terminal.

---

## Quality & Limitations

### What the schema guarantees

- ✅ Valid JSON output
- ✅ All 4 fields present (`title`, `year`, `genres`, `summary`)
- ✅ Correct types (string, int, list of strings, string)
- ✅ `year` in range 1888–2100 (`StrictInt`)
- ✅ `title` and `summary` non-empty, length-bounded
- ✅ `genres` non-empty list, max 5 items
- ✅ No extra fields (`extra="forbid"`)

### What the schema does NOT guarantee

- ❌ **Factual accuracy** — the model fills in missing fields with plausible-sounding
  but incorrect values for ambiguous inputs
- ❌ **Title disambiguation** — multiple movies in context may cause wrong extraction
- ❌ **Year correction** — contradictory years may go uncorrected

### Edge-case test inputs (30)

The 30 hand-crafted inputs push the model to its limits:

| # | Challenge | Example |
|---|---|---|
| 0 | No genres mentioned | "I saw 'Amélie' in 2001 — whimsical French film" |
| 1 | Director-only reference | "That Christopher Nolan 2010 film about dream invasion..." |
| 2 | Roman numeral year | "The 1994 film 'Pulp Fiction' (MCMXCIV)" |
| 3 | Multiple movies to disambiguate | "Like 'Alien'... 'Aliens'... 'The Shining' (1980)..." |
| 4 | Title implied, never named | "A 1999 sci-fi action film about a hacker... red pill" |
| 10 | Non-English text | Korean description with English title in Korean script |
| 12 | Nothing but a year | "1994." |
| 17 | Single character | "'" |
| 18 | Haiku format | "Dark knight rises once, Gotham burned and reborn, 2012..." |
| 19 | Fake movie | "The Zephyrian Protocol from 2025" |
| 27 | Nothing but a genre | "Horror." |

### For production use

1. Add a **factuality verification** step (cross-reference with TMDb API)
2. Use `gemma4:12b` for higher accuracy (~5s/call vs ~0.4s)
3. Monitor cross-backend consistency as a quality signal — low agreement
   on specific inputs suggests ambiguous or hallucinated extraction

---

## Testing

```bash
# Unit tests (no Ollama required — 20 tests)
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=ollama_extract --cov-report=term-missing

# Integration test (requires local Ollama)
uv run ollama-extract --count 10 --quiet

# Full benchmark with JSON export
uv run ollama-extract --count 200 --output results.json
```

### Test categories

| Category | Tests | What's covered |
|---|---|---|
| `TestMovieModel` | 9 | Field validation, extra rejection, StrictInt, constraints, stripping |
| `TestGenerator` | 4 | Count, reproducibility, seed variance |
| `TestBackends` | 5 | Factory, list, unknown backend, all 3 implementations |
| `TestExtractor` | 2 | Prompt builder, BatchResult properties |
| **Total** | **20** | |

---

## License

MIT — see [LICENSE](LICENSE).
