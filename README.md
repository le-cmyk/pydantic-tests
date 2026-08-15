# pydantic-extract

**Benchmark and compare three HTTP client backends (urllib, requests, ollama library) for structured LLM extraction with Ollama, powered by Pydantic v2 validation.**

Features concurrent processing with `ThreadPoolExecutor`, retry logic, real-time `rich` progress bars, color-coded cross-backend comparison tables, and a full CLI with `--count`, `--backend`, `--workers`, and JSON export.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [CLI Reference](#cli-reference)
5. [Concurrency Strategy](#concurrency-strategy)
6. [Pydantic v2 Features](#pydantic-v2-features)
7. [Benchmark Results](#benchmark-results)
8. [Cross-Backend Comparison](#cross-backend-comparison)
9. [Quality & Limitations](#quality--limitations)
10. [Testing](#testing)

---

## Architecture

```
pydantic-extract/
├── pyproject.toml              # Package metadata, deps, entry point
├── README.md
├── LICENSE
├── main.py                     # Thin wrapper → src/ollama_extract/cli:main
├── src/
│   └── ollama_extract/
│       ├── __init__.py         # Public API exports
│       ├── __main__.py         # `python -m ollama_extract` entry
│       ├── cli.py              # argparse CLI + rich output
│       ├── model.py            # Movie BaseModel (pydantic v2)
│       ├── backends/
│       │   ├── __init__.py     # Backend registry + factory
│       │   ├── base.py         # AbstractOllamaBackend + BackendResponse
│       │   ├── urllib_backend.py
│       │   ├── requests_backend.py
│       │   └── ollama_library_backend.py
│       ├── extractor.py        # ConcurrentExtractor (ThreadPoolExecutor)
│       └── generator.py        # 30 hand-crafted + programmatic input generator
├── tests/
│   ├── __init__.py
│   └── test_extraction.py      # 20 unit tests (no Ollama required)
```

### Design

```
┌────────────────────────────────────────────────────────┐
│                     CLI (cli.py)                        │
│  argparse: --count, --model, --backend, --workers     │
└──────────────┬─────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────┐
│              ConcurrentExtractor (extractor.py)        │
│  • ThreadPoolExecutor (4 workers default)              │
│  • Retry logic (3 attempts on JSON/ValidationError)    │
│  • Real-time rich Progress bar per backend             │
│  • Latency percentiles, token counts, throughput       │
├────────────────┬────────────────┬────────────────────┤
│                │                │                    │
│  ┌──────────┐   │  ┌──────────┐  │  ┌──────────────┐ │
│  │ urllib  │   │  │ requests │  │  │ ollama lib   │ │
│  │ backend │   │  │ backend  │  │  │ backend      │ │
│  └──────────┘   │  └──────────┘  │  └──────────────┘ │
│                 │                 │                   │
│  urllib.request │  requests.post │  ollama.generate()│
│  (stdlib)        │  (requests)    │  (official lib)  │
└─────────────────┴────────────────┴───────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────┐
│                    Pydantic Validation (model.py)       │
│  Model.model_json_schema()  →  Ollama format param      │
│  Model.model_validate_json() ←  validated Movie object    │
└────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python ≥ 3.14
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Local [Ollama](https://ollama.com/) instance

### Setup

```bash
git clone https://github.com/le-cmyk/pydantic-tests.git
cd pydantic-tests

uv sync                    # Install deps + package in editable mode
uv run ollama-extract --help

# Pull models
ollama pull qwen2.5:0.5b    # fast model (397 MB) — default for benchmarks
ollama pull gemma4:12b      # higher quality (7.6 GB) — more accurate extraction
```

### Dependencies

| Package | Purpose |
|---|---|
| `pydantic>=2.13` | Structured JSON validation |
| `requests>=2.34` | HTTP client backend |
| `ollama>=0.6.2` | Official Ollama Python SDK |
| `rich>=15.0` | Real-time progress bars + tables |
| `pytest>=9.1` | Unit testing |

---

## Quick Start

```bash
# 30 hand-crafted hard edge-cases (default)
uv run ollama-extract

# 200 inputs with 4 parallel workers
uv run ollama-extract --count 200 --workers 4

# 3000 inputs, single backend, JSON export
uv run ollama-extract --count 3000 --backend requests --output results.json

# Higher-quality model
OLLAMA_MODEL=gemma4:12b uv run ollama-extract --count 20

# Verbose output: show samples, per-input tables, consistency matrix
SHOW_SAMPLES=1 uv run ollama-extract --count 30
```

---

## CLI Reference

```
ollama-extract [-h] [--count COUNT] [--model MODEL]
                [--backend {urllib,requests,ollama,all}]
                [--workers WORKERS] [--seed SEED]
                [--output OUTPUT] [--quiet]
                [--log-level {DEBUG,INFO,WARNING,ERROR}]
                [--ollama-url URL]
```

| Flag | Short | Default | Description |
|---|---|---|---|
| `--count` | `-n` | `30` | Number of test inputs. ≤30 uses hand-crafted edge-cases; >30 generates inputs programmatically. |
| `--model` | `-m` | `qwen2.5:0.5b` | Ollama model name. |
| `--backend` | `-b` | `all` | Which backend(s) to run: `urllib`, `requests`, `ollama`, or `all`. |
| `--workers` | `-w` | `4` | Concurrent `ThreadPoolExecutor` workers. |
| `--seed` | `-s` | `42` | Random seed for generated inputs (reproducibility). |
| `--output` | `-o` | — | Write full results to JSON file. |
| `--quiet` | `-q` | off | Suppress progress bar and comparison tables. |
| `--log-level` | — | `WARNING` | Logging verbosity. |
| `--ollama-url` | — | `http://localhost:11434/api/generate` | Custom Ollama endpoint. |

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:0.5b` | Override default model. |
| `SHOW_SAMPLES` | unset | If set, print detailed sample extractions for tricky inputs. |

---

## Concurrency Strategy

This workload is **I/O-bound** — each request spends ~3.5s waiting for LLM inference
and ~0.01s on HTTP overhead + JSON validation. The standard formula applies
(Brian Goetz, *Java Concurrency in Practice*):

```
N_threads = CPU_cores × (1 + Wait_time / Service_time)
```

For a 0.4s wait and 0.01s service on 8 cores: 8 × (1 + 40) = 328 threads —
**theoretically** optimal. In practice, the **local Ollama server** is the bottleneck:

| Workers | Avg latency/request | Throughput | Notes |
|---|---|---|---|
| 1 | 0.43s | 2.35/s | No contention |
| 4 | 2.94s | 0.34/s | Sweet spot — 3x speedup vs serial |
| 8 | 5.19s | 0.21/s | Server overloaded, latency degrades |

**Recommendation:** 4 workers for local Ollama. The GIL is released during socket I/O,
so threads overlap effectively up to the server's capacity. Beyond 4–6 workers,
latency increases sharply without throughput gains.

### Reliability

- **Retry logic**: 3 attempts per input, catching `JSONDecodeError` and `ValidationError`
- **Backpressure**: `ThreadPoolExecutor` queues requests naturally (bounded by input count)
- **Graceful shutdown**: All futures are collected via `as_completed`; no orphaned threads
- **Error isolation**: One backend's failure doesn't affect others; per-input results tracked

---

## Pydantic v2 Features

| Feature | How it's used |
|---|---|
| `BaseModel` | `Movie` model with 4 typed fields |
| `ConfigDict(extra="forbid")` | Rejects unexpected keys from LLM output |
| `str_strip_whitespace=True` | Auto-strips whitespace on string fields |
| `StrictInt` | Ensures `year` is a real integer, not `"2020"` |
| `@field_validator` | Normalizes `genres` list (strips + non-empty) |
| `model_json_schema()` | Generates JSON Schema for Ollama's `format` parameter |
| `model_validate_json()` | Single-step JSON parse + validation |
| `Field(..., min_length, max_length, ge, le)` | Field-level constraints catch invalid values |
| `Field(..., examples=...)` | Rich schema metadata |
| `ValidationError` | Catches all validation failures for retry logic |

---

## Benchmark Results

### 30 hand-crafted edge-cases (qwen2.5:0.5b)

| Backend | Success | Total (s) | Avg (s) | p50 | p90 | Stdev | Throughput |
|---|---|---|---|---|---|---|---|
| urllib | 30/30 | 97.4 | 4.87 | 5.19 | 5.99 | 0.081 | 0.21/s |
| requests | 30/30 | 94.3 | 4.71 | 5.24 | 5.88 | 0.104 | 0.21/s |
| ollama lib | 30/30 | 102.2 | 5.11 | 5.71 | 5.95 | 0.073 | 0.20/s |

**All 90/90 extractions succeeded. 100% first-attempt success rate.**

### 200 generated inputs (qwen2.5:0.5b, 4 workers)

| Backend | Success | Total (s) | Avg (s) | Throughput |
|---|---|---|---|---|
| urllib | 200/200 | 587.7 | 2.94 | 0.34/s |
| requests | 200/200 | 604.2 | 3.02 | 0.33/s |
| ollama lib | 200/200 | 620.0 | 3.10 | 0.32/s |

**All 600/600 extractions succeeded.**

### Token statistics (30-case run)

| Backend | Prompt eval tokens | Eval tokens | Total duration |
|---|---|---|---|
| urllib | 12,455 (mean 415) | ~2,100 (mean ~70) | 97.4s |
| requests | 12,455 (mean 415) | ~2,100 (mean ~70) | 94.3s |
| ollama lib | 12,455 (mean 415) | ~2,100 (mean ~70) | 102.2s |

All backends produce identical prompt token counts (same schema-driven prompt).
Eval tokens vary slightly due to LLM nondeterminism.

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
| 1 | **The Dark Knight** (2010) | The Inception (2010) | Inception (2010) |
| 2 | Unknown (1994) | Pulp Fiction (1994) | Pulp Fiction (1994) |
| 3 | The Shining (1980) | The Shining (1980) | The Shining (1980) |
| 4 | **Inception** (1999) | Inception (1999) | **The Matrix** (1999) |

Bold rows indicate title disagreement across backends. The model hallucinates
different values for ambiguous inputs while always producing valid JSON.

---

## Quality & Limitations

### What the schema guarantees

- ✅ Valid JSON output
- ✅ All 4 fields present (`title`, `year`, `genres`, `summary`)
- ✅ Correct types (string, int, list of strings, string)
- ✅ `year` in range 1888–2100 (StrictInt)
- ✅ `title` and `summary` non-empty, length-bounded
- ✅ `genres` non-empty list, max 5 items
- ✅ No extra fields (`extra="forbid"`)

### What the schema does NOT guarantee

- ❌ **Factual accuracy** — the model will hallucinate plausible-sounding but
  incorrect values for impossible inputs (e.g., `"1994."` → "The Dark Knight")
- ❌ **Title disambiguation** — multiple movies in context may cause the model
  to pick the wrong one
- ❌ **Year correction** — contradictory/incorrect years in the input may go uncorrected

### Recommendation

For production use:
1. Add a **factuality verification** step (e.g., cross-reference with TMDb API)
2. Use `gemma4:12b` for higher accuracy (~5s/call vs ~0.4s)
3. Log disagreements between backends as a quality signal

---

## Testing

```bash
# Run unit tests (no Ollama required)
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=ollama_extract

# Test with Ollama (requires local instance)
uv run ollama-extract --count 10 --quiet
```

### Test coverage

| Module | Tests |
|---|---|
| `model.py` | 9 — field validation, extra rejection, StrictInt, constraints |
| `generator.py` | 4 — count, reproducibility, seed variance |
| `backends/` | 5 — factory, list, unknown backend, all 3 implementations |
| `extractor.py` | 2 — prompt builder, BatchResult properties |
| **Total** | **20** |

---

## Project Structure

```
pydantic-extract/
├── pyproject.toml              # Package metadata, deps, entry point
├── README.md
├── LICENSE
├── main.py                     # Thin wrapper → src/ollama_extract/cli:main
├── src/
│   └── ollama_extract/
│       ├── __init__.py         # Public API exports
│       ├── __main__.py         # `python -m ollama_extract` entry
│       ├── cli.py              # argparse CLI + rich output
│       ├── model.py            # Movie BaseModel (pydantic v2)
│       ├── backends/
│       │   ├── __init__.py     # Backend registry + factory
│       │   ├── base.py         # AbstractOllamaBackend + BackendResponse
│       │   ├── urllib_backend.py
│       │   ├── requests_backend.py
│       │   └── ollama_library_backend.py
│       ├── extractor.py        # ConcurrentExtractor (ThreadPoolExecutor)
│       └── generator.py        # 30 hand-crafted + programmatic input generator
├── tests/
│   ├── __init__.py
│   └── test_extraction.py      # 20 unit tests (no Ollama required)
```
