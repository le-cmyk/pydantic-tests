# pydantic-tests

Benchmark comparison of three HTTP client approaches for structured LLM extraction with Ollama, using Pydantic v2 models for validation.

## Overview

Each backend extracts movie metadata from natural-language text and validates it against a Pydantic `Movie` model:

```python
class Movie(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_default=True)

    title:   str      = Field(..., description="The title of the movie", min_length=1, max_length=200)
    year:    StrictInt = Field(..., description="The release year", ge=1888, le=2100)
    genres:  list[str] = Field(..., description="A list of genres", min_length=1, max_length=5)
    summary: str      = Field(..., description="A short one-sentence summary", min_length=1, max_length=500)
```

The JSON Schema is generated dynamically from the model via `model_json_schema()` and passed to Ollama's `format` parameter, which constrains the LLM to emit valid JSON matching the schema. Parsed output is then validated with `model_validate_json()`.

### Pydantic v2 features used

| Feature | Purpose |
|---|---|
| `ConfigDict(extra="forbid")` | Reject unexpected JSON keys from LLM output |
| `str_strip_whitespace=True` | Auto-strip whitespace on string fields |
| `StrictInt` | Ensure `year` is an integer, not a float string |
| `@field_validator` | Normalize `genres` list (strip + non-empty check) |
| `model_json_schema()` | Dynamically generate the JSON Schema for Ollama's `format` parameter |
| `model_validate_json()` | Single-step JSON parse + validation |
| `Field(..., examples=...)` | Rich metadata for documentation and schema |
| Constraint fields (`ge`, `le`, `min_length`, `max_length`) | Catch invalid values before application logic |

## Backends compared

### 1. urllib (standard library)

```python
from urllib import request

data = json.dumps(payload).encode("utf-8")
req = request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
with request.urlopen(req) as response:
    body = json.loads(response.read().decode("utf-8"))
```

- **Pros**: No external dependencies, always available
- **Cons**: More verbose, no built-in timeout/retry helpers, manual JSON encoding

### 2. requests

```python
import requests

resp = requests.post(OLLAMA_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=120)
body = resp.json()
```

- **Pros**: Clean, intuitive API, rich ecosystem, automatic JSON serialization
- **Cons**: External dependency (~40 kB wheel + urllib3)

### 3. ollama (official library)

```python
import ollama

resp = ollama.generate(model=MODEL_NAME, prompt=prompt, format=schema, stream=False)
```

- **Pros**: Simplest call signature, handles chat vs generate modes, model pulling, streaming
- **Cons**: Less control over raw HTTP requests; must pass schema as a `dict` (not a JSON string) to the `format` parameter

## Running

Requires a local Ollama instance. Two models are supported — `qwen2.5:0.5b` (fast, 397 MB) and `gemma4:12b` (higher quality, 7.6 GB).

```bash
ollama pull qwen2.5:0.5b    # fast model for benchmarking
ollama pull gemma4:12b      # higher-quality model

uv run main.py              # 20 hard edge-case extractions with gemma4:12b
uv run compare_ollama.py    # benchmark all three backends (uses qwen2.5:0.5b by default)

# Use a different model:
OLLAMA_MODEL=gemma4:12b uv run compare_ollama.py

# Show sample extractions on tricky inputs:
SHOW_SAMPLES=1 uv run compare_ollama.py
```

## Benchmark results

Benchmarked with `qwen2.5:0.5b` (fast model) across **30 hard edge-case inputs** including:
- Titles only in director references
- Years as Roman numerals and spelled-out numbers
- Multiple movies mentioned (must disambiguate)
- Non-English text and foreign scripts
- Text-speak / abbreviations
- Single-character inputs and impossible extractions
- Fake/hallucinated movie names
- Haikus and review-style inputs

| Backend | Success | Total time | Avg per call |
|---|---|---|---|
| urllib | 30/30 | 13.92s | 0.46s |
| requests | 30/30 | 13.85s | 0.46s |
| ollama lib | 30/30 | 13.61s | 0.45s |

**Total: 90/90 successful extractions.**

All three backends achieve identical success rates and comparable performance. The dominant cost is LLM inference time, not HTTP library overhead. Differences of ~0.01s between backends are within measurement noise.

### Quality observations (qwen2.5:0.5b on impossible inputs)

The schema guarantees **structural validity** but not **factual accuracy**. The model fills in missing fields with plausible-sounding but fabricated values:

| Input | Model's extraction |
|---|---|
| `"1994."` | Title: Unknown, Year: 1994, Genres: Comedy, Drama |
| `"'"` | Title: The Shawshank Redemption (hallucinated) |
| `"Dark knight rises once..."` (haiku) | Title: Dark Knight Rises, Year: 2012 (correct) |
| `"The Zephyrian Protocol" (fake movie)` | Extracted faithfully — it treated the fictional premise as fact |
| `"Horror."` | Title: Horror, Year: 2018 (fabricated) |

For production use, add a **factuality verification** step (e.g., cross-reference with a movie database API) or switch to a higher-quality model like `gemma4:12b`.

### gemma4:12b results (20 hard cases)

| Backend | Success | Total time | Avg per call |
|---|---|---|---|
| urllib | 20/20 | 24.50s | 4.90s |
| requests | 20/20 | 24.26s | 4.85s |
| ollama lib | 20/20 | 24.76s | 4.95s |

All 20 cases pass. The larger model is ~10× slower per call but produces more accurate titles and genres.

**Recommendation**: Use `ollama` (official library) for the cleanest ergonomics. Fall back to `requests` if you need custom HTTP behavior (proxies, custom auth, interceptors). Use `urllib` only in environments where adding dependencies is infeasible.

## Project structure

```
main.py            — extraction pipeline with 20 hard edge-case test inputs (gemma4:12b)
compare_ollama.py  — benchmark comparing urllib, requests, and ollama lib (qwen2.5:0.5b)
pyproject.toml     — dependencies and project config (uv-managed)
```
