# pydantic-tests

Benchmark comparison of three HTTP client approaches for structured LLM extraction with Ollama, using Pydantic v2 models for validation.

## Overview

Each backend extracts movie metadata from natural-language text and validates it against a Pydantic `Movie` model:

```python
class Movie(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_default=True)

    title: str = Field(..., description="The title of the movie", min_length=1, max_length=200)
    year: StrictInt = Field(..., description="The release year", ge=1888, le=2100)
    genres: list[str] = Field(..., description="A list of genres", min_length=1, max_length=5)
    summary: str = Field(..., description="A short one-sentence summary", min_length=1, max_length=500)
```

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
- **Cons**: Less control over raw HTTP requests, must pass schema as a `dict` (not a JSON string) to the `format` parameter

## Running

Requires a local Ollama instance with `gemma4:12b` pulled:

```bash
ollama pull gemma4:12b

uv run main.py          # extraction on 20 hard edge-case inputs
uv run compare_ollama.py # benchmark all three backends
```

## Benchmark results

| Backend | Success | Total time | Avg per call |
|---|---|---|---|
| urllib | 5/5 | 24.50s | 4.90s |
| requests | 5/5 | 24.26s | 4.85s |
| ollama lib | 5/5 | 24.76s | 4.95s |

All three backends achieve identical success rates and comparable performance (differences are within noise). The dominant cost is LLM inference time, not HTTP library overhead.

**Recommendation**: Use `ollama` (official library) for the cleanest ergonomics. Fall back to `requests` if you need custom HTTP behavior (proxies, custom auth, interceptors). Use `urllib` only in environments where adding dependencies is infeasible.

## Project structure

```
main.py            — extraction pipeline with 20 hard edge-case test inputs
compare_ollama.py  — benchmark comparing urllib, requests, and ollama lib
pyproject.toml     — dependencies and project config (uv-managed)
```
