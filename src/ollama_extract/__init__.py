"""Ollama structured extraction with Pydantic v2 validation.

A benchmarking toolkit that compares three HTTP client backends (urllib, requests,
and the official ollama library) for structured LLM extraction.  Features
concurrent processing with ThreadPoolExecutor, retry logic, real-time rich
progress bars, color-coded cross-backend comparison tables, and a full CLI.

Usage:
    uv run ollama-extract --count 30
    uv run ollama-extract --count 3000 --workers 4 --output results.json
    python -m ollama_extract --help
"""

from ollama_extract.model import Movie, get_movie_schema
from ollama_extract.backends import BackendName, get_backend, list_backends, OllamaBackend
from ollama_extract.extractor import ConcurrentExtractor, ExtractionResult, BatchResult
from ollama_extract.generator import TEST_INPUTS, generate_inputs

__version__ = "1.0.0"

__all__ = [
    "Movie",
    "get_movie_schema",
    "BackendName",
    "get_backend",
    "list_backends",
    "OllamaBackend",
    "ConcurrentExtractor",
    "ExtractionResult",
    "BatchResult",
    "TEST_INPUTS",
    "generate_inputs",
]
