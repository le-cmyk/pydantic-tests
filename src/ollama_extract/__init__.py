"""Ollama structured extraction with Pydantic v2 validation.

A benchmarking toolkit that compares three HTTP client backends (urllib, requests,
and the official ollama library) for structured LLM extraction.  Features
concurrent processing, retry logic, real-time progress, and color-coded
cross-backend comparison tables.
"""

from ollama_extract.model import Movie, get_movie_schema
from ollama_extract.backends import BackendName, get_backend, list_backends
from ollama_extract.extractor import ConcurrentExtractor, ExtractionResult, BatchResult
from ollama_extract.generator import TEST_INPUTS, generate_inputs

__version__ = "1.0.0"

__all__ = [
    "Movie",
    "get_movie_schema",
    "BackendName",
    "get_backend",
    "list_backends",
    "ConcurrentExtractor",
    "ExtractionResult",
    "BatchResult",
    "TEST_INPUTS",
    "generate_inputs",
]
