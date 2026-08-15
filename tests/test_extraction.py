"""Unit tests for the pydantic extraction pipeline (no Ollama required)."""

import pytest
from pydantic import ValidationError

from ollama_extract.model import Movie, get_movie_schema
from ollama_extract.generator import generate_inputs, TEST_INPUTS
from ollama_extract.extractor import ConcurrentExtractor


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestMovieModel:
    def test_valid_extraction(self):
        m = Movie(title="Inception", year=2010, genres=["sci-fi", "heist"],
                  summary="A thief who steals secrets through dreams.")
        assert m.title == "Inception"
        assert m.year == 2010
        assert m.genres == ["sci-fi", "heist"]

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            Movie.model_validate_json(
                '{"title": "X", "year": 2020, "genres": ["a"], "summary": "s", "extra": "bad"}'
            )

    def test_string_year_coerced_to_int(self):
        """StrictInt rejects float strings — must be an actual integer."""
        with pytest.raises(ValidationError):
            Movie.model_validate_json(
                '{"title": "X", "year": "2020", "genres": ["a"], "summary": "s"}'
            )

    def test_year_out_of_range(self):
        with pytest.raises(ValidationError):
            Movie.model_validate_json(
                '{"title": "X", "year": 1800, "genres": ["a"], "summary": "s"}'
            )
        with pytest.raises(ValidationError):
            Movie.model_validate_json(
                '{"title": "X", "year": 2200, "genres": ["a"], "summary": "s"}'
            )

    def test_empty_title_rejected(self):
        with pytest.raises(ValidationError):
            Movie.model_validate_json(
                '{"title": "", "year": 2020, "genres": ["a"], "summary": "s"}'
            )

    def test_whitespace_title_stripped(self):
        m = Movie(title="  Inception  ", year=2010, genres=["sci-fi"], summary="x")
        assert m.title == "Inception"

    def test_genre_validator_strips(self):
        m = Movie(title="X", year=2010, genres=[" sci-fi ", " heist "], summary="x")
        assert m.genres == ["sci-fi", "heist"]

    def test_empty_genre_list_rejected(self):
        with pytest.raises(ValidationError):
            Movie.model_validate_json(
                '{"title": "X", "year": 2020, "genres": [], "summary": "s"}'
            )

    def test_model_schema_has_required_fields(self):
        schema = get_movie_schema()
        assert set(schema["required"]) == {"title", "year", "genres", "summary"}
        assert "properties" in schema


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------

class TestGenerator:
    def test_test_inputs_has_30(self):
        assert len(TEST_INPUTS) == 30

    def test_generated_count(self):
        inputs = generate_inputs(100, seed=42)
        assert len(inputs) == 100
        assert all(isinstance(x, str) and len(x) > 0 for x in inputs)

    def test_generate_inputs_reproducible(self):
        a = generate_inputs(50, seed=123)
        b = generate_inputs(50, seed=123)
        assert a == b

    def test_generate_inputs_different_seeds(self):
        a = generate_inputs(50, seed=1)
        b = generate_inputs(50, seed=2)
        assert a != b


# ---------------------------------------------------------------------------
# Backend abstraction tests
# ---------------------------------------------------------------------------

class TestBackends:
    def test_get_backend_urllib(self):
        from ollama_extract.backends import get_backend
        b = get_backend("urllib", "test:model", "http://localhost:11434/api/generate")
        assert b.name == "urllib"

    def test_get_backend_requests(self):
        from ollama_extract.backends import get_backend
        b = get_backend("requests", "test:model", "http://localhost:11434/api/generate")
        assert b.name == "requests"

    def test_get_backend_ollama(self):
        from ollama_extract.backends import get_backend
        b = get_backend("ollama", "test:model", "http://localhost:11434/api/generate")
        assert b.name == "ollama"

    def test_get_backend_unknown(self):
        from ollama_extract.backends import get_backend
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("tensorflow", "test:model", "http://localhost:11434/api/generate")

    def test_list_backends(self):
        from ollama_extract.backends import list_backends
        backends = list_backends()
        assert "urllib" in backends
        assert "requests" in backends
        assert "ollama" in backends


# ---------------------------------------------------------------------------
# Extractor tests (mocked backend)
# ---------------------------------------------------------------------------

class TestExtractor:
    def test_prompt_builder_includes_schema(self):
        from ollama_extract.extractor import _build_prompt
        schema = get_movie_schema()
        prompt = _build_prompt("Test movie input", schema)
        assert "Extract the movie information" in prompt
        assert "JSON Schema" in prompt
        assert "Test movie input" in prompt

    def test_batch_result_properties(self):
        from ollama_extract.extractor import BatchResult, ExtractionResult
        r1 = ExtractionResult(index=0, input_text="x", success=True, elapsed=1.0, title="A", year=2010, genres=["g"])
        r2 = ExtractionResult(index=1, input_text="y", success=True, elapsed=2.0, title="B", year=2011, genres=["g"])
        r3 = ExtractionResult(index=2, input_text="z", success=False, elapsed=0.5, error="fail", error_type="ValidationError")
        batch = BatchResult(backend="test", results=[r1, r2, r3])

        assert batch.success_count == 2
        assert batch.failure_count == 1
        assert batch.total_elapsed == 3.5
        assert batch.avg_elapsed == 1.5  # (1.0 + 2.0) / 2
        assert len(batch.latencies) == 2
