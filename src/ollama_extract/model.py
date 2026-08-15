"""Pydantic data models for movie extraction."""

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
)

__all__ = ["Movie", "MovieSchema"]


class Movie(BaseModel):
    """A movie extracted from natural-language text.

    Uses Pydantic v2 strict validation to guarantee that LLM output
    matches the expected schema exactly — no extra fields, no type
    coercion, no empty strings.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
    )

    title: str = Field(
        ...,
        description="The title of the movie",
        min_length=1,
        max_length=200,
        examples=["Inception"],
    )
    year: StrictInt = Field(
        ...,
        description="The release year of the movie",
        ge=1888,
        le=2100,
        examples=[2010],
    )
    genres: list[str] = Field(
        ...,
        description="A list of movie genres",
        min_length=1,
        max_length=5,
        examples=[["sci-fi", "heist"]],
    )
    summary: str = Field(
        ...,
        description="A short one-sentence summary of the movie",
        min_length=1,
        max_length=500,
        examples=["A thief who steals secrets through dreams."],
    )

    @field_validator("genres")
    @classmethod
    def _normalize_genres(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("At least one genre is required")
        return [g.strip() for g in v]


def get_movie_schema() -> dict:
    """Return the JSON Schema for the Movie model (for Ollama's ``format`` param)."""
    return Movie.model_json_schema()
