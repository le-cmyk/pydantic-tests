import json
import logging
from urllib import request
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationError,
    field_validator,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma4:12b"
MAX_RETRIES = 3


class Movie(BaseModel):
    """A movie extracted from natural-language text."""

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


def build_prompt(raw_text: str, schema: dict[str, Any]) -> str:
    """Build a structured extraction prompt that includes the JSON schema."""
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


def call_ollama(prompt: str, json_schema: dict[str, Any]) -> str:
    """Send a request to Ollama's /api/generate endpoint and return the raw response text."""
    payload: dict[str, Any] = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": json_schema,
    }

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    with request.urlopen(req) as response:
        raw_response = json.loads(response.read().decode("utf-8"))
        return raw_response.get("response", "").strip()


def get_movie_data(raw_text: str) -> Movie:
    """Extract and validate a Movie from natural-language text via Ollama with retry."""
    json_schema = Movie.model_json_schema()
    prompt = build_prompt(raw_text, json_schema)

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            content = call_ollama(prompt, json_schema)
            if not content:
                raise ValueError("Ollama returned an empty response")

            return Movie.model_validate_json(content)

        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES} — parse error: {e}")
        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES} — unexpected error: {e}")

    raise RuntimeError(f"All {MAX_RETRIES} attempts failed. Last error: {last_error}")


def main() -> None:
    test_inputs = [
        # 0 — multiple movies mentioned, must pick the one in focus
        "I saw a double feature tonight: first 'The Matrix' from 1999, then 'The Matrix Reloaded' in 2003 — both sci-fi action flicks that blow minds with bullet-time choreography.",

        # 1 — no title at all, only plot description
        "A washed-up boxer in Philadelphia trains for a final fight against his nemesis Ivan Drago, who killed his friend Apollo Creed — it's Rocky's fourth outing, a 1985 sports drama.",

        # 2 — ambiguous / contradictory year cues
        "They said 'Back to the Future' was made in 1986 but it's about time travel to both 1955 and 2015 — the actual film is 1985, a sci-fi comedy adventure.",

        # 3 — title appears only in a subtitle, year implied
        "That Pixar film where the robot WALL-E falls in love and cleans up Earth — 2008, animated post-apocalyptic romance comedy.",

        # 4 — colloquial, year as a range
        "Caught an old kung fu movie from the early 70s — you know the one, 'Enter the Dragon' with Brubeck's music? 1973, martial arts action.",

        # 5 — non-English original title
        "Saw '도시폭격' ('The Wailing') on a Korean horror stream last night — 2016, folk horror mystery thriller.",

        # 6 — title with numbers and special chars
        "The 2010 film '127 Hours' is Danny Boyle's survival biographical drama where a hiker amputates his own arm.",

        # 7 — title embedded in a sentence, no quotes
        "That 2014 Christopher Nolan movie Interstellar explores relativity and love across dimensions as Cooper pilots a team through a wormhole.",

        # 8 — very short / minimal info
        "Casablanca.",

        # 9 — very long summary, must be condensed to one sentence
        "Citizen Kane, Orson Welles' 1941 masterpiece, tells the story of Charles Foster Kane, a wealthy newspaper tycoon whose life is examined after his death through the investigation of his mysterious last word 'Rosebud', exploring themes of power, corruption, and the American Dream through revolutionary cinematography and narrative structure.",

        # 10 — genre is implied, not stated
        "The 2017 film Get Out follows a Black photographer who visits his white girlfriend's family estate and discovers they're harvesting Black bodies — it's Jordan Peele's directorial debut.",

        # 11 — title in a foreign language with English translation
        "Watched '라라랜드' (La La Land, 2016) — a modern jazz musical romance that pays homage to classic Hollywood and MGM musicals.",

        # 12 — year is wrong / red herring
        "Everyone thinks 'Titanic' is about the 1996 film, but the 1997 James Cameron epic actually tells the story of the doomed 1912 ship, blending romance, disaster, and 3D spectacle.",

        # 13 — title not mentioned, must infer from plot
        "A young wizard discovers he's famous after receiving a letter to attend a magic school, faces a dark wizard who killed his parents, and learns about sacrifice — that's the 1997 UK fantasy adventure.",

        # 14 — multiple genres mashed together
        "Shrek (2001) is an animated comedy that parodies Disney fairy tales with Ogre protagonists, featuring Lord Farquaad's obsession with perfect kingdom order and Donkey's wise-cracking sidekick energy.",

        # 15 — informal text-speak
        "omg u hav 2 c 'The Shining' from 1980 — Jack Nicholson goes insane at the Overlook Hotel n his daughter sees ghosts n Danny rides a tricycle thru blood halls. scary horror psychological thriller.",

        # 16 — title and year in middle of long sentence
        "I can't even right now, but 'Everything Everywhere All at Once' came out in 2022 and it's this absurdist multiverse comedy sci-fi thing where Michelle Yeoh runs a laundromat and fights alternate selves — mind-blowing stuff.",

        # 17 — year missing, title is obscure
        "That David Lynch film 'Mulholland Drive' — a neo-noir psychological mystery about an amnesiac woman and a failed actress in Hollywood, with a blue key and a dumpster and a cowboy.",

        # 18 — genres described as emotions/themes, not labels
        "The 2010 film 'Black Swan' follows Nina, a ballerina consumed by perfection as she transforms into the Black Swan in 'Swan Lake' — dark psychological horror thriller.",

        # 19 — very long title with colon
        "The Lord of the Rings: The Return of the King (2003) concludes the trilogy as Frodo and Sam struggle to destroy the One Ring in Mordor while armies clash at Minas Tirith — epic fantasy war drama from New Line Cinema.",
    ]

    count_success = 0
    count_failures = 0

    for i, input_text in enumerate(test_inputs):
        try:
            movie = get_movie_data(input_text)
            count_success += 1
            print(f"[{i:2d}] {movie.title} ({movie.year}) — {', '.join(movie.genres)}")
        except Exception as e:
            count_failures += 1
            print(f"[{i:2d}] Failed: {e}")

    print(f"\nResults: {count_success} succeeded, {count_failures} failed")


if __name__ == "__main__":
    main()
