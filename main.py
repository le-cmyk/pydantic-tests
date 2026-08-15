import json
from urllib import request
from typing import List
from pydantic import BaseModel, Field

# 1. Define the Pydantic Model
class Movie(BaseModel):
    title: str = Field(description="The title of the movie")
    year: int = Field(description="The release year")
    genres: List[str] = Field(description="A list of genres")
    summary: str = Field(description="A short one-sentence summary")

def get_movie_data(prompt: str) -> Movie:
    url = "http://localhost:11434/api/generate"
    
    # Important: Ensure your model name matches exactly what 'ollama list' shows
    payload = {
        "model": "gemma4:12b", 
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"}
    )

    print(f"--- Debug: Sending request to {url} ---")
    with request.urlopen(req) as response:
        raw_bytes = response.read()
        raw_response = json.loads(raw_bytes.decode("utf-8"))
        
        # This is the actual text from the AI
        content_string = raw_response.get("response", "")
        
        print(f"--- Debug: Raw content from LLM ---\n{content_string}")

        if not content_string.strip():
            raise ValueError("Ollama returned an empty string.")

        # Parse the JSON from the AI response
        raw_json_dict = json.loads(content_string)
        return Movie.model_validate(raw_json_dict)



def main():

    # --- Execution ---
    raw_text = "I just watched 'Inception' - it's a sci-fi heist movie from 2010."
    try:
        result = get_movie_data(f"""Extract movie info into
        JSON make sure to respond only with the json no markdown:
        this is the class for the output
class Movie(BaseModel):
    title: str = Field(description="The title of the movie")
    year: int = Field(description="The release year")
    genres: List[str] = Field(description="A list of genres")
    summary: str = Field(description="A short one-sentence summary")
{raw_text}""")

        # Now you have full IDE autocomplete and type safety!
        print(f"Title: {result.title}")
        print(f"Year:  {result.year}")
        print(f"Genres:{', '.join(result.genres)}")
        print(f"Summary: {result.summary}")

    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()
