"""Test input generation — 30 hand-crafted edge-cases + programmatic generator."""

import random
from typing import Any

__all__ = ["TEST_INPUTS", "generate_inputs"]


# ---------------------------------------------------------------------------
#  Hand-crafted edge-case inputs (default test suite)
# ---------------------------------------------------------------------------

TEST_INPUTS: list[str] = [
    #  0 — title only in quotes, no genres
    "I saw 'Amélie' in 2001 — whimsical French film, no genres mentioned.",
    #  1 — director reference, must infer title
    "That Christopher Nolan 2010 film about dream invasion — you know, the spinning top ending.",
    #  2 — year as Roman numeral
    "The 1994 film 'Pulp Fiction' (MCMXCIV) — wait no that's not right, it's just 1994, a crime anthology.",
    #  3 — multiple movies, must pick the one described
    "Like 'Alien' from 1979 and 'Aliens' from 1992, 'The Shining' (1980) is Kubrick's horror masterpiece — not the other two.",
    #  4 — title implied through plot, never named
    "A 1999 sci-fi action film about a hacker who learns reality is an illusion and takes a red pill — you know the one.",
    #  5 — genre described as a feeling
    "That 2007 film 'There Will Be Blood' — it's oozing with greed, capitalism, and oil-soaked madness.",
    #  6 — very long compound title, no year given
    "The Lord of the Rings: The Return of the King concludes the trilogy — epic fantasy war from 2003.",
    #  7 — deliberate year misinformation
    "Everyone says 'Titanic' is from 1996, but James Cameron's epic disaster romance is actually 1997.",
    #  8 — title in a URL-like format
    "I found a file called 'the-matrix-1999-dvdrip' — that's the cyberpunk action classic right?",
    #  9 — text-speak / abbreviations
    "lmao 'Get Out' 2017 best horror social thriller everrr jordan peele killed it.",
    # 10 — non-English description, English title
    "이 영화 '인셍' 2010 년 작품은 드림 속에서 비밀을 훔치는 팀을 보여줘요. sci-fi heist.",
    # 11 — genre as single-word emotions
    "That 2012 film 'Argo' feels like suspense, tension, and nervous anxiety wrapped in a CIA thriller.",
    # 12 — nothing but a year (impossible extraction)
    "1994.",
    # 13 — two movies mashed into one sentence
    "It's like 'Casablanca' (1942) meets 'The English Patient' (1996) — two wartime romances, make a 1942 romantic drama about a cynical bartender.",
    # 14 — title is a common word, quoted
    "'Her' (2013) — a man falls in love with his AI operating system in this Spike Jonze sci-fi romance.",
    # 15 — genres described with comma-separated string in narrative
    "The 1972 film 'The Godfather' is crime, drama, and family saga all rolled into one mafia epic.",
    # 16 — year spelled out
    "I saw a movie about a ring that gives you supernatural powers — it's 'The Ring' from two thousand and one, a 2001 supernatural horror thriller remake.",
    # 17 — title is just one letter (impossible)
    "'",
    # 18 — input is a haiku
    "Dark knight rises once, Gotham burned and reborn, 2012 ends the trilogy.",
    # 19 — fake movie, must use null/unknown
    "I just watched 'The Zephyrian Protocol' from 2025 — a time-bending quantum espionage thriller that doesn't exist.",
    # 20 — title has typo in input
    "Just saw 'The Matirx' (1999) — a groundbreaking cyberpunk sci-fi action film with bullet-time.",
    # 21 — genre described through cinematography
    "That 1960 Hitchcock film 'Psycho' — shrieking violins, shower stabbing, psychological horror thriller at the Bates Motel.",
    # 22 — input is a movie review excerpt
    "As Roger Ebert said, 'Spirited Away' (2001) is 'an animated masterpiece where a shy girl navigates a spirit world' — pure magic from Miyazaki.",
    # 23 — title only in parentheses
    "That film from 2008 where the Joker says 'why so serious' — you know the Batman one.",
    # 24 — input mixes real and fictional
    "In 'The Lord of the Rings: The Fellowship of the Ring' (2001), Frodo must destroy the One Ring — fantasy adventure from 2013? No, 2001.",
    # 25 — genre through soundtrack description
    "That 1977 space opera 'Star Wars' with the epic orchestral score, lightsaber duels, and 'the force' — George Lucas's space fantasy epic.",
    # 26 — title in a foreign script with English subtitle
    "看过 2009 年的 'Inception'，这部 sci-fi 电影讲述了一个进入梦中的团队如何窃取秘密。",
    # 27 — nothing but a genre
    "Horror.",
    # 28 — title appears in both opening and closing
    "I just watched 'WALL-E'. That 2008 Pixar film 'WALL-E' — an animated post-apocalyptic romance comedy about a robot who finds love and saves Earth.",
    # 29 — contradictory years + no explicit title
    "That film from 2010, no wait 2011, the Christopher Nolan one about dreams within dreams with a spinning top — release year is 2010.",
]


# ---------------------------------------------------------------------------
#  Programmatic input generator (for --count > 30)
# ---------------------------------------------------------------------------

REAL_TITLES = [
    "Inception", "The Godfather", "Pulp Fiction", "The Dark Knight", "Fight Club",
    "Forrest Gump", "The Matrix", "Interstellar", "Spirited Away", "Amélie",
    "Casablanca", "The Lord of the Rings: The Return of the King",
    "Mad Max: Fury Road", "The Shawshank Redemption", "The Prestige", "Memento",
    "The Departed", "There Will Be Blood", "No Country for Old Men",
    "Her", "Moonlight", "Arrival", "Blade Runner 2049", "La La Land",
    "Get Out", "Us", "Black Panther", "Parasite", "1917", "Joker", "Dune",
]

FAKE_TITLES = [
    "The Zephyrian Protocol", "Quantum Paradox", "The Crimson Void",
    "Echoes of Yesterday", "The Last Horizon", "Project Nebula",
]

PLOTS = [
    "a thief who steals secrets from dreams",
    "a botanist who discovers a mysterious plant with deadly consequences",
    "a detective who hunts a serial killer copycat",
    "a young lion prince who must embrace his destiny",
    "a hacker who learns reality is an illusion and takes a red pill",
    "a retired assassin who comes out of retirement for one last job",
    "a journalist who uncovers a conspiracy that reaches the highest levels",
    "a scientist who creates artificial life with unintended consequences",
    "a teacher who discovers a student's dark secret",
    "a pilot who must deliver humanity's last hope across the galaxy",
]

GENRE_WORDS = ["sci-fi", "crime", "drama", "action", "horror", "comedy", "thriller", "fantasy", "romance"]

YEAR_STRINGS = ["2010", "1994", "2001", "2017", "1997", "1977", "1999", "2008", "1942", "1972", "2012", "2016"]
ROMAN_NUMERALS = {
    "1942": "MCMXLII", "1972": "MCMLXXII", "1977": "MCMLXXVII", "1985": "MCMLXXXV",
    "1994": "MCMXCIV", "1996": "MCMXCVI", "1997": "MCMXCVII", "1999": "MCMXCIX",
    "2001": "MMI", "2010": "MMX",
}
DIRECTORS = ["Christopher Nolan", "Quentin Tarantino", "Steven Spielberg", "Martin Scorsese", "Denis Villeneuve"]
CRITICS = ["Roger Ebert", "Peter Travers", "A.O. Scott", "Kenneth Turan"]


def _gen_standard(rng: random.Random) -> str:
    title = rng.choice(REAL_TITLES + FAKE_TITLES)
    year = rng.choice(YEAR_STRINGS)
    genre = rng.choice(GENRE_WORDS)
    plot = rng.choice(PLOTS)
    return f"I just watched '{title}' — it's a {year} {genre} film about {plot}."


def _gen_director(rng: random.Random) -> str:
    director = rng.choice(DIRECTORS)
    year = rng.choice(YEAR_STRINGS)
    plot = rng.choice(PLOTS)
    return f"That {director} {year} film about {plot} — you know the one."


def _gen_url_style(rng: random.Random) -> str:
    title = rng.choice(REAL_TITLES)
    year = rng.choice(YEAR_STRINGS)
    slug = title.lower().replace(" ", "-").replace(":", "").replace("'", "")
    return f"Found a file: {slug}-{year}-dvdrip.mkv — that's the right movie?"


def _gen_text_speak(rng: random.Random) -> str:
    title = rng.choice(REAL_TITLES)
    year = rng.choice(YEAR_STRINGS)
    genre = rng.choice(GENRE_WORDS)
    return f"lol just saw '{title}' {year} so {genre} best movie ever tbh"


def _gen_review(rng: random.Random) -> str:
    critic = rng.choice(CRITICS)
    title = rng.choice(REAL_TITLES + FAKE_TITLES)
    year = rng.choice(YEAR_STRINGS)
    genre = rng.choice(GENRE_WORDS)
    return f"As {critic} said, '{title}' ({year}) is a {genre} masterpiece."


def _gen_no_title(rng: random.Random) -> str:
    year = rng.choice(YEAR_STRINGS)
    genre = rng.choice(GENRE_WORDS)
    plot = rng.choice(PLOTS)
    return f"A {year} {genre} film about {plot}."


def _gen_roman_year(rng: random.Random) -> str:
    title = rng.choice(REAL_TITLES)
    year_key = rng.choice(list(ROMAN_NUMERALS.keys()))
    roman = ROMAN_NUMERALS[year_key]
    return f"The {year_key} film '{title}' ({roman}) — a must-see."


def _gen_contradictory(rng: random.Random) -> str:
    title = rng.choice(REAL_TITLES)
    y1, y2 = rng.sample(YEAR_STRINGS, 2)
    return f"I saw '{title}' in {y1}... no wait, it was {y2}."


def _gen_haiku(rng: random.Random) -> str:
    year = rng.choice(YEAR_STRINGS)
    return f"Movie title fades, themes linger in mind, {year} cinema."


def _gen_minimal(rng: random.Random) -> str:
    option = rng.randint(0, 2)
    if option == 0:
        return rng.choice(YEAR_STRINGS) + "."
    if option == 1:
        return "'"
    return rng.choice(GENRE_WORDS) + "."


def _gen_fake_movie(rng: random.Random) -> str:
    title = rng.choice(FAKE_TITLES)
    year = rng.choice(YEAR_STRINGS)
    return f"I just watched '{title}' from {year} — a time-bending thriller that doesn't exist."


def _gen_multiple_movies(rng: random.Random) -> str:
    t1, t2 = rng.sample(REAL_TITLES, 2)
    target = rng.choice(REAL_TITLES + FAKE_TITLES)
    year = rng.choice(YEAR_STRINGS)
    return f"Like '{t1}' and '{t2}', '{target}' ({year}) stands out."


def _gen_foreign_mixed(rng: random.Random) -> str:
    title = rng.choice(REAL_TITLES)
    year = rng.choice(YEAR_STRINGS)
    phrases = ["이 영화", "看过", " cette film", "dieser Film"]
    phrase = rng.choice(phrases)
    return f"{phrase} '{title}' {year} — truly unforgettable."


_GENERATORS = [
    _gen_standard, _gen_director, _gen_url_style, _gen_text_speak,
    _gen_review, _gen_no_title, _gen_roman_year, _gen_contradictory,
    _gen_haiku, _gen_minimal, _gen_fake_movie, _gen_multiple_movies,
    _gen_foreign_mixed,
]


def generate_inputs(n: int, seed: int = 42) -> list[str]:
    """Generate *n* diverse movie-extraction test inputs using a seeded RNG."""
    rng = random.Random(seed)
    return [rng.choice(_GENERATORS)(rng) for _ in range(n)]
