"""Entry point for the ollama_extract benchmarking tool.

Usage:
    uv run main.py --count 30
    uv run main.py --count 3000 --workers 4
    python -m ollama_extract --help
"""

from ollama_extract.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
