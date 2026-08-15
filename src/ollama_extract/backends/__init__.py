"""Backend factory and registry."""

from ollama_extract.backends.base import OllamaBackend, BackendResponse, BackendName
from ollama_extract.backends.urllib_backend import UrllibBackend
from ollama_extract.backends.requests_backend import RequestsBackend
from ollama_extract.backends.ollama_library_backend import OllamaLibraryBackend

__all__ = ["BackendName", "OllamaBackend", "BackendResponse", "get_backend", "list_backends"]

_BACKENDS: dict[str, type[OllamaBackend]] = {
    "urllib": UrllibBackend,
    "requests": RequestsBackend,
    "ollama": OllamaLibraryBackend,
}


def get_backend(name: str, model_name: str, ollama_url: str) -> OllamaBackend:
    """Instantiate a backend by name."""
    name_lower = name.lower()
    if name_lower not in _BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Available: {list_backends()}")
    return _BACKENDS[name_lower](model_name=model_name, ollama_url=ollama_url)


def list_backends() -> list[str]:
    """Return available backend names."""
    return list(_BACKENDS.keys())
