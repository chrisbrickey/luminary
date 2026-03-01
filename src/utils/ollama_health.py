"""Health check for Ollama service availability."""

import urllib.request
from urllib.error import URLError


def check_ollama_available(base_url: str = "http://localhost:11434") -> None:
    """Verify that Ollama is running and reachable.

    Sends a GET request to the Ollama API tags endpoint.

    Args:
        base_url: The base URL of the Ollama service.

    Raises:
        RuntimeError: If Ollama is not reachable.
    """
    try:
        urllib.request.urlopen(f"{base_url}/api/tags")
    except (URLError, ConnectionError):
        raise RuntimeError("Ollama is not running. Start it with: ollama serve")
