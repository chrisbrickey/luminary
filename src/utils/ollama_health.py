"""Health check and runtime-state capture for the local Ollama service."""

import json
import subprocess
import urllib.request
from urllib.error import URLError

from pydantic import BaseModel

_SUBPROCESS_TIMEOUT_SECONDS = 2


class OllamaRuntimeState(BaseModel):
    """Best-effort snapshot of the local Ollama server's runtime state.

    A None field means that piece of state could not be discovered
    (Ollama not running, HTTP call failed, subprocess timed out, etc.).
    Used to populate SystemSnapshot; not persisted directly.
    """

    pid: str | None = None
    uptime_seconds: str | None = None
    version: str | None = None
    loaded_models: str | None = None


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


def capture_ollama_state(
    base_url: str = "http://localhost:11434",
    port: int = 11434,
) -> OllamaRuntimeState:
    """Best-effort capture of Ollama process + server state. Never raises."""
    pid = _discover_ollama_pid(port)
    return OllamaRuntimeState(
        pid=pid,
        uptime_seconds=_discover_uptime_seconds(pid) if pid is not None else None,
        version=_fetch_ollama_version(base_url),
        loaded_models=_fetch_loaded_models(base_url),
    )


def _discover_ollama_pid(port: int) -> str | None:
    try:
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fp"],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if line.startswith("p"):
            candidate = line[1:].strip()
            if candidate.isdigit():
                return candidate
    return None


def _discover_uptime_seconds(pid: str) -> str | None:
    # Uses `etime` (portable) not `etimes` (Linux-only); output is [[DD-]HH:]MM:SS.
    try:
        result = subprocess.run(
            ["ps", "-o", "etime=", "-p", pid],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    seconds = _parse_etime_to_seconds(result.stdout)
    return str(seconds) if seconds is not None else None


def _parse_etime_to_seconds(value: str) -> int | None:
    """Parse ps `etime` output ([[DD-]HH:]MM:SS) into an integer number of seconds."""
    value = value.strip()
    if not value:
        return None
    days = 0
    if "-" in value:
        days_str, _, value = value.partition("-")
        if not days_str.isdigit():
            return None
        days = int(days_str)
    parts = value.split(":")
    if len(parts) not in (2, 3) or not all(p.isdigit() for p in parts):
        return None
    numbers = [int(p) for p in parts]
    hours, minutes, seconds = (0, *numbers) if len(numbers) == 2 else tuple(numbers)
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _fetch_ollama_version(base_url: str) -> str | None:
    try:
        with urllib.request.urlopen(f"{base_url}/api/version") as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, ConnectionError, ValueError, OSError):
        return None
    version = payload.get("version") if isinstance(payload, dict) else None
    return str(version) if version is not None else None


def _fetch_loaded_models(base_url: str) -> str | None:
    try:
        with urllib.request.urlopen(f"{base_url}/api/ps") as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, ConnectionError, ValueError, OSError):
        return None
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)
