"""External integration tests for Ollama runtime-state capture.

IMPORTANT: These tests make real calls against a locally running Ollama server.
- Requires Ollama to be running on localhost:11434 (`ollama serve`)
- Run with: uv run pytest -m external
- Excluded from default test run

These tests exist because the unit tests mock subprocess and urllib, which meant
a subtle bug (using Linux-only `etimes` instead of portable `etime`) shipped and
left `ollama_uptime_seconds` null in real eval artifacts. This suite catches
platform-specific regressions in the actual command invocations.
"""

import pytest

from src.utils.ollama_health import OllamaRuntimeState, capture_ollama_state, check_ollama_available


@pytest.fixture(scope="module")
def live_ollama_state() -> OllamaRuntimeState:
    """Capture Ollama state once for all tests in this module."""
    check_ollama_available()
    return capture_ollama_state()


@pytest.mark.integration
@pytest.mark.external
def test_capture_populates_pid(live_ollama_state: OllamaRuntimeState) -> None:
    assert live_ollama_state.pid is not None
    assert live_ollama_state.pid.isdigit()


@pytest.mark.integration
@pytest.mark.external
def test_capture_populates_uptime_seconds(live_ollama_state: OllamaRuntimeState) -> None:
    assert live_ollama_state.uptime_seconds is not None
    assert live_ollama_state.uptime_seconds.isdigit()
    assert int(live_ollama_state.uptime_seconds) >= 0


@pytest.mark.integration
@pytest.mark.external
def test_capture_populates_version(live_ollama_state: OllamaRuntimeState) -> None:
    assert live_ollama_state.version is not None
    assert live_ollama_state.version.strip() != ""


@pytest.mark.integration
@pytest.mark.external
def test_capture_populates_loaded_models(live_ollama_state: OllamaRuntimeState) -> None:
    import json

    assert live_ollama_state.loaded_models is not None
    payload = json.loads(live_ollama_state.loaded_models)
    assert isinstance(payload, dict)
    assert "models" in payload
