"""Unit tests for Ollama health check utility"""

import json
import subprocess
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest

from src.utils.ollama_health import (
    OllamaRuntimeState,
    _parse_etime_to_seconds,
    capture_ollama_state,
    check_ollama_available,
)

SAMPLE_PID = "4242"
SAMPLE_ETIME = "01:00:00"
SAMPLE_UPTIME_SECONDS = "3600"
SAMPLE_VERSION = "0.1.42"
SAMPLE_LOADED_MODELS_PAYLOAD: dict[str, list[dict[str, str]]] = {
    "models": [{"name": "sample-model", "expires_at": "2030-01-01T00:00:00Z"}]
}


def _lsof_result(pid: str | None) -> MagicMock:
    """Build a fake CompletedProcess for the lsof PID lookup."""
    stdout = f"p{pid}\n" if pid is not None else ""
    return MagicMock(returncode=0 if pid is not None else 1, stdout=stdout)


def _ps_result(etime: str) -> MagicMock:
    """Build a fake CompletedProcess for the `ps -o etime=` uptime lookup."""
    return MagicMock(returncode=0, stdout=f"{etime}\n")


def _http_response(payload: object) -> MagicMock:
    """Build a fake urlopen context manager returning JSON bytes."""
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode("utf-8")
    context = MagicMock()
    context.__enter__.return_value = response
    context.__exit__.return_value = False
    return context


class TestCheckOllamaAvailable:
    """Tests for check_ollama_available."""

    @patch("src.utils.ollama_health.urllib.request.urlopen")
    def test_returns_none_when_ollama_is_running(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = MagicMock()
        result = check_ollama_available()
        assert result is None
        mock_urlopen.assert_called_once_with("http://localhost:11434/api/tags")

    @patch("src.utils.ollama_health.urllib.request.urlopen")
    def test_raises_runtime_error_on_connection_refused(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.side_effect = URLError(ConnectionRefusedError("Connection refused"))
        with pytest.raises(RuntimeError, match="Ollama is not running"):
            check_ollama_available()

    @patch("src.utils.ollama_health.urllib.request.urlopen")
    def test_raises_runtime_error_on_url_error(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.side_effect = URLError("Name or service not known")
        with pytest.raises(RuntimeError, match="Ollama is not running"):
            check_ollama_available()

    @patch("src.utils.ollama_health.urllib.request.urlopen")
    def test_uses_custom_base_url(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = MagicMock()
        check_ollama_available(base_url="http://remote:9999")
        mock_urlopen.assert_called_once_with("http://remote:9999/api/tags")


class TestCaptureOllamaState:
    """Tests for capture_ollama_state."""

    @patch("src.utils.ollama_health.urllib.request.urlopen")
    @patch("src.utils.ollama_health.subprocess.run")
    def test_happy_path_populates_all_fields(
        self, mock_run: MagicMock, mock_urlopen: MagicMock
    ) -> None:
        mock_run.side_effect = [
            _lsof_result(SAMPLE_PID),
            _ps_result(SAMPLE_ETIME),
        ]
        mock_urlopen.side_effect = [
            _http_response({"version": SAMPLE_VERSION}),
            _http_response(SAMPLE_LOADED_MODELS_PAYLOAD),
        ]

        state = capture_ollama_state()

        assert isinstance(state, OllamaRuntimeState)
        assert state.pid == SAMPLE_PID
        assert state.uptime_seconds == SAMPLE_UPTIME_SECONDS
        assert state.version == SAMPLE_VERSION
        assert state.loaded_models is not None
        assert json.loads(state.loaded_models) == SAMPLE_LOADED_MODELS_PAYLOAD

    @patch("src.utils.ollama_health.urllib.request.urlopen")
    @patch("src.utils.ollama_health.subprocess.run")
    def test_lsof_failure_leaves_pid_and_uptime_none(
        self, mock_run: MagicMock, mock_urlopen: MagicMock
    ) -> None:
        mock_run.side_effect = [_lsof_result(None)]
        mock_urlopen.side_effect = [
            _http_response({"version": SAMPLE_VERSION}),
            _http_response(SAMPLE_LOADED_MODELS_PAYLOAD),
        ]

        state = capture_ollama_state()

        assert state.pid is None
        assert state.uptime_seconds is None
        assert state.version == SAMPLE_VERSION
        assert state.loaded_models is not None

    @patch("src.utils.ollama_health.urllib.request.urlopen")
    @patch("src.utils.ollama_health.subprocess.run")
    def test_ps_api_failure_only_nulls_loaded_models(
        self, mock_run: MagicMock, mock_urlopen: MagicMock
    ) -> None:
        mock_run.side_effect = [
            _lsof_result(SAMPLE_PID),
            _ps_result(SAMPLE_ETIME),
        ]
        mock_urlopen.side_effect = [
            _http_response({"version": SAMPLE_VERSION}),
            URLError("connection refused"),
        ]

        state = capture_ollama_state()

        assert state.pid == SAMPLE_PID
        assert state.uptime_seconds == SAMPLE_UPTIME_SECONDS
        assert state.version == SAMPLE_VERSION
        assert state.loaded_models is None

    @patch("src.utils.ollama_health.urllib.request.urlopen")
    @patch("src.utils.ollama_health.subprocess.run")
    def test_version_api_failure_only_nulls_version(
        self, mock_run: MagicMock, mock_urlopen: MagicMock
    ) -> None:
        mock_run.side_effect = [
            _lsof_result(SAMPLE_PID),
            _ps_result(SAMPLE_ETIME),
        ]
        mock_urlopen.side_effect = [
            URLError("connection refused"),
            _http_response(SAMPLE_LOADED_MODELS_PAYLOAD),
        ]

        state = capture_ollama_state()

        assert state.pid == SAMPLE_PID
        assert state.uptime_seconds == SAMPLE_UPTIME_SECONDS
        assert state.version is None
        assert state.loaded_models is not None

    @patch("src.utils.ollama_health.urllib.request.urlopen")
    @patch("src.utils.ollama_health.subprocess.run")
    def test_subprocess_timeout_returns_none_pid_without_raising(
        self, mock_run: MagicMock, mock_urlopen: MagicMock
    ) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="lsof", timeout=2)
        mock_urlopen.side_effect = [
            _http_response({"version": SAMPLE_VERSION}),
            _http_response(SAMPLE_LOADED_MODELS_PAYLOAD),
        ]

        state = capture_ollama_state()

        assert state.pid is None
        assert state.uptime_seconds is None
        assert state.version == SAMPLE_VERSION
        assert state.loaded_models is not None


class TestParseEtimeToSeconds:
    """Tests for parsing `ps -o etime=` output into seconds."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("00:05", 5),
            ("12:34", 12 * 60 + 34),
            ("01:00:00", 3600),
            ("02:03:04", 2 * 3600 + 3 * 60 + 4),
            ("1-00:00:00", 86400),
            ("02-08:46:05", 2 * 86400 + 8 * 3600 + 46 * 60 + 5),
            ("  12:34  \n", 12 * 60 + 34),
        ],
    )
    def test_parses_valid_formats(self, raw: str, expected: int) -> None:
        assert _parse_etime_to_seconds(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "   ",
            "not-a-time",
            "12",
            "12:34:56:78",
            "aa:bb",
            "-01:00:00",
            "1-",
            "1-aa:bb:cc",
        ],
    )
    def test_returns_none_for_malformed_input(self, raw: str) -> None:
        assert _parse_etime_to_seconds(raw) is None
