"""Tests for Ollama health check utility."""

from unittest.mock import patch, MagicMock
from urllib.error import URLError

import pytest

from src.utils.ollama_health import check_ollama_available


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
