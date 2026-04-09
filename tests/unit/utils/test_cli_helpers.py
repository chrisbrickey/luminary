"""Tests for src/utils/cli_helpers.py - common CLI utilities."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.utils.cli_helpers import (
    check_ollama_or_exit,
    exit_on_error,
    resolve_authors,
    validate_author,
)


class TestValidateAuthor:
    """Tests for validate_author function."""

    def test_valid_author_passes(self) -> None:
        """Should not raise exception for valid author."""
        # voltaire is a configured author
        validate_author("voltaire")
        # If we get here without exception, test passes

    def test_invalid_author_raises_value_error(self) -> None:
        """Should raise ValueError for unknown author."""
        with pytest.raises(ValueError, match="Unknown author: invalid_author"):
            validate_author("invalid_author")

    def test_error_message_includes_available_authors(self) -> None:
        """Should include list of available authors in error message."""
        with pytest.raises(ValueError, match="Available authors:"):
            validate_author("nonexistent")


class TestResolveAuthors:
    """Tests for resolve_authors function."""

    def test_returns_all_authors_when_none(self) -> None:
        """Should return all configured authors when author_arg is None."""
        logger = MagicMock(spec=logging.Logger)

        authors = resolve_authors(None, logger)

        # Should return a list with multiple authors
        assert isinstance(authors, list)
        assert len(authors) > 0
        assert "voltaire" in authors

    def test_logs_all_authors_message_when_none(self) -> None:
        """Should log appropriate message when processing all authors."""
        logger = MagicMock(spec=logging.Logger)

        resolve_authors(None, logger)

        logger.info.assert_called_once()
        call_arg = logger.info.call_args[0][0]
        assert "processing all configured authors" in call_arg.lower()

    def test_returns_single_author_list_when_specified(self) -> None:
        """Should return list with single author when specified."""
        logger = MagicMock(spec=logging.Logger)

        authors = resolve_authors("voltaire", logger)

        assert authors == ["voltaire"]

    def test_logs_specific_author_message(self) -> None:
        """Should log appropriate message when processing specific author."""
        logger = MagicMock(spec=logging.Logger)

        resolve_authors("voltaire", logger)

        logger.info.assert_called_once()
        call_arg = logger.info.call_args[0][0]
        assert "voltaire" in call_arg.lower()


class TestCheckOllamaOrExit:
    """Tests for check_ollama_or_exit function."""

    @patch("src.utils.cli_helpers.check_ollama_available")
    def test_passes_when_ollama_available(self, mock_check: MagicMock) -> None:
        """Should not exit when Ollama is available."""
        logger = MagicMock(spec=logging.Logger)
        mock_check.return_value = None  # Success

        check_ollama_or_exit(logger)

        # Should call check_ollama_available
        mock_check.assert_called_once()
        # Should not log any errors
        logger.error.assert_not_called()

    @patch("src.utils.cli_helpers.sys.exit")
    @patch("src.utils.cli_helpers.check_ollama_available")
    def test_exits_when_ollama_unavailable(
        self, mock_check: MagicMock, mock_exit: MagicMock
    ) -> None:
        """Should exit with code 1 when Ollama is unavailable."""
        logger = MagicMock(spec=logging.Logger)
        mock_check.side_effect = RuntimeError("Ollama not running")

        check_ollama_or_exit(logger)

        # Should log error
        logger.error.assert_called_once_with("Ollama not running")
        # Should exit with code 1
        mock_exit.assert_called_once_with(1)


class TestExitOnError:
    """Tests for exit_on_error function."""

    @patch("src.utils.cli_helpers.sys.exit")
    def test_logs_error_without_context(self, mock_exit: MagicMock) -> None:
        """Should log error message without context."""
        logger = MagicMock(spec=logging.Logger)
        error = ValueError("Test error")

        exit_on_error(logger, error)

        logger.error.assert_called_once_with("Test error")
        mock_exit.assert_called_once_with(1)

    @patch("src.utils.cli_helpers.sys.exit")
    def test_logs_error_with_context(self, mock_exit: MagicMock) -> None:
        """Should log error message with context."""
        logger = MagicMock(spec=logging.Logger)
        error = ValueError("Test error")

        exit_on_error(logger, error, context="during testing")

        # Should log with context and exc_info=True
        logger.error.assert_called_once_with(
            "Error during testing: Test error",
            exc_info=True
        )
        mock_exit.assert_called_once_with(1)

    @patch("src.utils.cli_helpers.sys.exit")
    def test_calls_sys_exit_with_1(self, mock_exit: MagicMock) -> None:
        """Should always call sys.exit(1)."""
        logger = MagicMock(spec=logging.Logger)
        error = RuntimeError("Any error")

        exit_on_error(logger, error)

        mock_exit.assert_called_once_with(1)
