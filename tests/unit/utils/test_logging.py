"""Tests for src/utils/logging.py - logging utilities for CLI scripts."""

import logging


from src.utils.logging import setup_cli_logging


class TestSetupDetailedLogging:
    """Tests for setup_detailed_logging function."""

    def test_returns_logger_instance(self) -> None:
        """Should return a Logger instance."""
        logger = setup_cli_logging()
        assert isinstance(logger, logging.Logger)

    def test_default_log_level_is_info(self) -> None:
        """Should set INFO level by default when verbose=False."""
        setup_cli_logging(verbose=False)
        assert logging.root.level == logging.INFO

    def test_verbose_sets_debug_level(self) -> None:
        """Should set DEBUG level when verbose=True."""
        setup_cli_logging(verbose=True)
        assert logging.root.level == logging.DEBUG

    def test_configures_detailed_format(self) -> None:
        """Should configure detailed format with timestamp, name, and level."""
        setup_cli_logging(verbose=True)

        # Verify that at least one handler was added to root logger
        assert len(logging.root.handlers) > 0

        # Verify the handler has the expected format
        handler = logging.root.handlers[0]
        formatter = handler.formatter
        assert formatter is not None

        # Verify format includes key components (timestamp, name, level, message)
        # Format string is: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        format_str = formatter._fmt
        assert format_str is not None
        assert "%(asctime)s" in format_str
        assert "%(name)s" in format_str
        assert "%(levelname)s" in format_str
        assert "%(message)s" in format_str

    def test_multiple_calls_allow_reconfiguration(self) -> None:
        """Should allow reconfiguration across multiple calls (force=True)."""
        logger1 = setup_cli_logging(verbose=False)
        assert logging.root.level == logging.INFO

        logger2 = setup_cli_logging(verbose=True)
        assert logging.root.level == logging.DEBUG

        # Both loggers should still be valid Logger instances
        assert isinstance(logger1, logging.Logger)
        assert isinstance(logger2, logging.Logger)
