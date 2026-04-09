"""Logging utilities for CLI scripts.

This module provides standardized logging configuration for all CLI scripts
in the project, ensuring consistent format and behavior.
"""

import logging


def setup_cli_logging(verbose: bool = False) -> logging.Logger:
    """Configure detailed logging for CLI scripts.

    Sets up logging with timestamp, logger name, and level information.
    Returns a logger for the calling module.

    Args:
        verbose: If True, sets DEBUG level; otherwise INFO (default: False)

    Returns:
        Logger instance for the calling module

    Example:
        >>> logger = setup_cli_logging(verbose=True)
        >>> logger.info("Processing started")
        2026-04-09 10:30:45,123 - __main__ - INFO - Processing started
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True  # Allow reconfiguration (Python 3.8+)
    )
    return logging.getLogger(__name__)
