"""Common utilities for CLI scripts.

This module provides shared functionality used across multiple CLI scripts,
reducing code duplication and standardizing patterns.
"""

import logging
import sys

from src.configs.loader_configs import INGEST_CONFIGS
from src.utils.ollama_health import check_ollama_available


def validate_author(author: str) -> None:
    """Validate that an author exists in INGEST_CONFIGS.

    Args:
        author: Author key to validate

    Raises:
        ValueError: If author is not found in INGEST_CONFIGS
    """
    if author not in INGEST_CONFIGS:
        raise ValueError(
            f"Unknown author: {author}. "
            f"Available authors: {', '.join(INGEST_CONFIGS.keys())}"
        )


def resolve_authors(author_arg: str | None, logger: logging.Logger) -> list[str]:
    """Resolve author argument to list of authors to process.

    If author_arg is None, returns all configured authors.
    If author_arg is provided, returns a single-item list with that author.
    Logs the selection to inform the user.

    Args:
        author_arg: Author key from command-line argument (None means all)
        logger: Logger instance for output

    Returns:
        List of author keys to process
    """
    if author_arg is None:
        authors = list(INGEST_CONFIGS.keys())
        logger.info(f"No author specified - processing all configured authors: {', '.join(authors)}")
    else:
        authors = [author_arg]
        logger.info(f"Processing author: {author_arg}")

    return authors


def check_ollama_or_exit(logger: logging.Logger) -> None:
    """Check Ollama availability and exit if unavailable.

    This is a convenience wrapper around check_ollama_available() that
    handles errors by logging and exiting, following the pattern used
    in CLI scripts.

    Args:
        logger: Logger instance for error output

    Note:
        Calls sys.exit(1) on failure. The return statement after sys.exit
        is for testing purposes when sys.exit is mocked.
    """
    try:
        check_ollama_available()
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)
        return  # For testing when sys.exit is mocked


def exit_on_error(logger: logging.Logger, error: Exception, context: str = "") -> None:
    """Log an error and exit with code 1.

    This standardizes the error-handling-and-exit pattern used across
    CLI scripts. Logs the error message and optionally includes context
    about what operation failed.

    Args:
        logger: Logger instance for error output
        error: The exception that occurred
        context: Optional context string (e.g., "during ingestion", "during scraping")

    Note:
        Calls sys.exit(1). The return statement after sys.exit is for
        testing purposes when sys.exit is mocked.
    """
    if context:
        logger.error(f"Error {context}: {error}", exc_info=True)
    else:
        logger.error(str(error))
    sys.exit(1)
    return  # For testing when sys.exit is mocked
