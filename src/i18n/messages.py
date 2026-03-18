"""Message loading and retrieval for internationalization (i18n).

This module provides functions to load locale-specific static strings
from YAML files and retrieve them with optional string interpolation.

This module is aware of DEFAULT_RESPONSE_LANGUAGE to safely fallback to defaults.
"""

from pathlib import Path
from typing import Any

import yaml

from src.configs.common import DEFAULT_RESPONSE_LANGUAGE

# Explicitly declare supported languages for static strings; facilitates constant-time lookup
SUPPORTED_LANGUAGES: frozenset[str] = frozenset({"en", "fr"})

# Cache for loaded locale data to avoid re-reading files
_LOCALE_CACHE: dict[str, dict[str, Any]] = {}


def _get_locale_path(language: str) -> Path:
    """Get the path to a locale YAML file.

    Args:
        language: ISO 639-1 language code (e.g., "en", "fr")

    Returns:
        Path to the locale file
    """
    # Locale files are in project root /locales directory
    project_root = Path(__file__).parent.parent.parent
    return project_root / "locales" / f"{language}.yaml"


def load_messages(language: str) -> dict[str, Any]:
    """Load messages for a specific language from YAML file.

    Results are cached to avoid repeated file I/O. Falls back to
    DEFAULT_RESPONSE_LANGUAGE if the requested language is not available.

    Args:
        language: ISO 639-1 language code (e.g., "en", "fr")

    Returns:
        Dictionary of message keys to values

    Raises:
        yaml.YAMLError: If locale file is malformed
    """

    # Use cached messages if available
    if language in _LOCALE_CACHE:
        return _LOCALE_CACHE[language]

    # If not cached...

    # Check if language is supported; Fallback to default language if not supported
    # CI ensures that default response language is always supported.
    if language not in SUPPORTED_LANGUAGES:
        return load_messages(DEFAULT_RESPONSE_LANGUAGE)

    # Load localization yaml file
    locale_path = _get_locale_path(language)

    # Load messages within yaml file
    with locale_path.open("r", encoding="utf-8") as f:
        messages = yaml.safe_load(f)

    if not isinstance(messages, dict):
        raise ValueError(f"Locale file {locale_path} must be parseable to dict")

    # Cache for future calls
    _LOCALE_CACHE[language] = messages
    return messages


def _get_nested_value(data: dict[str, Any], key: str) -> str:
    """Get a nested value from a dictionary using dot notation.

    Args:
        data: Dictionary to search
        key: Dot-separated key path (e.g., "chat.chatting_with")

    Returns:
        The string value at the key path

    Raises:
        KeyError: If key path doesn't exist
        ValueError: If final value is not a string
    """
    parts = key.split(".")
    current = data

    for part in parts:
        if not isinstance(current, dict):
            raise KeyError(f"Invalid key path: {key}")
        current = current[part]

    if not isinstance(current, str):
        raise ValueError(
            f"Message value at '{key}' must be a string, got {type(current)}"
        )

    return current


def get_message(key: str, language: str, **kwargs: Any) -> str:
    """Get a localized message with optional string interpolation.

    Args:
        key: Dot-separated message key (e.g., "chat.chatting_with")
        language: ISO 639-1 language code (e.g., "en", "fr")
        **kwargs: Variables for string interpolation (e.g., author="voltaire")
                 Author names are automatically capitalized.

    Returns:
        Localized and formatted message string

    Raises:
        FileNotFoundError: If locale file doesn't exist
        KeyError: If message key doesn't exist
        ValueError: If message value is not a string
    """
    messages = load_messages(language)
    template = _get_nested_value(messages, key)

    # Perform string interpolation if kwargs provided, including capitalization of author
    if kwargs:
        if "author" in kwargs:
            kwargs["author"] = kwargs["author"].title()
        return template.format(**kwargs)

    return template


def clear_cache() -> None:
    """Clear the locale cache.

    Useful for testing or hot-reloading locale files.
    """
    _LOCALE_CACHE.clear()
