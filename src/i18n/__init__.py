"""Localization (i18n) module for user-facing strings.

This module provides localized messages for the chat interface in multiple languages.
Locale strings are stored in YAML files in the locales/ directory.
"""

from src.i18n.messages import clear_cache, get_message, load_messages

__all__ = ["get_message", "load_messages", "clear_cache"]
