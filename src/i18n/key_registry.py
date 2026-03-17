"""Registry of all required i18n message keys.

This module dynamically collects all message keys defined in keys.py
to provide a centralized list for validation purposes.
"""

from src.i18n import keys


def _collect_all_keys() -> frozenset[str]:
    """Collect all message key constants defined in the keys module.

    Returns all module-level string constants that contain a dot,
    which is the pattern for message keys (e.g., 'chat.input_prompt').
    """
    return frozenset(
        value
        for name, value in vars(keys).items()
        if isinstance(value, str)
        and not name.startswith("_")
        and "." in value
    )


# Collection of all required keys for validation
ALL_REQUIRED_KEYS = _collect_all_keys()
