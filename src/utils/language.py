"""Language detection and localization utilities for user questions."""

from langdetect import LangDetectException, detect_langs  # type: ignore[import-untyped]

from src.i18n import get_message
from src.i18n.keys import STATUS_REFLECTING_SHORT, STATUS_REFLECTING_VERBOSE


def detect_language(text: str, default: str = "fr", min_length: int = 15) -> str:
    """Detect the language of the given text.

    Uses langdetect library with confidence thresholding. Returns default
    language for short texts, low-confidence detections, or on exceptions.

    Args:
        text: Input text to analyze
        default: Language code to return if detection fails or is uncertain (default: "fr")
        min_length: Minimum text length required for detection (default: 15)

    Returns:
        ISO 639-1 language code (e.g., "fr", "en")
    """
    # Handle empty or whitespace-only text
    if not text or not text.strip():
        return default

    # Short texts are unreliable for language detection
    if len(text.strip()) < min_length:
        return default

    try:
        # Get language probabilities
        langs = detect_langs(text)

        # Return top language only if confidence is high enough
        if langs and langs[0].prob >= 0.7:
            return str(langs[0].lang)

        # Low confidence → use default
        return default

    except LangDetectException:
        # Detection failed → use default
        return default


def get_reflecting_message(language: str, verbose: bool = False) -> str:
    """Return language-appropriate reflecting message for loading states.

    Args:
        language: ISO 639-1 language code (e.g., "fr", "en")
        verbose: If True, return extended message with timing note (for CLI).
                If False, return short message (for web UI).

    Returns:
        Localized reflecting message
    """
    # Default to French for unknown languages
    lang = language if language in ("en", "fr") else "fr"

    if verbose:
        return get_message(STATUS_REFLECTING_VERBOSE, lang)
    else:
        return get_message(STATUS_REFLECTING_SHORT, lang)
