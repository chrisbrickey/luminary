"""Utilities that support language detection and enforcement."""

from langdetect import LangDetectException, detect_langs  # type: ignore[import-untyped]

from src.configs.common import DEFAULT_RESPONSE_LANGUAGE

def detect_language(text: str, min_length: int = 15, confidence: float = 0.7) -> str:
    """Detect the language of the given text. In cases where the language
    cannot be detected to high confidence, returns DEFAULT_RESPONSE_LANGUAGE.

    Uses langdetect library with confidence thresholding. Returns default
    language for short texts, low-confidence detections, or on exceptions.

    Args:
        text: input text to analyze
        min_length: minimum text length required for detection
        confidence: probability threshold above which the detected language will be accepted

    Returns:
        ISO 639-1 language code (e.g., "fr", "en", "es", "sw")

    Examples:
        >>> detect_language("Bonjour, comment allez-vous?")
        'fr'
        >>> detect_language("Hola, ¿cómo estás?")
        'es'
        >>> detect_language("Buongiorno")  # Too short
        'en'  # Returns default
    """
    # Empty or too short text
    stripped = text.strip() if text else ""
    if not stripped or len(stripped) < min_length:
        return DEFAULT_RESPONSE_LANGUAGE

    try:
        langs = detect_langs(stripped)
    except LangDetectException:
        return DEFAULT_RESPONSE_LANGUAGE

    # Return top language if probability exceeds confidence threshold
    if langs and langs[0].prob >= confidence:
        return str(langs[0].lang)

    return DEFAULT_RESPONSE_LANGUAGE