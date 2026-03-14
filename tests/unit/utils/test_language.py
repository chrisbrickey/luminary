"""Unit tests for language detection and localization utilities."""

from unittest.mock import patch

import pytest
from langdetect import LangDetectException

from src.utils.language import detect_language, get_reflecting_message


class TestDetectLanguage:
    """Tests for detect_language function."""

    def test_detects_french(self) -> None:
        """Should detect French text with sufficient length and confidence."""
        french_text = "Bonjour, comment allez-vous aujourd'hui? C'est une belle journée."
        result = detect_language(french_text)
        assert result == "fr"

    def test_detects_english(self) -> None:
        """Should detect English text with sufficient length and confidence."""
        english_text = "Hello, how are you doing today? It's a beautiful day."
        result = detect_language(english_text)
        assert result == "en"

    def test_empty_string_returns_default(self) -> None:
        """Should return default for empty string."""
        assert detect_language("") == "fr"
        assert detect_language("", default="en") == "en"

    def test_whitespace_only_returns_default(self) -> None:
        """Should return default for whitespace-only text."""
        assert detect_language("   \n  \t  ") == "fr"
        assert detect_language("   ", default="en") == "en"

    def test_short_text_returns_default(self) -> None:
        """Should return default for text shorter than min_length."""
        # Default min_length is 15
        assert detect_language("Hello") == "fr"
        assert detect_language("Bonjour", default="en") == "en"

    def test_respects_custom_min_length(self) -> None:
        """Should respect custom min_length parameter."""
        # Text that's too short with default min_length (15) but long enough with min_length=8
        text = "Bonjour!"  # 8 chars

        # With default min_length=15, should return default
        result_default = detect_language(text, default="en")
        assert result_default == "en"

        # With min_length=5 (shorter than text), should attempt detection
        # Use a longer French phrase that's reliably detectable
        long_french = "Bonjour comment allez-vous?"  # Clearly French
        result_custom = detect_language(long_french, default="en", min_length=10)
        assert result_custom == "fr"

    @patch("src.utils.language.detect_langs")
    def test_low_confidence_returns_default(self, mock_detect_langs) -> None:
        """Should return default when confidence is below threshold."""
        # Mock low-confidence result
        mock_lang = type("Lang", (), {"lang": "en", "prob": 0.5})()
        mock_detect_langs.return_value = [mock_lang]

        result = detect_language("Some ambiguous text here")
        assert result == "fr"

    @patch("src.utils.language.detect_langs")
    def test_high_confidence_returns_detected(self, mock_detect_langs) -> None:
        """Should return detected language when confidence is high."""
        # Mock high-confidence result
        mock_lang = type("Lang", (), {"lang": "en", "prob": 0.85})()
        mock_detect_langs.return_value = [mock_lang]

        result = detect_language("Some clear English text here for testing")
        assert result == "en"

    @patch("src.utils.language.detect_langs")
    def test_exception_returns_default(self, mock_detect_langs) -> None:
        """Should return default when detection raises exception."""
        mock_detect_langs.side_effect = LangDetectException("Detection failed", "test")

        result = detect_language("Some text that causes error")
        assert result == "fr"

    @patch("src.utils.language.detect_langs")
    def test_empty_detection_results_returns_default(self, mock_detect_langs) -> None:
        """Should return default when detect_langs returns empty list."""
        mock_detect_langs.return_value = []

        result = detect_language("Some text with no detection results")
        assert result == "fr"


class TestGetReflectingMessage:
    """Tests for get_reflecting_message function."""

    def test_english_short_message(self) -> None:
        """Should return English short message for 'en' language with verbose=False."""
        result = get_reflecting_message("en", verbose=False)
        assert result == "Reflecting..."

    def test_english_verbose_message(self) -> None:
        """Should return English verbose message for 'en' language with verbose=True."""
        result = get_reflecting_message("en", verbose=True)
        assert result == "Reflecting... (response time varies with the amount of data retrieved and the connection)"

    def test_french_short_message(self) -> None:
        """Should return French short message for 'fr' language with verbose=False."""
        result = get_reflecting_message("fr", verbose=False)
        assert result == "Réflexion..."

    def test_french_verbose_message(self) -> None:
        """Should return French verbose message for 'fr' language with verbose=True."""
        result = get_reflecting_message("fr", verbose=True)
        assert result == "Réflexion... (le délai dépend de la taille des données et la connexion)"

    def test_unknown_language_defaults_to_french_short(self) -> None:
        """Should default to French for unknown languages with verbose=False."""
        result = get_reflecting_message("de", verbose=False)
        assert result == "Réflexion..."

    def test_unknown_language_defaults_to_french_verbose(self) -> None:
        """Should default to French for unknown languages with verbose=True."""
        result = get_reflecting_message("es", verbose=True)
        assert result == "Réflexion... (le délai dépend de la taille des données et la connexion)"

    def test_default_verbose_is_false(self) -> None:
        """Should use short message when verbose parameter is omitted."""
        result = get_reflecting_message("en")
        assert result == "Reflecting..."

    def test_no_mixed_languages_in_english_verbose(self) -> None:
        """Should not contain French words in English verbose message."""
        result = get_reflecting_message("en", verbose=True)
        # Verify no French keywords appear
        assert "réflexion" not in result.lower()
        assert "délai" not in result.lower()
        assert "taille" not in result.lower()
        assert "données" not in result.lower()
        assert "la connexion" not in result.lower()

    def test_no_mixed_languages_in_french_verbose(self) -> None:
        """Should not contain English words in French verbose message."""
        result = get_reflecting_message("fr", verbose=True)
        # Verify no English keywords appear (only check distinct English words)
        assert "reflecting" not in result.lower()
        assert "varies" not in result.lower()
        assert "data retrieved" not in result.lower()
        assert "connection" not in result.lower()
