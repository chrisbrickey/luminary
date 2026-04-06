"""Unit tests for language detection utilities."""

from unittest.mock import patch

from langdetect import LangDetectException

from src.configs.common import (
    DEFAULT_RESPONSE_LANGUAGE,
    ENGLISH_ISO_CODE,
    FRENCH_ISO_CODE,
    GERMAN_ISO_CODE,
    ITALIAN_ISO_CODE,
    SWAHILI_ISO_CODE,
)
from src.utils.language import detect_language


class TestDetectLanguage:
    """Tests for detect_language function."""

    def test_detects_french(self) -> None:
        """Should detect French text with sufficient length and confidence."""
        french_text = "Bonjour, comment allez-vous aujourd'hui? C'est une belle journée."
        result = detect_language(french_text)
        assert result == FRENCH_ISO_CODE

    def test_detects_english(self) -> None:
        """Should detect English text with sufficient length and confidence."""
        english_text = "Hello, how are you doing today? It's a beautiful day out there."
        result = detect_language(english_text)
        assert result == ENGLISH_ISO_CODE

    def test_detects_italian(self) -> None:
        """Should detect Italian text even though Italian is not supported for localization.

        This demonstrates that language detection works for any language,
        even if it is not explicitly supported with localized UI strings.
        """
        italian_text = "Come stai oggi? È una giornata bellissima e soleggiata."
        result = detect_language(italian_text)
        assert result == ITALIAN_ISO_CODE

    def test_detects_swahili(self) -> None:
        """Should detect Swahili (uncommon language) to demonstrate broad language support.

        This test documents that detection works for less common languages,
        even though they will never have localized UI strings.
        """
        swahili_text = "Habari yako leo? Ni siku nzuri sana ya kufanya kazi."
        result = detect_language(swahili_text)
        assert result == SWAHILI_ISO_CODE

    @patch("src.utils.language.detect_langs")
    def test_detection_failure_returns_default(self, mock_detect_langs) -> None:
        """Should return default when detect_langs returns empty list (detection failure)."""
        mock_detect_langs.return_value = []

        result = detect_language("Some text that fails detection")
        assert result == DEFAULT_RESPONSE_LANGUAGE

    def test_empty_string_returns_default(self) -> None:
        """Should return default for empty string."""
        assert detect_language("") == DEFAULT_RESPONSE_LANGUAGE

    def test_whitespace_only_returns_default(self) -> None:
        """Should return default for whitespace-only text."""
        assert detect_language("   \n  \t  ") == DEFAULT_RESPONSE_LANGUAGE

    def test_short_text_returns_default(self) -> None:
        """Should return default for text shorter than min_length."""
        # Default min_length is 15
        assert detect_language("Hello") == DEFAULT_RESPONSE_LANGUAGE

    def test_custom_min_length(self) -> None:
        """Should respect custom min_length parameter."""
        # Short text that's borderline for detection
        text = "Bonjour comment"  # 15 chars (exactly at default threshold)

        # With default min_length=15, this is at the threshold, so detection will run
        # But with slightly higher threshold (16), should return default immediately
        result_above_threshold = detect_language(text, min_length=16)
        assert result_above_threshold == DEFAULT_RESPONSE_LANGUAGE

        # With lower threshold (10), should attempt detection on this text
        # Use longer French text to ensure reliable detection
        longer_french = "Bonjour comment allez-vous?"  # 26 chars, clearly French
        result_custom = detect_language(longer_french, min_length=10)
        assert result_custom == FRENCH_ISO_CODE

    @patch("src.utils.language.detect_langs")
    def test_custom_confidence_threshold(self, mock_detect_langs) -> None:
        """Should respect custom confidence parameter.

        A detection probability that meets the default threshold (0.7)
        should be rejected when a higher threshold (0.9) is specified.
        """
        # Mock a confidence level that's acceptable with default (0.75 >= 0.7)
        # but too low for a stricter threshold (0.75 < 0.9)
        actual_detected_language = GERMAN_ISO_CODE
        mock_lang = type("Lang", (), {"lang": actual_detected_language, "prob": 0.75})()
        mock_detect_langs.return_value = [mock_lang]

        # With default confidence=0.7, should return detected language
        result_default = detect_language("....doesn't matter because library response is mocked....")
        assert result_default == actual_detected_language

        # With stricter confidence=0.9, should return default language
        result_strict = detect_language("....doesn't matter because library response is mocked....", confidence=0.9)
        assert result_strict == DEFAULT_RESPONSE_LANGUAGE

    @patch("src.utils.language.detect_langs")
    def test_low_confidence_returns_default(self, mock_detect_langs) -> None:
        """Should return default when confidence is below 0.7 threshold."""
        # Mock low-confidence result (0.5 < 0.7)
        mock_lang = type("Lang", (), {"lang": ENGLISH_ISO_CODE, "prob": 0.5})()
        mock_detect_langs.return_value = [mock_lang]

        result = detect_language("Some ambiguous text here")
        assert result == DEFAULT_RESPONSE_LANGUAGE

    @patch("src.utils.language.detect_langs")
    def test_high_confidence_returns_detected(self, mock_detect_langs) -> None:
        """Should return detected language when confidence is >= 0.7 threshold."""
        # Mock high-confidence result (0.85 >= 0.7)
        mock_lang = type("Lang", (), {"lang": GERMAN_ISO_CODE, "prob": 0.85})()
        mock_detect_langs.return_value = [mock_lang]

        result = detect_language("Some clear German text here for testing purposes")
        assert result == GERMAN_ISO_CODE

    @patch("src.utils.language.detect_langs")
    def test_exception_returns_default(self, mock_detect_langs) -> None:
        """Should return default when detection raises LangDetectException."""
        mock_detect_langs.side_effect = LangDetectException("Detection failed", "test")

        result = detect_language("Some text that causes error")
        assert result == DEFAULT_RESPONSE_LANGUAGE
