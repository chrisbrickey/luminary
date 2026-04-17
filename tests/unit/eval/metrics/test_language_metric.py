"""Tests for src/eval/metrics/language.py"""

from src.configs.common import ENGLISH_ISO_CODE, FRENCH_ISO_CODE
from src.eval.metrics.language import (
    language_content_compliance,
    language_metadata_compliance,
)
from src.schemas import MetricResult

# -- Shared test constants --------------------------------------------------

METADATA_METRIC_NAME = "language_metadata_compliance"
CONTENT_METRIC_NAME = "language_content_compliance"

# Sample texts for language content detection (20+ characters for reliable detection)
SAMPLE_FRENCH_TEXT = "Bonjour, comment allez-vous aujourd'hui? Je suis très heureux de vous voir."
SAMPLE_ENGLISH_TEXT = "Hello, how are you doing today? I am very happy to see you here."


class TestLanguageMetadataCompliance:
    """Tests for language_metadata_compliance metric."""

    def test_matching_language_metadata(self) -> None:
        """When expected and actual language match → score 1.0."""
        expected = FRENCH_ISO_CODE
        actual = FRENCH_ISO_CODE

        result = language_metadata_compliance(expected, actual)

        assert isinstance(result, MetricResult)
        assert result.name == METADATA_METRIC_NAME
        assert result.score == 1.0
        assert result.details["expected"] == FRENCH_ISO_CODE
        assert result.details["actual"] == FRENCH_ISO_CODE

    def test_mismatching_language_metadata(self) -> None:
        """When expected and actual language differ → score 0.0."""
        expected = ENGLISH_ISO_CODE
        actual = FRENCH_ISO_CODE

        result = language_metadata_compliance(expected, actual)

        assert isinstance(result, MetricResult)
        assert result.name == METADATA_METRIC_NAME
        assert result.score == 0.0
        assert result.details["expected"] == ENGLISH_ISO_CODE
        assert result.details["actual"] == FRENCH_ISO_CODE


class TestLanguageContentCompliance:
    """Tests for language_content_compliance metric."""

    def test_french_content_detected(self) -> None:
        """When French text is provided and expected=fr → score 1.0."""
        expected = FRENCH_ISO_CODE
        response_text = SAMPLE_FRENCH_TEXT

        result = language_content_compliance(expected, response_text)

        assert isinstance(result, MetricResult)
        assert result.name == CONTENT_METRIC_NAME
        assert result.score == 1.0
        assert result.details["expected"] == FRENCH_ISO_CODE
        assert result.details["detected"] == FRENCH_ISO_CODE

    def test_english_content_detected(self) -> None:
        """When English text is provided and expected=en → score 1.0."""
        expected = ENGLISH_ISO_CODE
        response_text = SAMPLE_ENGLISH_TEXT

        result = language_content_compliance(expected, response_text)

        assert isinstance(result, MetricResult)
        assert result.name == CONTENT_METRIC_NAME
        assert result.score == 1.0
        assert result.details["expected"] == ENGLISH_ISO_CODE
        assert result.details["detected"] == ENGLISH_ISO_CODE

    def test_french_content_mismatch(self) -> None:
        """When French text is provided but expected=en → score 0.0."""
        expected = ENGLISH_ISO_CODE
        response_text = SAMPLE_FRENCH_TEXT

        result = language_content_compliance(expected, response_text)

        assert isinstance(result, MetricResult)
        assert result.name == CONTENT_METRIC_NAME
        assert result.score == 0.0
        assert result.details["expected"] == ENGLISH_ISO_CODE
        assert result.details["detected"] == FRENCH_ISO_CODE

    def test_english_content_mismatch(self) -> None:
        """When English text is provided but expected=fr → score 0.0."""
        expected = FRENCH_ISO_CODE
        response_text = SAMPLE_ENGLISH_TEXT

        result = language_content_compliance(expected, response_text)

        assert isinstance(result, MetricResult)
        assert result.name == CONTENT_METRIC_NAME
        assert result.score == 0.0
        assert result.details["expected"] == FRENCH_ISO_CODE
        assert result.details["detected"] == ENGLISH_ISO_CODE

    def test_undetectable_content(self) -> None:
        """When text has no detectable language features → score 0.0 with error details."""
        expected = FRENCH_ISO_CODE
        response_text = ""  # Empty string cannot be detected

        result = language_content_compliance(expected, response_text)

        assert isinstance(result, MetricResult)
        assert result.name == CONTENT_METRIC_NAME
        assert result.score == 0.0
        assert result.details["expected"] == FRENCH_ISO_CODE
        assert "error" in result.details
        assert "detected" not in result.details  # No detection occurred
