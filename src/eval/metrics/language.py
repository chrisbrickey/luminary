"""Metrics that evaluate response language correctness."""

from typing import Any

import langdetect  # type: ignore[import-untyped]
from langdetect.lang_detect_exception import LangDetectException  # type: ignore[import-untyped]

from src.eval.metrics.base import MetricSpec, register_metric
from src.schemas import MetricResult


def language_metadata_compliance(expected_language: str, response_language: str) -> MetricResult:
    """Validate that the ChatResponse language metadata matches the expected language.

    This metric validates that the system correctly sets the language metadata field
    in ChatResponse to match the expected language for the author's response.
    Uses exact string matching for ISO 639-1 language codes (e.g., "fr", "en").

    Args:
        expected_language: Expected ISO 639-1 language code (e.g., "fr", "en")
        response_language: Actual language code from ChatResponse.language

    Returns:
        MetricResult with:
            - name: "language_metadata_compliance"
            - score: 1.0 if match, 0.0 if mismatch
            - details: {"expected": expected_language, "actual": response_language}
    """
    # Exact match comparison
    matches = expected_language == response_language
    score = 1.0 if matches else 0.0

    return MetricResult(
        name="language_metadata_compliance",
        score=score,
        details={
            "expected": expected_language,
            "actual": response_language,
        },
    )


def language_content_compliance(expected_language: str, response_text: str) -> MetricResult:
    """Validate that the actual response text content is in the expected language.

    This metric validates that the actual text content of the response is in the
    expected language by using language detection on the response text.
    Uses langdetect library to detect the language from the response text
    and compares it to the expected language.

    If language detection fails (e.g., empty text, no detectable features),
    returns score 0.0 with error details.

    Args:
        expected_language: Expected ISO 639-1 language code (e.g., "fr", "en")
        response_text: The actual response text to analyze

    Returns:
        MetricResult with:
            - name: "language_content_compliance"
            - score: 1.0 if detected == expected, 0.0 if mismatch or detection fails
            - details: {"expected": expected, "detected": detected_language}
                      or {"expected": expected, "error": error_message} if detection fails
    """
    try:
        # Detect the language from the response text
        detected_language = langdetect.detect(response_text)

        # Compare detected language to expected language
        matches = detected_language == expected_language
        score = 1.0 if matches else 0.0

        return MetricResult(
            name="language_content_compliance",
            score=score,
            details={
                "expected": expected_language,
                "detected": detected_language,
            },
        )
    except LangDetectException as e:
        # Language detection failed (empty text, no features, etc.)
        # Return score 0.0 with error details
        return MetricResult(
            name="language_content_compliance",
            score=0.0,
            details={
                "expected": expected_language,
                "error": str(e),
            },
        )


def _language_metadata_compliance_wrapper(example: Any, response: Any) -> MetricResult:
    """Wrapper to adapt language_metadata_compliance for the registry interface.

    Args:
        example: GoldenExample with language attribute
        response: ChatResponse with language attribute

    Returns:
        MetricResult from language_metadata_compliance
    """
    return language_metadata_compliance(
        expected_language=example.language,
        response_language=response.language,
    )


def _language_content_compliance_wrapper(example: Any, response: Any) -> MetricResult:
    """Wrapper to adapt language_content_compliance for the registry interface.

    Args:
        example: GoldenExample with language attribute
        response: ChatResponse with text attribute

    Returns:
        MetricResult from language_content_compliance
    """
    return language_content_compliance(
        expected_language=example.language,
        response_text=response.text,
    )


# Register the language_metadata_compliance metric in the global registry
register_metric(
    MetricSpec(
        name="language_metadata_compliance",
        compute=_language_metadata_compliance_wrapper,
        required_example_fields={"language"},
        required_response_fields={"language"},
        languages=None,  # Applies to all languages
        # not specifying default_threshold here will fall back to FALLBACK_THRESHOLD
    )
)

# Register the language_content_compliance metric in the global registry
register_metric(
    MetricSpec(
        name="language_content_compliance",
        compute=_language_content_compliance_wrapper,
        required_example_fields={"language"},
        required_response_fields={"text"},
        languages=None,  # Applies to all languages
        # not specifying default_threshold here will fall back to FALLBACK_THRESHOLD
    )
)
