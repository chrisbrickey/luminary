"""Keyword coverage metric for evaluating topical adherence in RAG responses."""

import re
from typing import Any

from src.eval.metrics.base import MetricSpec, register_metric
from src.schemas import MetricResult


def keyword_coverage(expected_keywords: list[str], response_text: str) -> MetricResult:
    """Check the fraction of expected keywords that appear in the response text.
    This metric is a cheap, deterministic proxy for "the response stayed on topic.
    It does not inspect retrieved chunks.

    This metric uses case-insensitive whole-word matching with a bounded suffix
    tolerance (`\\b<keyword>\\w{0,4}\\b`) so that common language-specific
    inflections (e.g., plurals, "-ent", "-ant", "-ing") are accepted as matches
    without over-matching unrelated longer words.

    Args:
        expected_keywords: words expected in the response, in the language of the response
        response_text: response text to scan

    Returns:
        MetricResult with:
            - name: "keyword_coverage"
            - score: fraction of expected keywords found (0.0 to 1.0); empty `expected_keywords` returns 1.0
            - details: {
                "found": list[str],    # keywords matched in response_text
                "missing": list[str]   # keywords not matched
              }
    """
    if not expected_keywords:
        return MetricResult(
            name="keyword_coverage",
            score=1.0,
            details={
                "found": [],
                "missing": [],
            },
        )

    found: list[str] = []
    missing: list[str] = []

    for keyword in expected_keywords:
        pattern = r'\b' + re.escape(keyword) + r'\w{0,4}\b'
        if re.search(pattern, response_text, re.IGNORECASE):
            found.append(keyword)
        else:
            missing.append(keyword)

    score = len(found) / len(expected_keywords)

    return MetricResult(
        name="keyword_coverage",
        score=score,
        details={
            "found": found,
            "missing": missing,
        },
    )


def _keyword_coverage_wrapper(example: Any, response: Any) -> MetricResult:
    """Adapt keyword_coverage for the registry interface."""
    return keyword_coverage(
        expected_keywords=example.expected_keywords,
        response_text=response.text,
    )


register_metric(
    MetricSpec(
        name="keyword_coverage",
        compute=_keyword_coverage_wrapper,
        required_example_fields={"expected_keywords"},
        required_response_fields={"text"},
        languages=None,  # Applies to all languages; per-example language lives on the example.
        default_threshold=0.6,
    )
)
