"""Keyword coverage metric for evaluating topical adherence in RAG responses."""

import re
from typing import Any

from src.eval.metrics.base import MetricSpec, register_metric
from src.schemas import MetricResult
from src.schemas.eval import KeywordEntry


def _first_match(token: str, response_text: str) -> str | None:
    """Return the actual matched substring from response_text, or None if token is absent.

    Applies suffix character limit (via regex) so that the match on a stem tolerates
    short inflections (plurals, "-ent", "-ant", "-ing") without over-matching unrelated longer words.
    The returned string is the inflected form as it appeared in the response (preserving case), to aid debugging.
    """
    pattern = r'\b' + re.escape(token) + r'\w{0,4}\b'
    m = re.search(pattern, response_text, re.IGNORECASE)
    return m.group(0) if m else None


def keyword_coverage(expected_keywords: list[KeywordEntry], response_text: str) -> MetricResult:
    """Check the fraction of expected keyword entries that appear in the response text.

    This metric is a cheap, deterministic proxy for "the response stayed on topic."
    It does not inspect retrieved chunks.

    Each KeywordEntry contains a primary keyword stem and a list of synonym stems.
    An entry is counted as "found" if the primary OR any synonym appear in the response
    within the suffix character allowance. The list of synonyms tolerate variation in
    word choice by the model while giving confidence that the response remains topic.

    Args:
        expected_keywords: KeywordEntry instances expected in the response
        response_text: response text to scan

    Returns:
        MetricResult that includes the entire word in the response that matched with a keyword within "found".
        Only primary keyword stems are included as "missing" to see at a glance which concepts were missed.

            - name: "keyword_coverage"
            - score: fraction of expected entries found (0.0 to 1.0); empty list returns 1.0
            - details: {
                "found": list[{"primary": str, "matched": str}],
                "missing": list[str]
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

    found: list[dict[str, str]] = []
    missing: list[str] = []

    for entry in expected_keywords:
        matched_substring: str | None = None
        for token in [entry.primary, *entry.synonyms]:
            matched_substring = _first_match(token, response_text)
            if matched_substring is not None:
                break

        if matched_substring is not None:
            found.append({"primary": entry.primary, "matched": matched_substring})
        else:
            missing.append(entry.primary)

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
