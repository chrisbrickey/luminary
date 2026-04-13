"""Citation metrics for evaluating source attribution in RAG responses."""

import re
from typing import Any

from src.eval.metrics.base import MetricSpec, register_metric
from src.schemas import MetricResult


def citation_accuracy(
    expected_source_titles: list[str],
    retrieved_source_titles: list[str],
) -> MetricResult:
    """Check if expected source title substrings are found in retrieved titles.

    Validates that the system correctly propagates source metadata (document titles)
    from retrieval through to the final response. Uses case-insensitive substring
    matching to allow flexibility for page numbers and formatting variations.

    Example:
        expected: ["Lettres philosophiques"]
        retrieved: ["Lettres philosophiques, page 12"]
        → Match found (substring, case-insensitive)

    Args:
        expected_source_titles: List of source titles that should be cited
        retrieved_source_titles: List of source titles that were actually cited

    Returns:
        MetricResult with:
            - name: "citation_accuracy"
            - score: fraction of expected titles found (0.0 to 1.0)
            - details: {
                "found": list[str],    # Expected titles that were cited
                "missing": list[str]   # Expected titles that were not cited
              }
    """
    # Handle empty expected list (vacuous truth: nothing expected = perfect)
    if not expected_source_titles:
        return MetricResult(
            name="citation_accuracy",
            score=1.0,
            details={
                "found": [],
                "missing": [],
            },
        )

    # Find which expected titles appear in retrieved titles (case-insensitive substring match)
    found_titles = []
    missing_titles = []

    for expected_title in expected_source_titles:
        # Check if this expected title appears as a substring in any retrieved title
        is_found = any(
            expected_title.lower() in retrieved_title.lower()
            for retrieved_title in retrieved_source_titles
        )

        if is_found:
            found_titles.append(expected_title)
        else:
            missing_titles.append(expected_title)

    # Calculate score: fraction of expected titles that were found
    score = len(found_titles) / len(expected_source_titles)

    return MetricResult(
        name="citation_accuracy",
        score=score,
        details={
            "found": found_titles,
            "missing": missing_titles,
        },
    )


def citation_to_retrieval_consistency(
    response_text: str,
    retrieved_chunk_sources: list[str],
) -> MetricResult:
    """Verify every citation in response text corresponds to an actually retrieved chunk.

    This metric validates runtime consistency between what the LLM cites and what the
    retrieval system actually returned. It catches two critical failure modes:
    1. Hallucinated citations - LLM invents sources that weren't retrieved
    2. Metadata propagation bugs - citations lost somewhere in the chain

    This complements citation_accuracy which tests against golden dataset expectations.

    Example:
        response_text: "Tolerance is key. [source: Lettres philosophiques]"
        retrieved_chunk_sources: ["Lettres philosophiques, page 12"]
        → Citation matches retrieved source (case-insensitive substring)

    Args:
        response_text: The LLM response text containing inline citations
        retrieved_chunk_sources: List of source titles from actually retrieved chunks

    Returns:
        MetricResult with:
            - name: "citation_to_retrieval_consistency"
            - score: fraction of citations that match retrieved sources (0.0 to 1.0)
            - details: {
                "matched": list[str],       # Citations found in retrieved sources
                "hallucinated": list[str]   # Citations NOT found in retrieved sources
              }
    """
    # Extract all citations from response text using regex
    # Pattern matches: [source: some title here]
    citation_pattern = r'\[source:\s*([^\]]+)\]'
    citations = re.findall(citation_pattern, response_text)

    # Handle empty citations (no citations = perfect consistency)
    if not citations:
        return MetricResult(
            name="citation_to_retrieval_consistency",
            score=1.0,
            details={
                "matched": [],
                "hallucinated": [],
            },
        )

    # Check each citation against retrieved sources (case-insensitive substring match)
    matched_citations = []
    hallucinated_citations = []

    for citation in citations:
        # Check if this citation appears as a substring in any retrieved source
        is_matched = any(
            citation.lower() in source.lower()
            for source in retrieved_chunk_sources
        )

        if is_matched:
            matched_citations.append(citation)
        else:
            hallucinated_citations.append(citation)

    # Calculate score: fraction of citations that matched
    score = len(matched_citations) / len(citations)

    return MetricResult(
        name="citation_to_retrieval_consistency",
        score=score,
        details={
            "matched": matched_citations,
            "hallucinated": hallucinated_citations,
        },
    )


def _citation_accuracy_wrapper(example: Any, response: Any) -> MetricResult:
    """Wrapper to adapt citation_accuracy for the registry interface.

    Args:
        example: GoldenExample with expected_source_titles attribute
        response: ChatResponse with retrieved_source_titles attribute

    Returns:
        MetricResult from citation_accuracy
    """
    return citation_accuracy(
        expected_source_titles=example.expected_source_titles,
        retrieved_source_titles=response.retrieved_source_titles,
    )


# Register the citation_accuracy metric in the global registry
register_metric(
    MetricSpec(
        name="citation_accuracy",
        compute=_citation_accuracy_wrapper,
        required_example_fields={"expected_source_titles"},
        required_response_fields={"retrieved_source_titles"},
        languages=None,  # Applies to all languages
        # not specifying default_threshold here will fall back to FALLBACK_THRESHOLD
    )
)


def _citation_to_retrieval_consistency_wrapper(example: Any, response: Any) -> MetricResult:
    """Wrapper to adapt citation_to_retrieval_consistency for the registry interface.

    Args:
        example: GoldenExample (not used for this metric)
        response: ChatResponse with text and retrieved_source_titles attributes

    Returns:
        MetricResult from citation_to_retrieval_consistency
    """
    return citation_to_retrieval_consistency(
        response_text=response.text,
        retrieved_chunk_sources=response.retrieved_source_titles,
    )


# Register the citation_to_retrieval_consistency metric in the global registry
register_metric(
    MetricSpec(
        name="citation_to_retrieval_consistency",
        compute=_citation_to_retrieval_consistency_wrapper,
        required_example_fields=set(),  # No example fields needed
        required_response_fields={"text", "retrieved_source_titles"},
        languages=None,  # Applies to all languages
        # not specifying default_threshold here will fall back to FALLBACK_THRESHOLD
    )
)
