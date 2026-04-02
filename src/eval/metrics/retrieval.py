"""Retrieval metrics for evaluating RAG system performance."""

from typing import Any

from src.eval.metrics.base import MetricSpec, register_metric
from src.schemas import MetricResult


def retrieval_relevance(
    expected_chunk_ids: list[str],
    retrieved_chunk_ids: list[str],
) -> MetricResult:
    """Calculate recall, precision, and F1 score for chunk retrieval quality.

    Measures retrieval quality using standard information retrieval metrics:
    - Recall: What fraction of expected chunks were found?
    - Precision: What fraction of retrieved chunks are relevant?
    - F1: Harmonic mean balancing both metrics

    High F1 means the system finds relevant chunks (recall) without
    retrieving too much noise (precision). This prevents wasting context
    window tokens and reduces LLM confusion from irrelevant information.

    Args:
        expected_chunk_ids: List of chunk IDs that should be retrieved
        retrieved_chunk_ids: List of chunk IDs that were actually retrieved

    Returns:
        MetricResult with:
            - name: "retrieval_relevance"
            - score: F1 score (0.0 to 1.0)
            - details: {
                "recall": float,
                "precision": float,
                "f1_score": float,
                "found": list[str],      # Expected chunks that were retrieved
                "missing": list[str],    # Expected chunks that were not retrieved
                "irrelevant": list[str]  # Retrieved chunks that were not expected
              }
    """
    # Handle empty expected list (vacuous truth: nothing expected = perfect)
    if not expected_chunk_ids:
        # When nothing is expected, any retrieval is technically "irrelevant"
        # but we treat this as a perfect case (score 1.0)
        return MetricResult(
            name="retrieval_relevance",
            score=1.0,
            details={
                "recall": 1.0,
                "precision": 1.0,
                "f1_score": 1.0,
                "found": [],
                "missing": [],
                "irrelevant": sorted(retrieved_chunk_ids),
            },
        )

    # Calculate set intersections and differences
    expected_set = set(expected_chunk_ids)
    retrieved_set = set(retrieved_chunk_ids)
    found_set = expected_set & retrieved_set
    missing_set = expected_set - retrieved_set
    irrelevant_set = retrieved_set - expected_set

    # Calculate recall: fraction of expected chunks that were found
    recall = len(found_set) / len(expected_set)

    # Calculate precision: fraction of retrieved chunks that are relevant
    # If nothing retrieved, precision is 1.0 (no false positives)
    if not retrieved_chunk_ids:
        precision = 1.0
    else:
        precision = len(found_set) / len(retrieved_set)

    # Calculate F1 score: harmonic mean of precision and recall
    # If both are 0, F1 is 0
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    return MetricResult(
        name="retrieval_relevance",
        score=f1_score,
        details={
            "recall": recall,
            "precision": precision,
            "f1_score": f1_score,
            "found": sorted(found_set),
            "missing": sorted(missing_set),
            "irrelevant": sorted(irrelevant_set),
        },
    )


def _retrieval_relevance_wrapper(example: Any, response: Any) -> MetricResult:
    """Wrapper to adapt retrieval_relevance for the registry interface.

    Args:
        example: GoldenExample with expected_chunk_ids attribute
        response: ChatResponse with retrieved_passage_ids attribute

    Returns:
        MetricResult from retrieval_relevance
    """
    return retrieval_relevance(
        expected_chunk_ids=example.expected_chunk_ids,
        retrieved_chunk_ids=response.retrieved_passage_ids,
    )


# Register the metric in the global registry
register_metric(
    MetricSpec(
        name="retrieval_relevance",
        compute=_retrieval_relevance_wrapper,
        required_example_fields={"expected_chunk_ids"},
        required_response_fields={"retrieved_passage_ids"},
        languages=None,  # Applies to all languages
    )
)
