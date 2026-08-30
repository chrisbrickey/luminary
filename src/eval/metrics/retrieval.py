"""Retrieval metrics for evaluating RAG system performance."""

import math
from typing import Any

from src.eval.metrics.base import MetricSpec, register_metric
from src.schemas import MetricResult


def _discounted_cumulative_gain(relevances: list[int]) -> float:
    """Sum binary relevances discounted by their 1-indexed rank."""
    return sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevances, start=1))


def retrieval_relevance(
    expected_chunk_ids: list[str],
    retrieved_chunk_ids: list[str],
) -> MetricResult:
    """Score chunk retrieval quality with recall@K, nDCG@K, and precision@N.

    Recall answers the question "did the retriever find the right passages".
    The reported score is recall@K: the fraction of expected chunks the retriever found anywhere in its top-K.

    Unlike F1 scoring, recall@K is not capped by a mismatch between K retrieved chunks
    and a smaller number of expected chunks in a golden dataset.

    Other components are calculated and persisted in eval artifacts for audit.
    But these are separate from the metric scoring.
    - nDCG@K rewards landing the expected chunks near the top, which is what
      matters to the downstream LLM reading the context window.
    - precision@N (N = expected chunk count) keeps a precision signal without
      penalizing the retriever for returning the K chunks requested by the chat chain

    Args:
        expected_chunk_ids: List of chunk IDs that should be retrieved
        retrieved_chunk_ids: List of chunk IDs that were actually retrieved in rank order (most relevant first)

    Returns:
        MetricResult with:
            - name: "retrieval_relevance"
            - score: recall@K (0.0 to 1.0)
            - details: {
                "recall_at_k": float,
                "ndcg_at_k": float,
                "precision_at_n": float,
                "k": int,               # Number of retrieved chunks
                "n": int,               # Number of expected chunks
                "f1_score": float,      # Legacy set-F1, kept for comparison
                                        # against pre-P.1 eval artifacts
                "found": list[str],     # Expected chunks that were retrieved
                "missing": list[str],   # Expected chunks that were not retrieved
                "irrelevant": list[str] # Retrieved chunks that were not expected
              }
    """
    expected_set = set(expected_chunk_ids)
    retrieved_set = set(retrieved_chunk_ids)
    k = len(retrieved_chunk_ids)
    n = len(expected_set)

    # Nothing was expected, so retrieval is trivially perfect.
    if not expected_set:
        return MetricResult(
            name="retrieval_relevance",
            score=1.0,
            details={
                "recall_at_k": 1.0,
                "ndcg_at_k": 1.0,
                "precision_at_n": 1.0,
                "k": k,
                "n": 0,
                "f1_score": 1.0,
                "found": [],
                "missing": [],
                "irrelevant": sorted(retrieved_chunk_ids),
            },
        )

    found_set = expected_set & retrieved_set
    missing_set = expected_set - retrieved_set
    irrelevant_set = retrieved_set - expected_set

    # recall@K: fraction of expected chunks found anywhere in the top-K
    recall_at_k = len(found_set) / n

    # nDCG@K: relevant chunks earn more when they rank higher
    relevances = [1 if chunk_id in expected_set else 0 for chunk_id in retrieved_chunk_ids]
    ideal_relevances = [1] * min(n, k)
    ideal_dcg = _discounted_cumulative_gain(ideal_relevances)
    ndcg_at_k = _discounted_cumulative_gain(relevances) / ideal_dcg if ideal_dcg > 0 else 0.0

    # precision@N: how much of the top-N slice (sized to the expected count) is relevant
    precision_at_n = len(expected_set & set(retrieved_chunk_ids[:n])) / n

    # Legacy plain set-F1; retained in details to compare with older eval artifacts
    set_precision = len(found_set) / len(retrieved_set) if retrieved_set else 0.0
    if set_precision + recall_at_k > 0:
        f1_score = 2 * (set_precision * recall_at_k) / (set_precision + recall_at_k)
    else:
        f1_score = 0.0

    return MetricResult(
        name="retrieval_relevance",
        score=recall_at_k,
        details={
            "recall_at_k": recall_at_k,
            "ndcg_at_k": ndcg_at_k,
            "precision_at_n": precision_at_n,
            "k": k,
            "n": n,
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
        # not specifying default_threshold here will fall back to FALLBACK_THRESHOLD
    )
)
