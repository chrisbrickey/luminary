"""Unit tests for retrieval_relevance metric"""

import pytest

from src.eval.metrics.retrieval import retrieval_relevance
from src.schemas import MetricResult

# -- Shared test constants --------------------------------------------------

METRIC_NAME = "retrieval_relevance"

# Expected nDCG values are derived by hand from the rank positions in each test.
# A change to the production formula will fail the test.
# Four decimal places is beyond the gap between any two rankings under test.
DETAIL_TOLERANCE = 1e-4

CHUNK_001 = "chunk_001"
CHUNK_002 = "chunk_002"
CHUNK_003 = "chunk_003"
CHUNK_004 = "chunk_004"
CHUNK_005 = "chunk_005"
CHUNK_006 = "chunk_006"
CHUNK_007 = "chunk_007"
CHUNK_008 = "chunk_008"
CHUNK_009 = "chunk_009"
CHUNK_010 = "chunk_010"

EXPECTED_THREE = [CHUNK_001, CHUNK_002, CHUNK_003]
SEVEN_IRRELEVANT = [
    CHUNK_004, CHUNK_005, CHUNK_006, CHUNK_007,
    CHUNK_008, CHUNK_009, CHUNK_010,
]


def _assert_details(result: MetricResult, **expected: object) -> None:
    """Compare the whole details dict at once so a renamed or added key fails."""
    assert result.details == pytest.approx(expected, abs=DETAIL_TOLERANCE)


class TestRetrievalRelevance:
    def test_perfect_retrieval_in_ideal_order(self) -> None:
        """All expected chunks retrieved in the same order they were expected."""
        expected = EXPECTED_THREE
        retrieved = list(EXPECTED_THREE)

        result = retrieval_relevance(expected, retrieved)

        assert isinstance(result, MetricResult)
        assert result.name == METRIC_NAME
        assert result.score == 1.0
        # Hits at ranks 1, 2, 3 are exactly the ideal ranking.
        _assert_details(
            result,
            recall_at_k=1.0,
            ndcg_at_k=1.0,
            precision_at_n=1.0,
            k=3,
            n=3,
            f1_score=1.0,
            found=EXPECTED_THREE,
            missing=[],
            irrelevant=[],
        )

    def test_full_recall_ranked_at_bottom_of_top_k(self) -> None:
        """All expected chunks found but ranked last: recall is perfect, ranking is not."""
        expected = EXPECTED_THREE
        retrieved = SEVEN_IRRELEVANT + EXPECTED_THREE

        result = retrieval_relevance(expected, retrieved)

        assert result.name == METRIC_NAME
        assert result.score == 1.0
        # Hits sit at ranks 8, 9, 10 against a 3-hit ideal, so nDCG falls well short of 1.0.
        # Legacy set-F1 is retained only for comparison, and shows the old ceiling.
        _assert_details(
            result,
            recall_at_k=1.0,
            ndcg_at_k=0.42496,
            precision_at_n=0.0,
            k=10,
            n=3,
            f1_score=0.4615,
            found=EXPECTED_THREE,
            missing=[],
            irrelevant=SEVEN_IRRELEVANT,
        )

    def test_full_recall_ranked_at_top_of_top_k(self) -> None:
        """All expected chunks found and ranked first: ranking is rewarded over the bottom case."""
        expected = EXPECTED_THREE
        retrieved = EXPECTED_THREE + SEVEN_IRRELEVANT

        result = retrieval_relevance(expected, retrieved)

        assert result.name == METRIC_NAME
        assert result.score == 1.0
        # Hits at ranks 1, 2, 3 match the ideal ranking, so nDCG is perfect even
        # though seven irrelevant chunks trail them. F1 scoring cannot
        # distinguish this case from the bottom-ranked case.
        _assert_details(
            result,
            recall_at_k=1.0,
            ndcg_at_k=1.0,
            precision_at_n=1.0,
            k=10,
            n=3,
            f1_score=0.4615,
            found=EXPECTED_THREE,
            missing=[],
            irrelevant=SEVEN_IRRELEVANT,
        )

    def test_ranking_position_changes_ndcg_but_not_recall(self) -> None:
        """Same chunks, better ranking: nDCG rewards it where recall and set-F1 cannot."""
        top = retrieval_relevance(EXPECTED_THREE, EXPECTED_THREE + SEVEN_IRRELEVANT)
        bottom = retrieval_relevance(EXPECTED_THREE, SEVEN_IRRELEVANT + EXPECTED_THREE)

        assert top.details["ndcg_at_k"] > bottom.details["ndcg_at_k"]
        assert top.score == bottom.score
        assert top.details["f1_score"] == bottom.details["f1_score"]

    def test_partial_retrieval_reflects_miss_in_ranking(self) -> None:
        """2 of 3 expected chunks found; a miss lowers precision_at_n and ndcg_at_k proportionally."""
        expected = EXPECTED_THREE
        retrieved = [CHUNK_001, CHUNK_004, CHUNK_002]

        result = retrieval_relevance(expected, retrieved)

        assert result.name == METRIC_NAME
        assert result.score == pytest.approx(2 / 3)
        # Hits at ranks 1 and 3 with a miss at rank 2, against a 3-hit ideal.
        _assert_details(
            result,
            recall_at_k=2 / 3,
            ndcg_at_k=0.7039,
            precision_at_n=2 / 3,
            k=3,
            n=3,
            f1_score=2 / 3,
            found=[CHUNK_001, CHUNK_002],
            missing=[CHUNK_003],
            irrelevant=[CHUNK_004],
        )

    def test_empty_retrieval_with_expected_chunks(self) -> None:
        """No chunks retrieved at all: every component score is 0.0, not the old precision=1.0."""
        expected = EXPECTED_THREE
        retrieved: list[str] = []

        result = retrieval_relevance(expected, retrieved)

        assert result.name == METRIC_NAME
        assert result.score == 0.0
        _assert_details(
            result,
            recall_at_k=0.0,
            ndcg_at_k=0.0,
            precision_at_n=0.0,
            k=0,
            n=3,
            f1_score=0.0,
            found=[],
            missing=EXPECTED_THREE,
            irrelevant=[],
        )

    def test_empty_expected_is_vacuous_truth(self) -> None:
        """Nothing was expected, so retrieval is trivially perfect (unchanged legacy behavior)."""
        expected: list[str] = []
        retrieved = [CHUNK_001, CHUNK_002]

        result = retrieval_relevance(expected, retrieved)

        assert result.name == METRIC_NAME
        assert result.score == 1.0
        _assert_details(
            result,
            recall_at_k=1.0,
            ndcg_at_k=1.0,
            precision_at_n=1.0,
            k=2,
            n=0,
            f1_score=1.0,
            found=[],
            missing=[],
            irrelevant=sorted(retrieved),
        )

    def test_zero_overlap_with_non_empty_retrieval(self) -> None:
        """Retrieved chunks share nothing with expected chunks: every component score is 0.0."""
        expected = EXPECTED_THREE
        retrieved = [CHUNK_004, CHUNK_005]

        result = retrieval_relevance(expected, retrieved)

        assert result.name == METRIC_NAME
        assert result.score == 0.0
        _assert_details(
            result,
            recall_at_k=0.0,
            ndcg_at_k=0.0,
            precision_at_n=0.0,
            k=2,
            n=3,
            f1_score=0.0,
            found=[],
            missing=EXPECTED_THREE,
            irrelevant=sorted(retrieved),
        )
