"""Tests for src/eval/metrics/retrieval.py — retrieval_relevance metric."""

import pytest

from src.eval.metrics.retrieval import retrieval_relevance
from src.schemas import MetricResult

# -- Shared test constants --------------------------------------------------

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


class TestRetrievalRelevance:
    def test_perfect_retrieval(self) -> None:
        """All expected found, no extras → recall=1.0, precision=1.0, F1=1.0."""
        expected = [CHUNK_001, CHUNK_002, CHUNK_003]
        retrieved = [CHUNK_001, CHUNK_002, CHUNK_003]

        result = retrieval_relevance(expected, retrieved)

        assert isinstance(result, MetricResult)
        assert result.name == "retrieval_relevance"
        assert result.score == 1.0
        assert result.details["recall"] == 1.0
        assert result.details["precision"] == 1.0
        assert result.details["f1_score"] == 1.0
        assert set(result.details["found"]) == {CHUNK_001, CHUNK_002, CHUNK_003}
        assert result.details["missing"] == []
        assert result.details["irrelevant"] == []

    def test_high_recall_low_precision(self) -> None:
        """3 of 3 expected found, but 7 irrelevant → recall=1.0, precision=0.3, F1=0.46."""
        expected = [CHUNK_001, CHUNK_002, CHUNK_003]
        retrieved = [
            # 3 relevant, 7 irrelevant
            CHUNK_001, CHUNK_002, CHUNK_003,
            CHUNK_004, CHUNK_005, CHUNK_006, CHUNK_007,
            CHUNK_008, CHUNK_009, CHUNK_010
        ]

        result = retrieval_relevance(expected, retrieved)

        assert result.name == "retrieval_relevance"
        assert result.details["recall"] == 1.0
        assert result.details["precision"] == pytest.approx(3.0 / 10.0, abs=0.01)  # 0.3
        assert result.details["f1_score"] == pytest.approx(0.46, abs=0.01)
        assert result.score == pytest.approx(0.46, abs=0.01)
        assert set(result.details["found"]) == {CHUNK_001, CHUNK_002, CHUNK_003}
        assert result.details["missing"] == []
        assert set(result.details["irrelevant"]) == {
            CHUNK_004, CHUNK_005, CHUNK_006, CHUNK_007,
            CHUNK_008, CHUNK_009, CHUNK_010
        }

    def test_high_precision_low_recall(self) -> None:
        """1 of 3 expected found, no extras → recall=0.33, precision=1.0, F1=0.5."""
        expected = [CHUNK_001, CHUNK_002, CHUNK_003]
        retrieved = [CHUNK_001]

        result = retrieval_relevance(expected, retrieved)

        assert result.name == "retrieval_relevance"
        assert result.details["recall"] == pytest.approx(1.0 / 3.0, abs=0.01)  # 0.33
        assert result.details["precision"] == 1.0
        assert result.details["f1_score"] == pytest.approx(0.5, abs=0.01)
        assert result.score == pytest.approx(0.5, abs=0.01)
        assert result.details["found"] == [CHUNK_001]
        assert set(result.details["missing"]) == {CHUNK_002, CHUNK_003}
        assert result.details["irrelevant"] == []

    def test_balanced_partial(self) -> None:
        """2 of 3 expected found, 2 retrieved total → recall=0.67, precision=1.0, F1=0.8."""
        expected = [CHUNK_001, CHUNK_002, CHUNK_003]
        retrieved = [CHUNK_001, CHUNK_002]

        result = retrieval_relevance(expected, retrieved)

        assert result.name == "retrieval_relevance"
        assert result.details["recall"] == pytest.approx(2.0 / 3.0, abs=0.01)  # 0.67
        assert result.details["precision"] == 1.0
        assert result.details["f1_score"] == pytest.approx(0.8, abs=0.01)
        assert result.score == pytest.approx(0.8, abs=0.01)
        assert set(result.details["found"]) == {CHUNK_001, CHUNK_002}
        assert result.details["missing"] == [CHUNK_003]
        assert result.details["irrelevant"] == []

    def test_no_chunks_found_but_not_empty(self) -> None:
        """0 of 3 expected IDs in retrieved → recall=0.0, precision=0.0, F1=0.0."""
        expected = [CHUNK_001, CHUNK_002, CHUNK_003]
        retrieved = [CHUNK_004, CHUNK_005]

        result = retrieval_relevance(expected, retrieved)

        assert result.name == "retrieval_relevance"
        assert result.score == 0.0
        assert result.details["recall"] == 0.0
        assert result.details["precision"] == 0.0
        assert result.details["f1_score"] == 0.0
        assert result.details["found"] == []
        assert set(result.details["missing"]) == {CHUNK_001, CHUNK_002, CHUNK_003}
        assert set(result.details["irrelevant"]) == {CHUNK_004, CHUNK_005}

    def test_no_expected_ids(self) -> None:
        """Empty expected list → score 1.0 (vacuous truth)."""
        expected: list[str] = []
        retrieved = [CHUNK_001, CHUNK_002]

        result = retrieval_relevance(expected, retrieved)

        assert result.name == "retrieval_relevance"
        assert result.score == 1.0
        assert result.details["recall"] == 1.0
        assert result.details["precision"] == 1.0
        assert result.details["f1_score"] == 1.0
        assert result.details["found"] == []
        assert result.details["missing"] == []
        # When no expected chunks, all retrieved are "irrelevant" in the strict sense,
        # but the vacuous truth case treats this as perfect (nothing was needed)
        assert set(result.details["irrelevant"]) == {CHUNK_001, CHUNK_002}

    def test_empty_retrieval(self) -> None:
        """Empty retrieved list with non-empty expected → recall=0.0, precision=0.0, F1=0.0."""
        expected = [CHUNK_001, CHUNK_002, CHUNK_003]
        retrieved: list[str] = []

        result = retrieval_relevance(expected, retrieved)

        assert result.name == "retrieval_relevance"
        assert result.score == 0.0
        assert result.details["recall"] == 0.0
        assert result.details["precision"] == 1.0  # No false positives when nothing retrieved
        assert result.details["f1_score"] == 0.0
        assert result.details["found"] == []
        assert set(result.details["missing"]) == {CHUNK_001, CHUNK_002, CHUNK_003}
        assert result.details["irrelevant"] == []
