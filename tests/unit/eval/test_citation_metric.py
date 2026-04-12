"""Tests for src/eval/metrics/citation.py — citation metrics."""

import pytest

from src.eval.metrics.citation import citation_accuracy, citation_to_retrieval_consistency
from src.schemas import MetricResult

# -- Shared test constants --------------------------------------------------

ACCURACY_METRIC = "citation_accuracy"
CONSISTENCY_METRIC = "citation_to_retrieval_consistency"

# Source titles (generic fake data)
SOURCE_WORK_A = "test work alpha"
SOURCE_WORK_B = "sample document beta"
SOURCE_WORK_C = "generic treatise gamma"

# Retrieved titles (with page numbers and variations)
RETRIEVED_WORK_A_P12 = f"{SOURCE_WORK_A}, page 12"
RETRIEVED_WORK_A_CAPITALIZED_P45 = f"{SOURCE_WORK_A.capitalize()}, page 45"  # Different capitalization
RETRIEVED_WORK_B_CH3 = "sample document beta, chapter 3"
RETRIEVED_WORK_C_P8 = "generic treatise gamma, page 8"
RETRIEVED_OTHER = "unrelated work delta"

# Citation strings for consistency tests (as they appear in response text)
CITATION_WORK_A = f"[source: {SOURCE_WORK_A}]"
CITATION_WORK_B = f"[source: {SOURCE_WORK_B}]"
CITATION_WORK_C = f"[source: {SOURCE_WORK_C}]"
CITATION_OTHER = "[source: unrelated work delta]"
CITATION_HALLUCINATED = "[source: nonexistent work epsilon]"

# Response text samples with citations
RESPONSE_WITH_CITATION_A = f"This is a claim. {CITATION_WORK_A}"
RESPONSE_WITH_CITATIONS_AB = f"First claim. {CITATION_WORK_A} Second claim. {CITATION_WORK_B}"
RESPONSE_WITH_MIXED_CITATIONS = f"Valid claim. {CITATION_WORK_A} Another claim. {CITATION_HALLUCINATED}"
RESPONSE_NO_CITATIONS = "This response has no citations at all."


class TestCitationAccuracy:
    """Tests for citation_accuracy metric."""

    def test_all_titles_found(self) -> None:
        """All expected substrings present → score 1.0."""
        expected = [SOURCE_WORK_A, SOURCE_WORK_B]
        retrieved = [RETRIEVED_WORK_A_P12, RETRIEVED_WORK_B_CH3]

        result = citation_accuracy(expected, retrieved)

        assert isinstance(result, MetricResult)
        assert result.name == ACCURACY_METRIC
        assert result.score == 1.0
        assert set(result.details["found"]) == {SOURCE_WORK_A, SOURCE_WORK_B}
        assert result.details["missing"] == []

    def test_partial_titles_found(self) -> None:
        """1 of 2 expected substrings present → score 0.5."""
        expected = [SOURCE_WORK_A, SOURCE_WORK_B]
        retrieved = [RETRIEVED_WORK_A_P12, RETRIEVED_OTHER]

        result = citation_accuracy(expected, retrieved)

        assert result.name == ACCURACY_METRIC
        assert result.score == pytest.approx(0.5, abs=0.01)
        assert result.details["found"] == [SOURCE_WORK_A]
        assert result.details["missing"] == [SOURCE_WORK_B]

    def test_case_insensitive_matching(self) -> None:
        """Lowercase query matches capitalized retrieved title → score 1.0."""

        # Expected uses lowercase, retrieved has varied capitalization
        expected = [SOURCE_WORK_A.lower()]
        retrieved = [RETRIEVED_WORK_A_CAPITALIZED_P45]  # "Test Work Alpha, page 45"
        result = citation_accuracy(expected, retrieved)

        assert result.name == ACCURACY_METRIC
        assert result.score == 1.0
        assert result.details["found"] == [SOURCE_WORK_A.lower()]
        assert result.details["missing"] == []

    def test_no_expected_titles(self) -> None:
        """Empty expected list → score 1.0 (vacuous truth)."""
        expected: list[str] = []
        retrieved = [RETRIEVED_WORK_A_P12, RETRIEVED_WORK_B_CH3]

        result = citation_accuracy(expected, retrieved)

        assert result.name == ACCURACY_METRIC
        assert result.score == 1.0
        assert result.details["found"] == []
        assert result.details["missing"] == []

    def test_no_retrieved_titles(self) -> None:
        """Empty retrieved list with non-empty expected → score 0.0."""
        expected = [SOURCE_WORK_A, SOURCE_WORK_B]
        retrieved: list[str] = []

        result = citation_accuracy(expected, retrieved)

        assert result.name == ACCURACY_METRIC
        assert result.score == 0.0
        assert result.details["found"] == []
        assert set(result.details["missing"]) == {SOURCE_WORK_A, SOURCE_WORK_B}


class TestCitationToRetrievalConsistency:
    """Tests for citation_to_retrieval_consistency metric."""

    def test_all_citations_match(self) -> None:
        """All citations found in retrieved sources → score 1.0."""
        response_text = RESPONSE_WITH_CITATIONS_AB  # Contains citations to WORK_A and WORK_B
        retrieved_sources = [RETRIEVED_WORK_A_P12, RETRIEVED_WORK_B_CH3]

        result = citation_to_retrieval_consistency(response_text, retrieved_sources)

        assert isinstance(result, MetricResult)
        assert result.name == CONSISTENCY_METRIC
        assert result.score == 1.0
        assert len(result.details["matched"]) == 2
        assert result.details["hallucinated"] == []

    def test_partial_citations_match(self) -> None:
        """2 of 3 citations match retrieved sources → score 0.67."""
        # Response contains 3 citations: WORK_A, WORK_B, and HALLUCINATED
        response_text = f"{RESPONSE_WITH_CITATIONS_AB} Third claim. {CITATION_HALLUCINATED}"
        retrieved_sources = [RETRIEVED_WORK_A_P12, RETRIEVED_WORK_B_CH3]

        result = citation_to_retrieval_consistency(response_text, retrieved_sources)

        assert result.name == CONSISTENCY_METRIC
        assert result.score == pytest.approx(0.67, abs=0.01)
        assert len(result.details["matched"]) == 2
        assert len(result.details["hallucinated"]) == 1

    def test_no_citations_match(self) -> None:
        """0 of 2 citations in retrieved sources → score 0.0 (hallucinated)."""
        # Response cites WORK_C and HALLUCINATED, but only WORK_A and WORK_B were retrieved
        response_text = f"First claim. {CITATION_WORK_C} Second claim. {CITATION_HALLUCINATED}"
        retrieved_sources = [RETRIEVED_WORK_A_P12, RETRIEVED_WORK_B_CH3]

        result = citation_to_retrieval_consistency(response_text, retrieved_sources)

        assert result.name == CONSISTENCY_METRIC
        assert result.score == 0.0
        assert result.details["matched"] == []
        assert len(result.details["hallucinated"]) == 2

    def test_no_citations_in_response(self) -> None:
        """Empty response citations → score 1.0 (vacuous truth)."""
        response_text = RESPONSE_NO_CITATIONS
        retrieved_sources = [RETRIEVED_WORK_A_P12, RETRIEVED_WORK_B_CH3]

        result = citation_to_retrieval_consistency(response_text, retrieved_sources)

        assert result.name == CONSISTENCY_METRIC
        assert result.score == 1.0
        assert result.details["matched"] == []
        assert result.details["hallucinated"] == []

    def test_case_insensitive_matching(self) -> None:
        """Lowercase citation matches capitalized source title → score 1.0."""
        # Citation uses lowercase, retrieved has capitalized title
        citation_lowercase = f"[source: {SOURCE_WORK_A.lower()}]"
        response_text = f"This is a claim. {citation_lowercase}"
        retrieved_sources = [RETRIEVED_WORK_A_CAPITALIZED_P45]  # "Test Work Alpha, page 45"

        result = citation_to_retrieval_consistency(response_text, retrieved_sources)

        assert result.name == CONSISTENCY_METRIC
        assert result.score == 1.0
        assert len(result.details["matched"]) == 1
        assert result.details["hallucinated"] == []
