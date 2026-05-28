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
RETRIEVED_WORK_A_P1 = f"{SOURCE_WORK_A}, page 1"
RETRIEVED_WORK_A_P4 = f"{SOURCE_WORK_A}, page 4"
RETRIEVED_WORK_A_P5 = f"{SOURCE_WORK_A}, page 5"
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

# Multi-page summary citations (LLM aggregates inline citations at end of response)
CITATION_WORK_A_PAGES_1_4_5_EN = f"[source: {SOURCE_WORK_A}, pages 1, 4, and 5]"
CITATION_WORK_A_PAGES_1_4_5_FR = f"[source: {SOURCE_WORK_A}, pages 1, 4 et 5]"
CITATION_WORK_A_PAGES_1_AND_4 = f"[source: {SOURCE_WORK_A}, pages 1 and 4]"

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

    def test_multi_page_citation_all_pages_match(self) -> None:
        """Multi-page summary citation expands into single-page equivalents → all match → 1.0."""
        response_text = f"Several claims. {CITATION_WORK_A_PAGES_1_4_5_EN}"
        retrieved_sources = [RETRIEVED_WORK_A_P1, RETRIEVED_WORK_A_P4, RETRIEVED_WORK_A_P5]

        result = citation_to_retrieval_consistency(response_text, retrieved_sources)

        assert result.name == CONSISTENCY_METRIC
        assert result.score == 1.0
        assert len(result.details["matched"]) == 3
        assert result.details["hallucinated"] == []

    def test_multi_page_citation_partial_match(self) -> None:
        """Multi-page summary citation with one page missing from retrieved → score 2/3."""
        response_text = f"Several claims. {CITATION_WORK_A_PAGES_1_4_5_EN}"
        # Only pages 1 and 4 retrieved; page 5 was not retrieved
        retrieved_sources = [RETRIEVED_WORK_A_P1, RETRIEVED_WORK_A_P4]

        result = citation_to_retrieval_consistency(response_text, retrieved_sources)

        assert result.name == CONSISTENCY_METRIC
        assert result.score == pytest.approx(2 / 3, abs=0.01)
        assert len(result.details["matched"]) == 2
        assert len(result.details["hallucinated"]) == 1

    def test_multi_page_citation_french_separator(self) -> None:
        """Multi-page summary citation using French 'et' separator → all match → 1.0."""
        response_text = f"Plusieurs revendications. {CITATION_WORK_A_PAGES_1_4_5_FR}"
        retrieved_sources = [RETRIEVED_WORK_A_P1, RETRIEVED_WORK_A_P4, RETRIEVED_WORK_A_P5]

        result = citation_to_retrieval_consistency(response_text, retrieved_sources)

        assert result.name == CONSISTENCY_METRIC
        assert result.score == 1.0
        assert len(result.details["matched"]) == 3
        assert result.details["hallucinated"] == []

    def test_multi_page_citation_two_pages(self) -> None:
        """Two-page summary citation 'pages X and Y' → both expand and match → 1.0."""
        response_text = f"Two claims. {CITATION_WORK_A_PAGES_1_AND_4}"
        retrieved_sources = [RETRIEVED_WORK_A_P1, RETRIEVED_WORK_A_P4]

        result = citation_to_retrieval_consistency(response_text, retrieved_sources)

        assert result.name == CONSISTENCY_METRIC
        assert result.score == 1.0
        assert len(result.details["matched"]) == 2
        assert result.details["hallucinated"] == []

    def test_mixed_single_and_multi_page_citations(self) -> None:
        """Mix of inline single-page and summary multi-page citations counted individually."""
        # 1 single-page inline citation + 1 three-page summary citation = 4 individual requirements
        single_page_citation = f"[source: {RETRIEVED_WORK_A_P1}]"
        response_text = (
            f"Inline claim. {single_page_citation} "
            f"Summary citation. {CITATION_WORK_A_PAGES_1_4_5_EN}"
        )
        retrieved_sources = [RETRIEVED_WORK_A_P1, RETRIEVED_WORK_A_P4, RETRIEVED_WORK_A_P5]

        result = citation_to_retrieval_consistency(response_text, retrieved_sources)

        assert result.name == CONSISTENCY_METRIC
        assert result.score == 1.0
        assert len(result.details["matched"]) == 4
        assert result.details["hallucinated"] == []
