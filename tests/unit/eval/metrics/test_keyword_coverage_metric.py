"""Unit tests for keyword coverage metric"""

import pytest

from src.eval.metrics.keyword_coverage import keyword_coverage
from src.schemas import MetricResult

# -- Shared test constants --------------------------------------------------

METRIC_NAME = "keyword_coverage"

# French keywords; Accented words exercise unicode handling
KEYWORD_ALPHA_FR = "élément"
KEYWORD_BETA_FR = "modèle"
KEYWORD_GAMMA_FR = "système"

# English keywords; plain ASCII
KEYWORD_ALPHA_EN = "widget"
KEYWORD_BETA_EN = "sample"

# Used only by the inflection test; chosen because it has several common
# suffix forms (-es, -ed, -ing) that fit within the metric's 4-char bound.
KEYWORD_INFLECTED_EN = "process"

# French response text snippets
RESPONSE_ALL_KEYWORDS_FR = (
    "L'élément est listé. Le modèle est défini. Le système fonctionne."
)
RESPONSE_PARTIAL_KEYWORDS_FR = "L'élément est listé. Le modèle est défini."
RESPONSE_NO_KEYWORDS_FR = "Cette phrase ne contient aucun terme pertinent."
RESPONSE_CAPITALIZED_ALPHA_FR = "Élément est listé."
# Exercises the bounded-suffix regex: "éléments" = "élément" + "s" (plural).
RESPONSE_INFLECTED_ALPHA_FR = "Les éléments sont nombreux."

# English response text snippets.
RESPONSE_ALL_KEYWORDS_EN = "The widget and sample are ready."
RESPONSE_PARTIAL_KEYWORDS_EN = "The widget is ready."
RESPONSE_NO_KEYWORDS_EN = "This sentence contains no relevant terms."
RESPONSE_CAPITALIZED_ALPHA_EN = "Widget is here."
# Exercises the bounded-suffix regex with two inflections:
# "processes" = "process" + "es"; "processed" = "process" + "ed".
RESPONSE_INFLECTED_EN = "Several processes ran. The data was processed."


class TestKeywordCoverageFrench:
    """Tests for keyword_coverage on French text (accented characters)."""

    def test_all_keywords_found_fr(self) -> None:
        """All expected keywords present → score 1.0."""
        expected = [KEYWORD_ALPHA_FR, KEYWORD_BETA_FR, KEYWORD_GAMMA_FR]

        result = keyword_coverage(expected, RESPONSE_ALL_KEYWORDS_FR)

        assert isinstance(result, MetricResult)
        assert result.name == METRIC_NAME
        assert result.score == 1.0
        assert set(result.details["found"]) == {KEYWORD_ALPHA_FR, KEYWORD_BETA_FR, KEYWORD_GAMMA_FR}
        assert result.details["missing"] == []

    def test_partial_keywords_found_fr(self) -> None:
        """2 of 3 keywords present → score ≈ 0.67."""
        expected = [KEYWORD_ALPHA_FR, KEYWORD_BETA_FR, KEYWORD_GAMMA_FR]

        result = keyword_coverage(expected, RESPONSE_PARTIAL_KEYWORDS_FR)

        assert isinstance(result, MetricResult)
        assert result.name == METRIC_NAME
        assert result.score == pytest.approx(2 / 3, abs=0.01)
        assert set(result.details["found"]) == {KEYWORD_ALPHA_FR, KEYWORD_BETA_FR}
        assert result.details["missing"] == [KEYWORD_GAMMA_FR]

    def test_no_keywords_found_fr(self) -> None:
        """0 of 3 keywords present → score 0.0."""
        expected = [KEYWORD_ALPHA_FR, KEYWORD_BETA_FR, KEYWORD_GAMMA_FR]

        result = keyword_coverage(expected, RESPONSE_NO_KEYWORDS_FR)

        assert isinstance(result, MetricResult)
        assert result.name == METRIC_NAME
        assert result.score == 0.0
        assert result.details["found"] == []
        assert set(result.details["missing"]) == {KEYWORD_ALPHA_FR, KEYWORD_BETA_FR, KEYWORD_GAMMA_FR}

    def test_case_insensitive_fr(self) -> None:
        """Capitalized form in response matches lowercase keyword → score 1.0."""
        expected = [KEYWORD_ALPHA_FR]

        result = keyword_coverage(expected, RESPONSE_CAPITALIZED_ALPHA_FR)

        assert isinstance(result, MetricResult)
        assert result.name == METRIC_NAME
        assert result.score == 1.0
        assert result.details["found"] == [KEYWORD_ALPHA_FR]
        assert result.details["missing"] == []

    def test_inflection_match_fr(self) -> None:
        """Keyword matches inflected (plural) form via bounded-suffix tolerance → score 1.0."""
        expected = [KEYWORD_ALPHA_FR]

        result = keyword_coverage(expected, RESPONSE_INFLECTED_ALPHA_FR)

        assert isinstance(result, MetricResult)
        assert result.name == METRIC_NAME
        assert result.score == 1.0
        assert result.details["found"] == [KEYWORD_ALPHA_FR]
        assert result.details["missing"] == []

    def test_no_expected_keywords_fr(self) -> None:
        """Empty expected_keywords list → score 1.0 (vacuous truth), found=[], missing=[]."""
        expected: list[str] = []

        result = keyword_coverage(expected, RESPONSE_ALL_KEYWORDS_FR)

        assert isinstance(result, MetricResult)
        assert result.name == METRIC_NAME
        assert result.score == 1.0
        assert result.details["found"] == []
        assert result.details["missing"] == []


class TestKeywordCoverageEnglish:
    """Tests for keyword_coverage on English text (plain ASCII)."""

    def test_all_keywords_found_en(self) -> None:
        """All expected keywords present → score 1.0."""
        expected = [KEYWORD_ALPHA_EN, KEYWORD_BETA_EN]

        result = keyword_coverage(expected, RESPONSE_ALL_KEYWORDS_EN)

        assert isinstance(result, MetricResult)
        assert result.name == METRIC_NAME
        assert result.score == 1.0
        assert set(result.details["found"]) == {KEYWORD_ALPHA_EN, KEYWORD_BETA_EN}
        assert result.details["missing"] == []

    def test_partial_keywords_found_en(self) -> None:
        """1 of 2 keywords present → score 0.5."""
        expected = [KEYWORD_ALPHA_EN, KEYWORD_BETA_EN]

        result = keyword_coverage(expected, RESPONSE_PARTIAL_KEYWORDS_EN)

        assert isinstance(result, MetricResult)
        assert result.name == METRIC_NAME
        assert result.score == pytest.approx(0.5, abs=0.01)
        assert result.details["found"] == [KEYWORD_ALPHA_EN]
        assert result.details["missing"] == [KEYWORD_BETA_EN]

    def test_no_keywords_found_en(self) -> None:
        """0 of 2 keywords present → score 0.0."""
        expected = [KEYWORD_ALPHA_EN, KEYWORD_BETA_EN]

        result = keyword_coverage(expected, RESPONSE_NO_KEYWORDS_EN)

        assert isinstance(result, MetricResult)
        assert result.name == METRIC_NAME
        assert result.score == 0.0
        assert result.details["found"] == []
        assert set(result.details["missing"]) == {KEYWORD_ALPHA_EN, KEYWORD_BETA_EN}

    def test_case_insensitive_en(self) -> None:
        """Capitalized form in response matches lowercase keyword → score 1.0."""
        expected = [KEYWORD_ALPHA_EN]

        result = keyword_coverage(expected, RESPONSE_CAPITALIZED_ALPHA_EN)

        assert isinstance(result, MetricResult)
        assert result.name == METRIC_NAME
        assert result.score == 1.0
        assert result.details["found"] == [KEYWORD_ALPHA_EN]
        assert result.details["missing"] == []

    def test_inflection_match_en(self) -> None:
        """Keyword matches multiple inflected forms via bounded-suffix tolerance → score 1.0."""
        expected = [KEYWORD_INFLECTED_EN]

        result = keyword_coverage(expected, RESPONSE_INFLECTED_EN)

        assert isinstance(result, MetricResult)
        assert result.name == METRIC_NAME
        assert result.score == 1.0
        assert result.details["found"] == [KEYWORD_INFLECTED_EN]
        assert result.details["missing"] == []

    def test_no_expected_keywords_en(self) -> None:
        """Empty list → score 1.0, found=[], missing=[]."""
        expected: list[str] = []

        result = keyword_coverage(expected, RESPONSE_ALL_KEYWORDS_EN)

        assert isinstance(result, MetricResult)
        assert result.name == METRIC_NAME
        assert result.score == 1.0
        assert result.details["found"] == []
        assert result.details["missing"] == []
