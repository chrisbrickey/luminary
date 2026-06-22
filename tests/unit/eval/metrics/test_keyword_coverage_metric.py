"""Unit tests for keyword coverage metric"""

import pytest

from src.eval.metrics.keyword_coverage import keyword_coverage
from src.schemas import MetricResult
from src.schemas.eval import KeywordEntry

# -- Shared test constants --------------------------------------------------

METRIC_NAME = "keyword_coverage"

# ----- ENGLISH (plain ASCII) -----
ALPHA_EN = "widget"
BETA_EN = "sample"

# Used only by the inflection test; chosen because it has several common
# suffix forms (-es, -ed, -ing) that fit within the metric's 4-char bound.
INFLECTED_EN = "process"

PRIMARY_ABSENT_EN = "term"
SYNONYM_FOUND_EN = "word"
SYNONYM_ABSENT_EN = "label"
ALL_ABSENT_PRIMARY_EN = "widget"
ALL_ABSENT_SYNONYM_A_EN = "gadget"
ALL_ABSENT_SYNONYM_B_EN = "gizmo"

# English response text snippets
RESPONSE_ALL_KEYWORDS_EN = "The widget and sample are ready."
RESPONSE_PARTIAL_KEYWORDS_EN = "The widget is ready."
RESPONSE_NO_KEYWORDS_EN = "This sentence contains no relevant terms."
RESPONSE_CAPITALIZED_ALPHA_EN = "Widget is here."
RESPONSE_INFLECTED_EN = "Several processes ran. The data was processed."
RESPONSE_WITH_ONE_SYNONYM_EN = "The word was remarkable."
RESPONSE_NO_SYNONYMS_EN = "This sentence mentions nothing relevant."

# ----- FRENCH (accented words exercise unicode handling) -----
ALPHA_FR = "élément"
BETA_FR = "modèle"
GAMMA_FR = "système"

PRIMARY_ABSENT_FR = "élément"
SYNONYM_PRESENT_FR = "composant"

# French response text snippets
RESPONSE_ALL_KEYWORDS_FR = (
    "L'élément est listé. Le modèle est défini. Le système fonctionne."
)
RESPONSE_PARTIAL_KEYWORDS_FR = "L'élément est listé. Le modèle est défini."
RESPONSE_NO_KEYWORDS_FR = "Cette phrase ne contient aucun terme pertinent."
RESPONSE_CAPITALIZED_ALPHA_FR = "Élément est listé."
RESPONSE_INFLECTED_ALPHA_FR = "Les éléments sont nombreux."  # plural
RESPONSE_WITH_SYNONYM_FR = "Le composant est défini correctement."


def _primaries(found: list[dict[str, str]]) -> list[str]:
    """Return list of primary stems from a `details['found']` list-of-dicts."""
    return [entry["primary"] for entry in found]


class TestKeywordCoverageFrench:
    """Tests for keyword_coverage on French text (accented characters)."""

    def test_all_keywords_found_fr(self) -> None:
        """All expected keywords present → score 1.0."""
        expected = [
            KeywordEntry(primary=ALPHA_FR, synonyms=[]),
            KeywordEntry(primary=BETA_FR, synonyms=[]),
            KeywordEntry(primary=GAMMA_FR, synonyms=[]),
        ]

        result = keyword_coverage(expected, RESPONSE_ALL_KEYWORDS_FR)

        assert isinstance(result, MetricResult)
        assert result.name == METRIC_NAME
        assert result.score == 1.0
        assert set(_primaries(result.details["found"])) == {ALPHA_FR, BETA_FR, GAMMA_FR}
        assert result.details["missing"] == []

    def test_partial_keywords_found_fr(self) -> None:
        """2 of 3 keywords present → score ≈ 0.67."""
        expected = [
            KeywordEntry(primary=ALPHA_FR, synonyms=[]),
            KeywordEntry(primary=BETA_FR, synonyms=[]),
            KeywordEntry(primary=GAMMA_FR, synonyms=[]),
        ]

        result = keyword_coverage(expected, RESPONSE_PARTIAL_KEYWORDS_FR)

        assert isinstance(result, MetricResult)
        assert result.name == METRIC_NAME
        assert result.score == pytest.approx(2 / 3, abs=0.01)
        assert set(_primaries(result.details["found"])) == {ALPHA_FR, BETA_FR}
        assert result.details["missing"] == [GAMMA_FR]

    def test_no_keywords_found_fr(self) -> None:
        """0 of 3 keywords present → score 0.0."""
        expected = [
            KeywordEntry(primary=ALPHA_FR, synonyms=[]),
            KeywordEntry(primary=BETA_FR, synonyms=[]),
            KeywordEntry(primary=GAMMA_FR, synonyms=[]),
        ]

        result = keyword_coverage(expected, RESPONSE_NO_KEYWORDS_FR)

        assert isinstance(result, MetricResult)
        assert result.name == METRIC_NAME
        assert result.score == 0.0
        assert result.details["found"] == []
        assert set(result.details["missing"]) == {ALPHA_FR, BETA_FR, GAMMA_FR}

    def test_case_insensitive_fr(self) -> None:
        """Capitalized form in response matches lowercase keyword → score 1.0.

        The `matched` field captures the response's original casing.
        """
        expected = [KeywordEntry(primary=ALPHA_FR, synonyms=[])]

        result = keyword_coverage(expected, RESPONSE_CAPITALIZED_ALPHA_FR)

        assert isinstance(result, MetricResult)
        assert result.name == METRIC_NAME
        assert result.score == 1.0
        assert result.details["found"] == [{"primary": ALPHA_FR, "matched": "Élément"}]
        assert result.details["missing"] == []

    def test_inflection_match_fr(self) -> None:
        """Keyword matches inflected (plural) form via bounded-suffix tolerance → score 1.0.

        The `matched` field captures the inflected form actually present in the response.
        """
        expected = [KeywordEntry(primary=ALPHA_FR, synonyms=[])]

        result = keyword_coverage(expected, RESPONSE_INFLECTED_ALPHA_FR)

        assert isinstance(result, MetricResult)
        assert result.name == METRIC_NAME
        assert result.score == 1.0
        assert result.details["found"] == [{"primary": ALPHA_FR, "matched": "éléments"}]
        assert result.details["missing"] == []

    def test_no_expected_keywords_fr(self) -> None:
        """Empty expected_keywords list → score 1.0 (vacuous truth), found=[], missing=[]."""
        expected: list[KeywordEntry] = []

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
        expected = [
            KeywordEntry(primary=ALPHA_EN, synonyms=[]),
            KeywordEntry(primary=BETA_EN, synonyms=[]),
        ]

        result = keyword_coverage(expected, RESPONSE_ALL_KEYWORDS_EN)

        assert isinstance(result, MetricResult)
        assert result.name == METRIC_NAME
        assert result.score == 1.0
        assert set(_primaries(result.details["found"])) == {ALPHA_EN, BETA_EN}
        assert result.details["missing"] == []

    def test_partial_keywords_found_en(self) -> None:
        """1 of 2 keywords present → score 0.5."""
        expected = [
            KeywordEntry(primary=ALPHA_EN, synonyms=[]),
            KeywordEntry(primary=BETA_EN, synonyms=[]),
        ]

        result = keyword_coverage(expected, RESPONSE_PARTIAL_KEYWORDS_EN)

        assert isinstance(result, MetricResult)
        assert result.name == METRIC_NAME
        assert result.score == pytest.approx(0.5, abs=0.01)
        assert _primaries(result.details["found"]) == [ALPHA_EN]
        assert result.details["missing"] == [BETA_EN]

    def test_no_keywords_found_en(self) -> None:
        """0 of 2 keywords present → score 0.0."""
        expected = [
            KeywordEntry(primary=ALPHA_EN, synonyms=[]),
            KeywordEntry(primary=BETA_EN, synonyms=[]),
        ]

        result = keyword_coverage(expected, RESPONSE_NO_KEYWORDS_EN)

        assert isinstance(result, MetricResult)
        assert result.name == METRIC_NAME
        assert result.score == 0.0
        assert result.details["found"] == []
        assert set(result.details["missing"]) == {ALPHA_EN, BETA_EN}

    def test_case_insensitive_en(self) -> None:
        """Capitalized form in response matches lowercase keyword → score 1.0.

        The `matched` field captures the response's original casing.
        """
        expected = [KeywordEntry(primary=ALPHA_EN, synonyms=[])]

        result = keyword_coverage(expected, RESPONSE_CAPITALIZED_ALPHA_EN)

        assert isinstance(result, MetricResult)
        assert result.name == METRIC_NAME
        assert result.score == 1.0
        assert result.details["found"] == [{"primary": ALPHA_EN, "matched": "Widget"}]
        assert result.details["missing"] == []

    def test_inflection_match_en(self) -> None:
        """Keyword matches multiple inflected forms via bounded-suffix tolerance → score 1.0.

        The `matched` field captures the first inflected form encountered in the response.
        """
        expected = [KeywordEntry(primary=INFLECTED_EN, synonyms=[])]

        result = keyword_coverage(expected, RESPONSE_INFLECTED_EN)

        assert isinstance(result, MetricResult)
        assert result.name == METRIC_NAME
        assert result.score == 1.0
        # re.search returns the first match: "processes" precedes "processed" in the response.
        assert result.details["found"] == [{"primary": INFLECTED_EN, "matched": "processes"}]
        assert result.details["missing"] == []

    def test_no_expected_keywords_en(self) -> None:
        """Empty list → score 1.0, found=[], missing=[]."""
        expected: list[KeywordEntry] = []

        result = keyword_coverage(expected, RESPONSE_ALL_KEYWORDS_EN)

        assert isinstance(result, MetricResult)
        assert result.name == METRIC_NAME
        assert result.score == 1.0
        assert result.details["found"] == []
        assert result.details["missing"] == []


class TestKeywordCoverageSynonyms:
    """Tests for synonym-variant matching in keyword_coverage."""

    def test_synonym_match_counts_as_found_fr(self) -> None:
        """Primary absent but a synonym present → entry counted as found; score 1.0.

        The `matched` field captures the synonym's actual substring in the response.
        """
        # PRIMARY_ABSENT_FR ("élément") is NOT in RESPONSE_WITH_SYNONYM_FR;
        # SYNONYM_PRESENT_FR ("composant") IS present.
        expected = [
            KeywordEntry(primary=PRIMARY_ABSENT_FR, synonyms=[SYNONYM_PRESENT_FR]),
        ]

        result = keyword_coverage(expected, RESPONSE_WITH_SYNONYM_FR)

        assert isinstance(result, MetricResult)
        assert result.name == METRIC_NAME
        assert result.score == 1.0
        # `primary` is the canonical stem; `matched` is the synonym's appearance in the response.
        assert result.details["found"] == [
            {"primary": PRIMARY_ABSENT_FR, "matched": "composant"}
        ]
        assert result.details["missing"] == []

    def test_all_variants_missing_reports_primary_in_missing(self) -> None:
        """Primary and all synonyms absent → score 0.0; missing contains primary only."""
        expected = [
            KeywordEntry(
                primary=ALL_ABSENT_PRIMARY_EN,
                synonyms=[ALL_ABSENT_SYNONYM_A_EN, ALL_ABSENT_SYNONYM_B_EN],
            ),
        ]

        result = keyword_coverage(expected, RESPONSE_NO_SYNONYMS_EN)

        assert isinstance(result, MetricResult)
        assert result.score == 0.0
        assert result.details["found"] == []
        # missing contains only the primary, never the synonyms
        assert result.details["missing"] == [ALL_ABSENT_PRIMARY_EN]

    def test_multiple_synonyms_only_one_needs_to_match(self) -> None:
        """Multiple synonyms; response contains only one → entry counted as found."""
        # "word" is in RESPONSE_WITH_ONE_SYNONYM_EN; "label" is not.
        expected = [
            KeywordEntry(
                primary=PRIMARY_ABSENT_EN,
                synonyms=[SYNONYM_FOUND_EN, SYNONYM_ABSENT_EN],
            ),
        ]

        result = keyword_coverage(expected, RESPONSE_WITH_ONE_SYNONYM_EN)

        assert isinstance(result, MetricResult)
        assert result.score == 1.0
        assert result.details["found"] == [
            {"primary": PRIMARY_ABSENT_EN, "matched": "word"}
        ]
        assert result.details["missing"] == []

    def test_mixed_entries_primary_match_synonym_match_miss(self) -> None:
        """3 entries: primary matches, synonym matches, all absent → score 2/3."""
        # Entry 1: primary "widget" IS in response.
        # Entry 2: primary "term" absent; synonym "word" IS in response.
        # Entry 3: primary "sample" and synonym "nonexistent_term" are both absent.
        response = "The widget is ready. The word was notable."
        expected = [
            KeywordEntry(primary=ALPHA_EN, synonyms=[]),                           # primary matches
            KeywordEntry(primary=PRIMARY_ABSENT_EN, synonyms=[SYNONYM_FOUND_EN]),  # synonym matches
            KeywordEntry(primary=BETA_EN, synonyms=["nonexistent_term"]),          # synonym absent
        ]

        result = keyword_coverage(expected, response)

        assert isinstance(result, MetricResult)
        assert result.score == pytest.approx(2 / 3, abs=0.01)
        assert set(_primaries(result.details["found"])) == {ALPHA_EN, PRIMARY_ABSENT_EN}
        assert result.details["missing"] == [BETA_EN]

    def test_found_contains_primary_not_synonym_string(self) -> None:
        """details['found'][i]['primary'] is the canonical stem; the synonym appears only in `matched`."""
        synonym = SYNONYM_PRESENT_FR  # "composant"; the actual match
        primary = PRIMARY_ABSENT_FR   # "élément"; absent from response
        expected = [KeywordEntry(primary=primary, synonyms=[synonym])]

        result = keyword_coverage(expected, RESPONSE_WITH_SYNONYM_FR)

        # The found entry exposes the primary as the canonical label and the synonym as matched.
        assert result.details["found"] == [{"primary": primary, "matched": synonym}]
        assert synonym not in result.details["missing"]

    def test_missing_contains_primary_not_synonym_string(self) -> None:
        """details['missing'] lists primary strings, never synonym strings."""
        primary = ALL_ABSENT_PRIMARY_EN
        synonym_a = ALL_ABSENT_SYNONYM_A_EN
        synonym_b = ALL_ABSENT_SYNONYM_B_EN
        expected = [KeywordEntry(primary=primary, synonyms=[synonym_a, synonym_b])]

        result = keyword_coverage(expected, RESPONSE_NO_SYNONYMS_EN)

        assert primary in result.details["missing"]
        assert synonym_a not in result.details["missing"]
        assert synonym_b not in result.details["missing"]

    def test_empty_synonyms_primary_matches(self) -> None:
        """KeywordEntry with synonyms=[] still works: primary present → found."""
        expected = [KeywordEntry(primary=ALPHA_EN, synonyms=[])]

        result = keyword_coverage(expected, RESPONSE_ALL_KEYWORDS_EN)

        assert result.score == 1.0
        assert result.details["found"] == [{"primary": ALPHA_EN, "matched": "widget"}]
        assert result.details["missing"] == []

    def test_empty_synonyms_primary_absent(self) -> None:
        """KeywordEntry with synonyms=[] still works: primary absent → missing."""
        expected = [KeywordEntry(primary=ALPHA_EN, synonyms=[])]

        result = keyword_coverage(expected, RESPONSE_NO_KEYWORDS_EN)

        assert result.score == 0.0
        assert result.details["found"] == []
        assert result.details["missing"] == [ALPHA_EN]
