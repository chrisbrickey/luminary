"""Unit tests for formatting utilities"""

import pytest

from src.configs.common import ENGLISH_ISO_CODE, FRENCH_ISO_CODE
from src.schemas import ChatResponse
from src.schemas.chat import SourceReference
from src.utils.formatting import format_page_ranges, format_sources, group_sources

# --- Shared test constants ---

TITLE_MULTI_PAGE = "Sample Work A"
TITLE_SOME_WORK = "Sample Work B"
TITLE_SOURCE_A = "Source A"
TITLE_SOURCE_B = "Source B"
URL_FALLBACK_TITLE = "https://example.com/doc3"

EXPECTED_NONE_EN = "**Sources:** none"
EXPECTED_NONE_FR = "**Sources :** aucune"


def _make_response(
    sources: list[SourceReference],
    titles: list[str] | None = None,
    language: str = ENGLISH_ISO_CODE,
) -> ChatResponse:
    """Build a ChatResponse with the given structured sources (and optional legacy titles)."""
    return ChatResponse(
        text="Answer text",
        retrieved_contexts=["Context"] * len(sources),
        retrieved_passage_ids=[f"id{i}" for i in range(len(sources))],
        retrieved_source_titles=titles if titles is not None else [],
        retrieved_sources=sources,
        language=language,
    )


@pytest.fixture
def response_no_sources() -> ChatResponse:
    """ChatResponse with no sources."""
    return _make_response(sources=[])


@pytest.fixture
def response_with_sources() -> ChatResponse:
    """ChatResponse with multiple distinct single-page sources, titles populated too."""
    return _make_response(
        sources=[
            SourceReference(title=TITLE_SOURCE_A, page_number=1),
            SourceReference(title=TITLE_SOURCE_B, page_number=2),
        ],
        titles=[f"{TITLE_SOURCE_A}, page 1", f"{TITLE_SOURCE_B}, page 2"],
    )


class TestGroupSources:
    """Tests for group_sources function."""

    def test_empty_sources(self) -> None:
        assert group_sources([]) == []

    def test_single_title_pages_deduped_and_sorted(self) -> None:
        sources = [
            SourceReference(title=TITLE_MULTI_PAGE, page_number=18),
            SourceReference(title=TITLE_MULTI_PAGE, page_number=13),
            SourceReference(title=TITLE_MULTI_PAGE, page_number=23),
            SourceReference(title=TITLE_MULTI_PAGE, page_number=1),
        ]
        assert group_sources(sources) == [(TITLE_MULTI_PAGE, [1, 13, 18, 23])]

    def test_duplicate_pages_deduped(self) -> None:
        sources = [
            SourceReference(title=TITLE_MULTI_PAGE, page_number=13),
            SourceReference(title=TITLE_MULTI_PAGE, page_number=13),
        ]
        assert group_sources(sources) == [(TITLE_MULTI_PAGE, [13])]

    def test_page_less_only_title(self) -> None:
        sources = [SourceReference(title=TITLE_SOME_WORK)]
        assert group_sources(sources) == [(TITLE_SOME_WORK, [])]

    def test_page_less_and_paged_entries_for_same_title_merge(self) -> None:
        sources = [
            SourceReference(title=TITLE_SOME_WORK, page_number=None),
            SourceReference(title=TITLE_SOME_WORK, page_number=5),
        ]
        assert group_sources(sources) == [(TITLE_SOME_WORK, [5])]

    def test_multiple_titles_ordered_by_first_appearance(self) -> None:
        sources = [
            SourceReference(title=TITLE_SOURCE_B, page_number=1),
            SourceReference(title=TITLE_SOURCE_A, page_number=2),
            SourceReference(title=TITLE_SOURCE_B, page_number=3),
        ]
        assert group_sources(sources) == [
            (TITLE_SOURCE_B, [1, 3]),
            (TITLE_SOURCE_A, [2]),
        ]


class TestFormatPageRanges:
    """Tests for format_page_ranges function. Input is assumed pre-sorted and deduped."""

    @pytest.mark.parametrize(
        "pages, expected",
        [
            ([], ""),
            ([18], "18"),
            ([1, 18], "1, 18"),
            ([11, 12], "11-12"),
            ([11, 12, 13], "11-13"),
            ([1, 11, 12, 13, 18], "1, 11-13, 18"),
            ([1, 2, 3, 7, 8, 20], "1-3, 7-8, 20"),
            ([0, 1], "0-1"),
        ],
    )
    def test_format_page_ranges(self, pages: list[int], expected: str) -> None:
        assert format_page_ranges(pages) == expected


class TestFormatSources:
    """Tests for format_sources function."""

    def test_no_sources_english(self, response_no_sources: ChatResponse) -> None:
        result = format_sources(response_no_sources, ENGLISH_ISO_CODE)
        assert result == EXPECTED_NONE_EN

    def test_no_sources_french(self, response_no_sources: ChatResponse) -> None:
        result = format_sources(response_no_sources, FRENCH_ISO_CODE)
        assert result == EXPECTED_NONE_FR

    def test_one_page_english_is_singular(self) -> None:
        response = _make_response(
            sources=[SourceReference(title=TITLE_MULTI_PAGE, page_number=18)]
        )
        result = format_sources(response, ENGLISH_ISO_CODE)
        expected = f"**Sources:**\n- {TITLE_MULTI_PAGE} (page: 18)"
        assert result == expected

    def test_pages_out_of_order_are_sorted_deduped_and_ranged_english(self) -> None:
        response = _make_response(
            sources=[
                SourceReference(title=TITLE_MULTI_PAGE, page_number=18),
                SourceReference(title=TITLE_MULTI_PAGE, page_number=13),
                SourceReference(title=TITLE_MULTI_PAGE, page_number=1),
                SourceReference(title=TITLE_MULTI_PAGE, page_number=12),
                SourceReference(title=TITLE_MULTI_PAGE, page_number=11),
            ]
        )
        result = format_sources(response, ENGLISH_ISO_CODE)
        expected = f"**Sources:**\n- {TITLE_MULTI_PAGE} (pages: 1, 11-13, 18)"
        assert result == expected

    def test_pages_out_of_order_are_sorted_deduped_and_ranged_french(self) -> None:
        response = _make_response(
            sources=[
                SourceReference(title=TITLE_MULTI_PAGE, page_number=18),
                SourceReference(title=TITLE_MULTI_PAGE, page_number=13),
                SourceReference(title=TITLE_MULTI_PAGE, page_number=1),
                SourceReference(title=TITLE_MULTI_PAGE, page_number=12),
                SourceReference(title=TITLE_MULTI_PAGE, page_number=11),
            ],
            language=FRENCH_ISO_CODE,
        )
        result = format_sources(response, FRENCH_ISO_CODE)
        expected = f"**Sources :**\n- {TITLE_MULTI_PAGE} (pages : 1, 11-13, 18)"
        assert result == expected

    def test_two_page_run_is_plural(self) -> None:
        response = _make_response(
            sources=[
                SourceReference(title=TITLE_SOME_WORK, page_number=11),
                SourceReference(title=TITLE_SOME_WORK, page_number=12),
            ]
        )
        result = format_sources(response, ENGLISH_ISO_CODE)
        expected = f"**Sources:**\n- {TITLE_SOME_WORK} (pages: 11-12)"
        assert result == expected

    def test_page_less_source_has_no_parenthetical(self) -> None:
        response = _make_response(sources=[SourceReference(title=TITLE_SOME_WORK)])
        result = format_sources(response, ENGLISH_ISO_CODE)
        expected = f"**Sources:**\n- {TITLE_SOME_WORK}"
        assert result == expected

    def test_url_fallback_title_renders_bare_url(self) -> None:
        response = _make_response(sources=[SourceReference(title=URL_FALLBACK_TITLE)])
        result = format_sources(response, ENGLISH_ISO_CODE)
        expected = f"**Sources:**\n- {URL_FALLBACK_TITLE}"
        assert result == expected

    def test_two_titles_appear_in_first_appearance_order(self) -> None:
        response = _make_response(
            sources=[
                SourceReference(title=TITLE_SOURCE_B, page_number=1),
                SourceReference(title=TITLE_SOURCE_A, page_number=2),
            ]
        )
        result = format_sources(response, ENGLISH_ISO_CODE)
        expected = f"**Sources:**\n- {TITLE_SOURCE_B} (page: 1)\n- {TITLE_SOURCE_A} (page: 2)"
        assert result == expected

    def test_uses_default_language(self, response_with_sources: ChatResponse) -> None:
        """Not passing language should use DEFAULT_RESPONSE_LANGUAGE (en)."""
        result = format_sources(response_with_sources)
        expected = f"**Sources:**\n- {TITLE_SOURCE_A} (page: 1)\n- {TITLE_SOURCE_B} (page: 2)"
        assert result == expected

    def test_footer_reads_only_structured_sources_not_legacy_titles(self) -> None:
        """Contract: an empty retrieved_sources renders 'none' even if legacy titles are present."""
        response = _make_response(sources=[], titles=[TITLE_SOURCE_A])
        result = format_sources(response, ENGLISH_ISO_CODE)
        assert result == EXPECTED_NONE_EN
