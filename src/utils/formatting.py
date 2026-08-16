"""Formatting utilities for chat responses."""

from collections.abc import Sequence

from src.configs.common import DEFAULT_RESPONSE_LANGUAGE
from src.i18n import get_message
from src.i18n.keys import SOURCES_LABEL, SOURCES_NONE, SOURCES_PAGE_PLURAL, SOURCES_PAGE_SINGULAR
from src.schemas import ChatResponse
from src.schemas.chat import SourceReference


def group_sources(sources: Sequence[SourceReference]) -> list[tuple[str, list[int]]]:
    """Collapse references to one entry per title, pages deduped and ascending.

    Titles are ordered by first appearance. A title seen only without a page
    still gets an entry with an empty page list.
    """
    grouped: dict[str, list[int]] = {}
    for source in sources:
        pages = grouped.setdefault(source.title, [])
        if source.page_number is not None and source.page_number not in pages:
            pages.append(source.page_number)
    return [(title, sorted(pages)) for title, pages in grouped.items()]


def format_page_ranges(pages: Sequence[int]) -> str:
    """Render ascending page numbers, collapsing consecutive runs: [1,11,12,13] -> '1, 11-13'.

    Assumes the input is already sorted and deduped.
    """
    parts: list[str] = []
    run_start: int | None = None
    run_end: int | None = None
    for page in pages:
        if run_end is not None and page == run_end + 1:  # extend the open run
            run_end = page
            parts[-1] = f"{run_start}-{run_end}"
        else:  # start a new run
            run_start = run_end = page
            parts.append(str(page))
    return ", ".join(parts)


def _format_source_entry(title: str, pages: list[int], language: str) -> str:
    """Render one bullet's text: bare title, or title plus a localized page parenthetical."""
    if not pages:
        return title
    key = SOURCES_PAGE_SINGULAR if len(pages) == 1 else SOURCES_PAGE_PLURAL
    return f"{title} {get_message(key, language, pages=format_page_ranges(pages))}"


def format_sources(
    response: ChatResponse,
    language: str = DEFAULT_RESPONSE_LANGUAGE,
) -> str:
    """Format source citations as a markdown bullet list, one bullet per title.

    Uses a unified markdown format that works for both CLI and web UI.
    In Streamlit, the label renders as bold and items as a bullet list.
    In CLI, the markdown characters display as-is but remain readable.

    Args:
        response: ChatResponse with retrieved_sources
        language: ISO 639-1 language code (defaults to DEFAULT_RESPONSE_LANGUAGE)

    Returns:
        Formatted sources string

    Example:
        format_sources(response, "en")
        # Returns:
        # "**Sources:**
        # - Lettres Philosophiques 1734 (pages: 1, 11-13, 18)"
        #
        # Or if no sources:
        # "**Sources:** none"
    """
    grouped = group_sources(response.retrieved_sources)
    label = get_message(SOURCES_LABEL, language)

    if not grouped:
        none_text = get_message(SOURCES_NONE, language)
        return f"{label} {none_text}"

    items = "\n".join(f"- {_format_source_entry(title, pages, language)}" for title, pages in grouped)
    return f"{label}\n{items}"
