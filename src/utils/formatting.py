"""Formatting utilities for chat responses."""

from src.configs.common import DEFAULT_RESPONSE_LANGUAGE
from src.i18n import get_message
from src.i18n.keys import SOURCES_LABEL, SOURCES_NONE
from src.schemas import ChatResponse


def deduplicate_sources(response: ChatResponse) -> list[str]:
    """Deduplicate source titles while preserving order.

    Args:
        response: ChatResponse with retrieved_source_titles

    Returns:
        List of deduplicated source titles in order of first appearance
    """
    seen = set()
    deduplicated = []
    for title in response.retrieved_source_titles:
        if title not in seen:
            seen.add(title)
            deduplicated.append(title)
    return deduplicated


def format_sources(
    response: ChatResponse,
    language: str = DEFAULT_RESPONSE_LANGUAGE,
) -> str:
    """Format source citations as a markdown bullet list.

    Uses a unified markdown format that works for both CLI and web UI.
    In Streamlit, the label renders as bold and items as a bullet list.
    In CLI, the markdown characters display as-is but remain readable.

    Args:
        response: ChatResponse with retrieved_source_titles
        language: ISO 639-1 language code (defaults to DEFAULT_RESPONSE_LANGUAGE)

    Returns:
        Formatted sources string

    Example:
        format_sources(response, "en")
        # Returns:
        # "**Sources:**
        # - Source A
        # - Source B"
        #
        # Or if no sources:
        # "**Sources:** none"
    """
    sources = deduplicate_sources(response)
    label = get_message(SOURCES_LABEL, language)

    if not sources:
        none_text = get_message(SOURCES_NONE, language)
        return f"{label} {none_text}"

    items = "\n".join(f"- {source}" for source in sources)
    return f"{label}\n{items}"
