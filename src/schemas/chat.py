"""Pydantic schemas for chat functionality."""

from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator


class SourceReference(BaseModel):
    """A retrieved source: document title with optional page number."""

    title: str
    page_number: int | None = None

    @field_validator("page_number", mode="before")
    @classmethod
    def _coerce_page_number(cls, value: Any) -> int | None:
        """Chunk metadata is untyped, so drop non-int pages rather than fail a chat turn."""
        if value is None:
            return None
        if isinstance(value, bool):  # bool is a subclass of int, so check it first
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
        return None


class ChatResponse(BaseModel):
    """Structured response from the chat chain."""

    # RAW TEXT
    text: str                                    # the LLM-generated answer to the user's query
    language: Annotated[str, Field(pattern=r"^[a-z]{2}$")]  # ISO 639-1 two-letter code (e.g., "fr", "en")

    # CHUNKS
    retrieved_passage_ids: list[str]             # chunk_ids from ChunkInfo returned by the retriever
    retrieved_contexts: list[str]                # full text of the top-k chunks fed to the LLM as context

    # SOURCES CITED: human-readable document titles and page numbers that are built deterministically from retrieved chunks after LLM responds
    retrieved_sources: list[SourceReference] = Field(default_factory=list)  # structured (title, page) pairs for the sources cited footer; defaults to [] for eval backwards compatibility
    retrieved_source_titles: list[str]           # could be derived from retrieved_sources but maintained for eval backwards compatibility

