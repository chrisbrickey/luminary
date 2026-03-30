"""Pydantic schemas for chat functionality."""

from typing import Annotated

from pydantic import BaseModel, Field


class ChatResponse(BaseModel):
    """Structured response from the chat chain."""

    text: str                                    # the LLM-generated answer to the user's query
    retrieved_passage_ids: list[str]             # chunk_ids from ChunkInfo returned by the retriever
    retrieved_contexts: list[str]                # full text of the top-k chunks fed to the LLM as context
    retrieved_source_titles: list[str]           # human-readable document titles from retrieved chunks
    language: Annotated[str, Field(pattern=r"^[a-z]{2}$")]  # ISO 639-1 two-letter code (e.g., "fr", "en")
