"""Pydantic schemas to validate data structures."""

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class WikisourceCollection(BaseModel):
    """Specification for loading a collection from Wikisource.

    Attributes:
        document_id: Stable slug used to group all chunks from this collection
        document_title: Human-readable display name of the collection
        author: Author name (lowercase)
        page_title_template: Wikisource page title with {n} placeholder
        total_pages: Total number of pages (auto-discovered if None)
        api_url: Wikisource API URL
    """

    document_id: str = Field(
        ...,
        description="stable slug used to group all chunks from this collection"
    )
    document_title: str = Field(
        ...,
        description="human-readable display name of the collection"
    )
    author: str = Field(
        ...,
        description="author name (lowercase)"
    )
    page_title_template: str = Field(
        ...,
        description="wikisource page title with {n} placeholder"
    )
    total_pages: int | None = Field(
        None,
        description="total number of pages (auto-discovered if None)"
    )
    api_url: str = Field(
        "https://fr.wikisource.org/w/api.php",
        description="Wikisource API URL"
    )


class ChunkInfo(BaseModel):
    """Metadata for a single text chunk stored in ChromaDB.

    chunk_id is a SHA256 hash of document_id:chunk_index.
    This results in a unique and idempotent ID. We can re-run
    ingestion without creating duplicate entries in ChromaDB.
    NB: If we ever change the chunking logic (e.g., chunk size),
    then the previously ingested data should be cleared and re-ingested.

    extra="allow" permits attachment of domain-specific fields
    (e.g., letter_number=3, author="voltaire"). Such extra fields
    would be validated as-is (no type checking) and stored
    alongside the required fields."""

    model_config = ConfigDict(extra="allow")

    chunk_id: str                      # unique ID for this chunk
    chunk_index: int                   # 0-indexed position of this chunk within its document
    document_id: str                   # unique ID for the source; groups all chunks from the same document
    document_title: str | None = None  # optional human-readable display name for the source document
    author: str | None = None          # lowercase author key used for metadata filtering (e.g. "voltaire")
    source: str                        # link to the source document (e.g., a URL, a local file path)

    @field_validator("author")
    @classmethod
    def author_must_be_lowercase(cls, v: str | None) -> str | None:
        if v is not None and v != v.lower():
            raise ValueError(f"author must be lowercase, got: {v!r}")
        return v

class ChatResponse(BaseModel):
    """Structured response from the chat chain."""

    text: str                                    # the LLM-generated answer to the user's query
    retrieved_passage_ids: list[str]             # chunk_ids from ChunkInfo returned by the retriever
    retrieved_contexts: list[str]                # full text of the top-k chunks fed to the LLM as context
    retrieved_source_titles: list[str]           # human-readable document titles from retrieved chunks
    language: Annotated[str, Field(pattern=r"^[a-z]{2}$")]  # ISO 639-1 two-letter code (e.g., "fr", "en")


class MetricResult(BaseModel):
    """Result from a single metric on a single example.

    Uses standard scoring attributes:
        ge=0.0: greater than or equal to 0
        le=1.0: less than or equal to 1
    """

    name: str = Field(..., description="Name of the metric")
    score: float = Field(..., ge=0.0, le=1.0, description="Score 0.0 to 1.0 where 1.0 is perfect")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional details about the metric result")
