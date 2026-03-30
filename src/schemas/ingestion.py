"""Pydantic schemas for data ingestion and document loading."""

from pydantic import BaseModel, Field


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
