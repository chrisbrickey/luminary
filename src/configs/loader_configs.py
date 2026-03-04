"""Configuration for document loaders."""

from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_DB_PATH = Path("data/chroma_db")

class WikisourceCollectionConfig(BaseModel):
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


# Pre-built configuration for Voltaire's 25 Lettres philosophiques
LETTRES_PHILOSOPHIQUES_CONFIG = WikisourceCollectionConfig(
    document_id="voltaire_lettres_philosophiques-1734",
    document_title="Lettres Philosophiques 1734",
    author="voltaire",
    page_title_template="Lettres philosophiques/Lettre {n}",
    total_pages=None,  # Auto-discovery
    api_url="https://fr.wikisource.org/w/api.php"
)


# Registry mapping author key to configuration
INGEST_CONFIGS: dict[str, WikisourceCollectionConfig] = {
    "voltaire": LETTRES_PHILOSOPHIQUES_CONFIG,
}