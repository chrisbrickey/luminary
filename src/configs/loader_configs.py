"""Configuration for document loaders."""

from src.schemas import WikisourceCollection


# Pre-built specification for Voltaire's 25 Lettres philosophiques
LETTRES_PHILOSOPHIQUES = WikisourceCollection(
    document_id="voltaire_lettres_philosophiques-1734",
    document_title="Lettres Philosophiques 1734",
    author="voltaire",
    page_title_template="Lettres philosophiques/Lettre {n}",
    total_pages=None,  # Auto-discovery
    api_url="https://fr.wikisource.org/w/api.php"
)


# Registry mapping author key to collection specification
INGEST_CONFIGS: dict[str, WikisourceCollection] = {
    "voltaire": LETTRES_PHILOSOPHIQUES,
}