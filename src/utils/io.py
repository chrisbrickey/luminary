"""Input/output utilities for documents."""

import json
import logging
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def save_documents_to_disk(
    documents: list[Document],
    directory: Path | str
) -> list[Path]:
    """Save a list of Documents to disk as JSON files.

    Each document is saved as page_NN.json where NN is the page number
    extracted from the document's metadata.

    Args:
        documents: List of Document objects to save
        directory: Directory path where files will be saved

    Returns:
        List of Path objects for the created files

    Raises:
        ValueError: If a document is missing page_number metadata
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for doc in documents:
        # Extract page number from metadata
        page_number = doc.metadata.get('page_number')
        if page_number is None:
            raise ValueError(
                f"Document is missing 'page_number' in metadata: {doc.metadata}"
            )

        # Create filename
        filename = f"page_{page_number:02d}.json"
        filepath = directory / filename

        # Prepare document data for JSON serialization
        doc_data = {
            'page_content': doc.page_content,
            'metadata': doc.metadata
        }

        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(doc_data, f, ensure_ascii=False, indent=2)

        saved_paths.append(filepath)
        logger.debug(f"Saved document to {filepath}")

    logger.info(f"Saved {len(saved_paths)} documents to {directory}")
    return saved_paths


def load_documents_from_disk(directory: Path | str) -> list[Document]:
    """Load documents from JSON files in a directory.

    Loads all page_*.json files from the specified directory and reconstructs
    Document objects with their original page_content and metadata.

    Args:
        directory: Directory containing page_*.json files

    Returns:
        List of Document objects loaded from disk, sorted by filename

    Raises:
        FileNotFoundError: If directory does not exist
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    documents = []

    # Load all page_*.json files
    for json_file in sorted(directory.glob("page_*.json")):
        with open(json_file, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)

        doc = Document(
            page_content=doc_data['page_content'],
            metadata=doc_data['metadata']
        )
        documents.append(doc)

    logger.debug(f"Loaded {len(documents)} documents from {directory}")
    return documents
