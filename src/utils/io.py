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
