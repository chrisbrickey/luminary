"""Split LangChain Documents into chunks with ChunkInfo-compatible metadata."""

import hashlib
from typing import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.schemas import ChunkInfo


def chunk_documents(
    documents: Sequence[Document],
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Split documents into smaller chunks with validated metadata.

    Args:
        documents: List of LangChain Documents to chunk
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of overlapping characters between chunks

    Returns:
        List of Document chunks with enriched metadata including chunk_id,
        chunk_index, and all original metadata preserved.

    Raises:
        ValueError: If chunk metadata fails ChunkInfo validation

    Notes:
        - Chunks are split using RecursiveCharacterTextSplitter with
          separators optimized for French prose
        - Each chunk gets a deterministic chunk_id (SHA256 hash of
          document_id:page_number:chunk_index, truncated to 12 chars)
        - All chunks are validated against ChunkInfo schema before return
        - Empty documents produce no chunks
    """
    # Configure splitter for French prose with paragraph/sentence boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks: list[Document] = []

    for doc in documents:
        # Skip empty documents
        if not doc.page_content.strip():
            continue

        # Split document into chunks
        doc_chunks = splitter.split_documents([doc])

        # Enrich each chunk with metadata
        for chunk_index, chunk in enumerate(doc_chunks):
            # Generate deterministic chunk ID
            document_id = chunk.metadata.get("document_id", "")
            page_number = chunk.metadata.get("page_number", 0)
            chunk_id = _generate_chunk_id(document_id, page_number, chunk_index)

            # Add chunk-specific metadata while preserving all original metadata
            chunk.metadata["chunk_id"] = chunk_id
            chunk.metadata["chunk_index"] = chunk_index

            # Validate metadata against ChunkInfo schema
            _validate_chunk_metadata(chunk.metadata)

            all_chunks.append(chunk)

    return all_chunks


def _generate_chunk_id(document_id: str, page_number: int, chunk_index: int) -> str:
    """Generate deterministic chunk ID from document_id, page_number, and chunk_index.

    Args:
        document_id: Unique identifier for the source document collection
        page_number: Page number within the document
        chunk_index: 0-indexed position of chunk within the page

    Returns:
        12-character truncated SHA256 hash of "document_id:page_number:chunk_index"
    """
    composite_key = f"{document_id}:{page_number}:{chunk_index}"
    hash_digest = hashlib.sha256(composite_key.encode("utf-8")).hexdigest()
    return hash_digest[:12]


def _validate_chunk_metadata(metadata: dict[str, object]) -> None:
    """Validate chunk metadata against ChunkInfo schema.

    Args:
        metadata: Dictionary of metadata to validate

    Raises:
        ValueError: If metadata fails ChunkInfo validation
    """
    try:
        ChunkInfo.model_validate(metadata)
    except Exception as e:
        raise ValueError(f"Chunk metadata validation failed: {e}") from e
