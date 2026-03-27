"""ChromaDB vectorstore operations for embeddings storage and retrieval."""

from typing import Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from src.configs.common import VECTOR_DB_PATH
from src.configs.vectorstore_config import COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL


def _get_embeddings_instance(embeddings: Embeddings | None) -> Embeddings:
    """Get embeddings instance, using default Ollama if none provided.

    Args:
        embeddings: Optional embeddings instance. If None, creates
            OllamaEmbeddings with DEFAULT_EMBEDDING_MODEL.

    Returns:
        Embeddings instance ready to use
    """
    if embeddings is None:
        return OllamaEmbeddings(model=DEFAULT_EMBEDDING_MODEL)
    return embeddings


def _extract_and_validate_chunk_ids(chunks: Sequence[Document]) -> list[str]:
    """Extract chunk IDs from documents, validating all chunks have IDs.

    Args:
        chunks: Sequence of Document chunks to extract IDs from

    Returns:
        List of chunk_id strings in same order as input chunks

    Raises:
        ValueError: If any chunk is missing 'chunk_id' in metadata
    """
    chunk_ids = []
    for chunk in chunks:
        chunk_id = chunk.metadata.get("chunk_id")
        if not chunk_id:
            raise ValueError(
                f"Chunk missing 'chunk_id' in metadata: {chunk.metadata}"
            )
        chunk_ids.append(chunk_id)
    return chunk_ids


def embed_and_store(
    chunks: Sequence[Document],
    collection_name: str = COLLECTION_NAME,
    embeddings: Embeddings | None = None,
) -> Chroma:
    """Embed chunks and store them in ChromaDB with idempotent IDs.

    Args:
        chunks: List of Document chunks to embed and store. Each chunk must
            have a 'chunk_id' in its metadata.
        collection_name: Name of the ChromaDB collection (default: COLLECTION_NAME)
        embeddings: Embeddings instance to use. If None, defaults to
            OllamaEmbeddings(model=DEFAULT_EMBEDDING_MODEL)

    Returns:
        Chroma vectorstore instance containing the embedded chunks

    Raises:
        ValueError: If any chunk is missing 'chunk_id' in metadata

    Notes:
        - Uses chunk_id as document ID for idempotent upserts on re-run
        - Re-running with same chunks will update existing embeddings, not duplicate
    """
    embeddings_instance = _get_embeddings_instance(embeddings)
    chunk_ids = _extract_and_validate_chunk_ids(chunks)

    # Create vectorstore with explicit IDs for idempotent upserts
    vectorstore = Chroma.from_documents(
        documents=list(chunks),
        embedding=embeddings_instance,
        ids=chunk_ids,
        collection_name=collection_name,
        persist_directory=str(VECTOR_DB_PATH),
    )

    return vectorstore
