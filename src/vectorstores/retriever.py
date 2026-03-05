"""Build a LangChain retriever backed by an existing ChromaDB collection."""

import logging
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings

from src.configs.vectorstore_config import COLLECTION_NAME, DEFAULT_K, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def build_retriever(
    persist_dir: Path | str,
    collection_name: str = COLLECTION_NAME,
    embeddings: Embeddings | None = None,
    k: int = DEFAULT_K,
    author: str | None = None,
) -> VectorStoreRetriever:
    """Build a retriever from an existing ChromaDB vectorstore.

    Args:
        persist_dir: Directory path where ChromaDB vectorstore is persisted
        collection_name: Name of the ChromaDB collection (default: COLLECTION_NAME)
        embeddings: Embeddings instance to use. If None, defaults to
            OllamaEmbeddings(model=EMBEDDING_MODEL)
        k: Number of documents to retrieve (default: DEFAULT_K)
        author: Optional author filter. If provided, only chunks from this
            author will be retrieved using ChromaDB metadata filtering

    Returns:
        VectorStoreRetriever configured with the specified parameters

    Notes:
        - The persist_dir must contain an existing ChromaDB collection
        - Author filtering is applied at the ChromaDB level, not post-retrieval
        - The author field in metadata must match exactly (case-sensitive)
    """
    # Use Ollama embeddings as default if none provided
    if embeddings is None:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    logger.info(
        "Opening ChromaDB collection '%s' at %s (k=%d, author=%r)",
        collection_name,
        persist_dir,
        k,
        author,
    )

    # Load existing vectorstore
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    # Build search kwargs with optional author filter
    search_kwargs: dict[str, Any] = {"k": k}
    if author is not None:
        search_kwargs["filter"] = {"author": author}

    # Convert to retriever
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    return retriever
