"""Configuration constants for ChromaDB vectorstore operations."""

from src.configs.common import DEFAULT_EMBEDDING_MODEL, VECTOR_DB_PATH

# ChromaDB collection name for storing philosopher embeddings
COLLECTION_NAME = "philosophes"

# Default number of chunks to retrieve per query
DEFAULT_K = 5

__all__ = [
    "VECTOR_DB_PATH",
    "COLLECTION_NAME",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_K",
]
