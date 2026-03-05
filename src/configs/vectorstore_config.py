"""Configuration constants for ChromaDB vectorstore operations."""

from src.configs.common import DEFAULT_DB_PATH

# ChromaDB collection name for storing philosopher embeddings
COLLECTION_NAME = "philosophes"

# Embedding model for Ollama embeddings
EMBEDDING_MODEL = "nomic-embed-text"

# Default number of chunks to retrieve per query
DEFAULT_K = 5

__all__ = [
    "DEFAULT_DB_PATH",
    "COLLECTION_NAME",
    "EMBEDDING_MODEL",
    "DEFAULT_K",
]
