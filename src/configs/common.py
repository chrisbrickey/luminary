"""Common configuration constants used across the application."""

from pathlib import Path

# ChromaDB database path - used by both document loaders and vectorstore operations
DEFAULT_DB_PATH = Path("data/chroma_db")
