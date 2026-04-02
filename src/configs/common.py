"""Common configuration constants used across modules."""

import os
from pathlib import Path

# Raw documents path; Used by scraping and embedding scripts
RAW_DATA_PATH = Path("data/raw")

# Vector database path; Configured per environment via CHROMA_DB_PATH env var
# Defaults to "data/chroma_db" for local development
VECTOR_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", "data/chroma_db"))

# Default LLM for chat chain
DEFAULT_CHAT_MODEL = "mistral"

# Default embedding model for vector store operations
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

# Default response language (overridden by language detected in user's prompt)
DEFAULT_RESPONSE_LANGUAGE = "en"
