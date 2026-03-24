"""Common configuration constants used across the application."""

import os
from pathlib import Path

# Raw documents directory - used by scraping and embedding scripts
RAW_DATA_PATH = Path("data/raw")

# ChromaDB vector database path - configured per environment via CHROMA_DB_PATH env var
# Defaults to "data/chroma_db" for local development
VECTOR_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", "data/chroma_db"))

# Default LLM model for chat and generation tasks
DEFAULT_LLM_MODEL = "mistral"

# Default response language (overridden by language detected in user's prompt)
DEFAULT_RESPONSE_LANGUAGE = "en"
