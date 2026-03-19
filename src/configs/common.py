"""Common configuration constants used across the application."""

import os
from pathlib import Path

# Raw documents directory - used by scraping and embedding scripts
DEFAULT_RAW_DIR = Path("data/raw")

# ChromaDB database path - configured per environment via CHROMA_DB_PATH env var
# Defaults to "data/chroma_db" for local development
DEFAULT_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", "data/chroma_db"))

# Default LLM model for chat and generation tasks
DEFAULT_LLM_MODEL = "mistral"

# Default response language (overridden by language detected in user's prompt)
DEFAULT_RESPONSE_LANGUAGE = "en"
