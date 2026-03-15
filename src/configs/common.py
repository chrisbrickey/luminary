"""Common configuration constants used across the application."""

from pathlib import Path

# Raw documents directory - used by scraping and embedding scripts
DEFAULT_RAW_DIR = Path("data/raw")

# ChromaDB database path - used by both document loaders and vectorstore operations
DEFAULT_DB_PATH = Path("data/chroma_db")

# Default LLM model for chat and generation tasks
DEFAULT_LLM_MODEL = "mistral"

# Default response language (overridden by language detected in user's prompt)
DEFAULT_RESPONSE_LANGUAGE = "en"
