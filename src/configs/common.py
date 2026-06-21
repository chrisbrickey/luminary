"""Common configuration constants used across modules."""

import os
from pathlib import Path

# Raw documents path; Used by scraping and embedding scripts
RAW_DATA_PATH = Path("data/raw")

# Vector database path; Configured per environment via CHROMA_DB_PATH env var
# Defaults to "data/chroma_db" for local development
VECTOR_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", "data/chroma_db"))

# Default embedding model for vector store operations
# Initial embedding model was nomic-embed-text (768 dimemnsions).
# Changed to a multilingual model, bge-m3 (1024 dimensions), to minimize retrieval gap between prompts of different languages.
DEFAULT_EMBEDDING_MODEL = "bge-m3"

# Default LLM for chat chain
DEFAULT_CHAT_MODEL = "mistral"

# Determinism settings for chat LLM
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_LLM_SEED: int = 42

# Language codes (ISO 639-1)
ENGLISH_ISO_CODE = "en"
FRENCH_ISO_CODE = "fr"
GERMAN_ISO_CODE = "de"
ITALIAN_ISO_CODE = "it"
SPANISH_ISO_CODE = "es"
SWAHILI_ISO_CODE = "sw"

# Default response language (overridden by language detected in user's prompt)
DEFAULT_RESPONSE_LANGUAGE = ENGLISH_ISO_CODE
