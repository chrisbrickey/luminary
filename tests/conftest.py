"""Shared test fixtures and configuration."""

import os
from typing import List

# Disable ChromaDB telemetry before import
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# Workaround for Python 3.14 compatibility with ChromaDB
# Set default values for ChromaDB settings that have type inference issues
os.environ.setdefault("CHROMA_SERVER_NOFILE", "65536")

from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """Fake embeddings that return constant 8-dimensional vectors.

    Used in tests to avoid real embedding API calls while still testing
    vectorstore operations.
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return constant 8-dim vector for each text."""
        return [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        """Return constant 8-dim vector for query."""
        return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
