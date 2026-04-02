"""Shared test fixtures and configuration."""

import os
from pathlib import Path
from typing import Any, Callable, List
from unittest.mock import Mock

import pytest

# Disable ChromaDB telemetry before import
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# Workaround for Python 3.14 compatibility with ChromaDB
# Set default values for ChromaDB settings that have type inference issues
os.environ.setdefault("CHROMA_SERVER_NOFILE", "65536")

# Import modules that use VECTOR_DB_PATH so we can reference them directly
from src.vectorstores import chroma, retriever

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


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


class FakeChatModel(BaseChatModel):
    """Fake chat model for testing that returns predictable responses."""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a fake response."""
        response_text = "This is a test response."
        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        return "fake-chat-model"


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    """Provide FakeEmbeddings instance."""
    return FakeEmbeddings()


@pytest.fixture
def fake_chat_model() -> FakeChatModel:
    """Provide FakeChatModel instance."""
    return FakeChatModel()


@pytest.fixture
def mock_retriever_with_docs() -> Callable[[List[Document]], Mock]:
    """Factory for creating mock retriever with specified documents."""

    def _make(docs: List[Document]) -> Mock:
        retriever = Mock()
        retriever.invoke.return_value = docs
        return retriever

    return _make


@pytest.fixture
def mock_llm_with_response() -> Callable[[str], Mock]:
    """Factory for creating mock LLM with specified response."""

    def _make(response_text: str = "Test response") -> Mock:
        llm = Mock()
        llm.invoke.return_value = AIMessage(content=response_text)
        return llm

    return _make


@pytest.fixture
def make_test_document() -> Callable[..., Document]:
    """Factory for creating test documents with default metadata."""

    def _make(
        content: str,
        chunk_id: str,
        chunk_index: int = 0,
        doc_id: str = "test-doc",
        title: str = "Test Document",
        author: str = "voltaire",
        page_number: int | None = None,
    ) -> Document:
        metadata = {
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "document_id": doc_id,
            "document_title": title,
            "author": author,
            "source": f"https://example.com/{doc_id}",
        }
        if page_number is not None:
            metadata["page_number"] = page_number
        return Document(page_content=content, metadata=metadata)

    return _make


@pytest.fixture
def test_db_path(tmp_path: Path, monkeypatch) -> Path:
    """Provide a temporary ChromaDB path with centralized monkeypatching.

    This fixture patches VECTOR_DB_PATH in all modules that import it,
    ensuring test isolation without needing to repeat monkeypatch calls
    in every test file.

    Args:
        tmp_path: pytest's temporary directory fixture
        monkeypatch: pytest's monkeypatch fixture

    Returns:
        Path to temporary test database directory
    """
    db_path = tmp_path / "chroma_db"

    # Patch VECTOR_DB_PATH only in the modules that actually use it
    # We don't patch src.configs.common because the modules import with
    # "from src.configs.common import VECTOR_DB_PATH" which creates local bindings
    monkeypatch.setattr(chroma, "VECTOR_DB_PATH", db_path)
    monkeypatch.setattr(retriever, "VECTOR_DB_PATH", db_path)

    return db_path


@pytest.fixture
def setup_test_db(test_db_path: Path, fake_embeddings: FakeEmbeddings) -> tuple[Path, FakeEmbeddings]:
    """Provide empty ChromaDB path with FakeEmbeddings for integration tests.

    This fixture provides the foundation for integration tests that need a
    ChromaDB instance. The database is empty - tests should populate it with
    their own documents using embed_and_store().

    Returns:
        tuple: (test_db_path, fake_embeddings) where:
            - test_db_path: Path to temporary test database
            - fake_embeddings: FakeEmbeddings instance for deterministic testing
    """
    return test_db_path, fake_embeddings


@pytest.fixture(autouse=True)
def clear_i18n_cache() -> None:
    """Clear i18n message cache before each test to ensure isolation."""
    from src.i18n import clear_cache
    clear_cache()


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: List[pytest.Item]
) -> None:
    """Print banner when running external tests."""
    # Check if external marker expression is being used to exclude tests
    markexpr = config.getoption("-m", default="")
    is_excluding_external = "not external" in markexpr

    # Only print banner if external tests will actually run
    has_external_tests = any(
        item.get_closest_marker("external") is not None
        for item in items
    )

    if has_external_tests and not is_excluding_external:
        print("\n" + "=" * 70)
        print("RUNNING EXTERNAL TESTS")
        print("=" * 70)
        print("These tests make real rpc calls and may take several minutes.")
        print("=" * 70 + "\n")
