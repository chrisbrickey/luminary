"""Shared test fixtures and configuration."""

import os
from pathlib import Path
from typing import Any, Callable, List
from unittest.mock import Mock

import pytest
from langchain_core.prompts import ChatPromptTemplate

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

# --- Fake author constants for unit tests ---
#
# Import from fake_authors module for easy access across all tests
from fake_authors import FAKE_AUTHOR_A, FAKE_AUTHOR_B, FAKE_AUTHOR_C  # noqa: F401


def _fake_prompt_factory() -> ChatPromptTemplate:
    """Generic fake prompt factory for test authors."""
    return ChatPromptTemplate.from_messages([("system", "You are {author}"), ("human", "{question}")])


# Lazy import to avoid circular dependencies
def _get_author_config_class() -> type:
    """Get AuthorConfig class without circular import."""
    from src.configs.authors import AuthorConfig
    return AuthorConfig


def _create_fake_author_config(exit_message: str) -> Any:
    """Create a fake AuthorConfig for testing.

    Args:
        exit_message: The exit message for this fake author

    Returns:
        AuthorConfig instance for testing
    """
    AuthorConfig = _get_author_config_class()
    return AuthorConfig(
        prompt_factory=_fake_prompt_factory,
        exit_message=exit_message,
    )


# Fake author configs available to all unit tests
# These are created lazily to avoid import issues
def get_fake_author_configs() -> dict[str, Any]:
    """Get fake AUTHOR_CONFIGS dictionary for unit tests.

    Returns:
        Dictionary mapping fake author names to AuthorConfig instances
    """
    return {
        FAKE_AUTHOR_A: _create_fake_author_config("Au revoir - Condorcet"),
        FAKE_AUTHOR_B: _create_fake_author_config("Farewell - Mary Wollstonecraft"),
        FAKE_AUTHOR_C: _create_fake_author_config("Adieu - Diderot"),
    }


# --- Test fixtures ---


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
def mock_author_configs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Mock AUTHOR_CONFIGS with fake authors for isolated unit testing.

    Unit tests should not depend on production author configurations.
    This fixture ensures tests use generic fake authors instead.

    Not autouse - tests that need mocked authors should use this fixture
    explicitly or create their own autouse fixture.

    Usage in test files:
        @pytest.fixture(autouse=True)
        def _mock_authors(mock_author_configs):
            '''Apply author mocking to all tests in this file.'''
            pass

    Returns:
        Dictionary of fake AUTHOR_CONFIGS
    """
    from src.configs import authors

    fake_configs = get_fake_author_configs()
    monkeypatch.setattr(authors, "AUTHOR_CONFIGS", fake_configs)
    return fake_configs


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
