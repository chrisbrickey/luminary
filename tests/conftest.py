"""Shared test fixtures and configuration."""

import os
from typing import Any, Callable, List
from unittest.mock import Mock

import pytest

# Disable ChromaDB telemetry before import
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# Workaround for Python 3.14 compatibility with ChromaDB
# Set default values for ChromaDB settings that have type inference issues
os.environ.setdefault("CHROMA_SERVER_NOFILE", "65536")

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


@pytest.fixture(autouse=True)
def clear_i18n_cache() -> None:
    """Clear i18n message cache before each test to ensure isolation."""
    from src.i18n import clear_cache
    clear_cache()


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: List[pytest.Item]
) -> None:
    """Print banner when running external tests."""
    # Check if any collected items have the external marker
    has_external_tests = any(
        item.get_closest_marker("external") is not None
        for item in items
    )

    if has_external_tests:
        print("\n" + "=" * 70)
        print("RUNNING EXTERNAL TESTS")
        print("=" * 70)
        print("These tests make real rpc calls and may take several minutes.")
        print("=" * 70 + "\n")
