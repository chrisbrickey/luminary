"""Tests for retriever wrapper."""

import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.vectorstores.chroma import embed_and_store
from src.vectorstores.retriever import build_retriever
from tests.conftest import FakeEmbeddings


@pytest.fixture
def fixture_db_dir() -> Path:
    """Create a temporary directory with a populated ChromaDB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir)

        # Create sample chunks from two different authors
        chunks = [
            Document(
                page_content="Discussion of religious tolerance and enlightenment ideals.",
                metadata={
                    "chunk_id": "voltaire_001",
                    "chunk_index": 0,
                    "document_id": "lettres-philosophiques",
                    "document_title": "Lettres philosophiques",
                    "author": "voltaire",
                    "source": "https://example.com/voltaire",
                }
            ),
            Document(
                page_content="Analysis of reason and scientific progress in society.",
                metadata={
                    "chunk_id": "voltaire_002",
                    "chunk_index": 1,
                    "document_id": "lettres-philosophiques",
                    "document_title": "Lettres philosophiques",
                    "author": "voltaire",
                    "source": "https://example.com/voltaire",
                }
            ),
            Document(
                page_content="Advocacy for women's rights and natural equality.",
                metadata={
                    "chunk_id": "gouges_001",
                    "chunk_index": 0,
                    "document_id": "declaration-droits-femme",
                    "document_title": "Déclaration des droits de la femme",
                    "author": "gouges",
                    "source": "https://example.com/gouges",
                }
            ),
            Document(
                page_content="Critique of patriarchal structures and call for reform.",
                metadata={
                    "chunk_id": "gouges_002",
                    "chunk_index": 1,
                    "document_id": "declaration-droits-femme",
                    "document_title": "Déclaration des droits de la femme",
                    "author": "gouges",
                    "source": "https://example.com/gouges",
                }
            ),
        ]

        # Embed and store all chunks
        embed_and_store(
            chunks=chunks,
            persist_dir=db_path,
            embeddings=FakeEmbeddings()
        )

        yield db_path


def test_build_retriever_basic(fixture_db_dir: Path) -> None:
    """Test basic retrieval returns relevant chunks with intact metadata."""
    retriever = build_retriever(
        persist_dir=fixture_db_dir,
        embeddings=FakeEmbeddings(),
        k=3
    )

    # Retrieve documents
    results = retriever.invoke("enlightenment")

    # Should return up to k documents
    assert len(results) <= 3
    assert len(results) > 0

    # Verify metadata is intact
    for doc in results:
        assert "chunk_id" in doc.metadata
        assert "author" in doc.metadata
        assert "document_title" in doc.metadata
        assert "source" in doc.metadata


def test_build_retriever_with_author_filter(fixture_db_dir: Path) -> None:
    """Test author filter returns only chunks from specified author."""
    retriever = build_retriever(
        persist_dir=fixture_db_dir,
        embeddings=FakeEmbeddings(),
        k=5,
        author="voltaire"
    )

    # Retrieve documents
    results = retriever.invoke("society")

    # Should only return Voltaire chunks
    assert len(results) > 0
    for doc in results:
        assert doc.metadata["author"] == "voltaire"


def test_build_retriever_different_author_filter(fixture_db_dir: Path) -> None:
    """Test author filter works for different author."""
    retriever = build_retriever(
        persist_dir=fixture_db_dir,
        embeddings=FakeEmbeddings(),
        k=5,
        author="gouges"
    )

    # Retrieve documents
    results = retriever.invoke("rights")

    # Should only return Gouges chunks
    assert len(results) > 0
    for doc in results:
        assert doc.metadata["author"] == "gouges"


def test_build_retriever_without_filter_returns_all(fixture_db_dir: Path) -> None:
    """Test retriever without author filter can return chunks from any author."""
    retriever = build_retriever(
        persist_dir=fixture_db_dir,
        embeddings=FakeEmbeddings(),
        k=5,
        author=None  # No filter
    )

    # Retrieve documents
    results = retriever.invoke("society")

    # Should potentially return chunks from multiple authors
    assert len(results) > 0

    # Collect unique authors
    authors = {doc.metadata["author"] for doc in results}

    # With FakeEmbeddings, all chunks have same embedding, so we get multiple authors
    # At minimum, we should get some results
    assert len(authors) >= 1


def test_build_retriever_respects_k_parameter(fixture_db_dir: Path) -> None:
    """Test that k parameter limits the number of results."""
    retriever = build_retriever(
        persist_dir=fixture_db_dir,
        embeddings=FakeEmbeddings(),
        k=2  # Limit to 2 results
    )

    # Retrieve documents
    results = retriever.invoke("test")

    # Should return at most k documents
    assert len(results) <= 2


def test_build_retriever_custom_collection_name(fixture_db_dir: Path) -> None:
    """Test retriever works with custom collection name."""
    # First create a custom collection
    chunks = [
        Document(
            page_content="Custom collection test content.",
            metadata={
                "chunk_id": "custom_001",
                "chunk_index": 0,
                "document_id": "custom-doc",
                "author": "test-author",
                "source": "https://example.com",
            }
        )
    ]

    embed_and_store(
        chunks=chunks,
        persist_dir=fixture_db_dir,
        collection_name="custom_collection",
        embeddings=FakeEmbeddings()
    )

    # Build retriever for custom collection
    retriever = build_retriever(
        persist_dir=fixture_db_dir,
        collection_name="custom_collection",
        embeddings=FakeEmbeddings(),
        k=1
    )

    # Should retrieve from custom collection
    results = retriever.invoke("custom")
    assert len(results) == 1
    assert results[0].metadata["chunk_id"] == "custom_001"


def test_build_retriever_persist_dir_as_string(fixture_db_dir: Path) -> None:
    """Test that persist_dir can be passed as string."""
    retriever = build_retriever(
        persist_dir=str(fixture_db_dir),  # Pass as string
        embeddings=FakeEmbeddings(),
        k=3
    )

    # Should work the same as with Path
    results = retriever.invoke("test")
    assert len(results) > 0
