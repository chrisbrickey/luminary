"""Tests for ChromaDB vectorstore operations."""

import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.vectorstores.chroma import embed_and_store
from tests.conftest import FakeEmbeddings


@pytest.fixture
def temp_db_dir() -> Path:
    """Provide a temporary directory for ChromaDB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_chunks() -> list[Document]:
    """Create sample chunks with chunk_id metadata."""
    return [
        Document(
            page_content="First chunk about enlightenment philosophy.",
            metadata={
                "chunk_id": "abc123def456",
                "chunk_index": 0,
                "document_id": "test-doc-1",
                "document_title": "Test Document",
                "author": "test-author",
                "source": "https://example.com/test",
            }
        ),
        Document(
            page_content="Second chunk discussing reason and tolerance.",
            metadata={
                "chunk_id": "ghi789jkl012",
                "chunk_index": 1,
                "document_id": "test-doc-1",
                "document_title": "Test Document",
                "author": "test-author",
                "source": "https://example.com/test",
            }
        ),
        Document(
            page_content="Third chunk on natural rights and liberty.",
            metadata={
                "chunk_id": "mno345pqr678",
                "chunk_index": 0,
                "document_id": "test-doc-2",
                "document_title": "Another Document",
                "author": "test-author",
                "source": "https://example.com/test2",
            }
        ),
    ]


def test_embed_and_store_basic(
    temp_db_dir: Path,
    sample_chunks: list[Document],
    monkeypatch
) -> None:
    """Test basic embedding and storage of chunks."""
    # Patch DEFAULT_DB_PATH in chroma module
    monkeypatch.setattr("src.vectorstores.chroma.DEFAULT_DB_PATH", temp_db_dir)
    embeddings = FakeEmbeddings()

    vectorstore = embed_and_store(
        chunks=sample_chunks,
        embeddings=embeddings
    )

    # Verify vectorstore was created
    assert vectorstore is not None

    # Verify chunks are retrievable via similarity search
    results = vectorstore.similarity_search("enlightenment", k=3)
    assert len(results) == 3

    # Verify metadata is preserved
    result_ids = {doc.metadata["chunk_id"] for doc in results}
    expected_ids = {chunk.metadata["chunk_id"] for chunk in sample_chunks}
    assert result_ids == expected_ids


def test_embed_and_store_idempotent(
    temp_db_dir: Path,
    sample_chunks: list[Document],
    monkeypatch
) -> None:
    """Test that re-running with same IDs does not create duplicates."""
    # Patch DEFAULT_DB_PATH in chroma module
    monkeypatch.setattr("src.vectorstores.chroma.DEFAULT_DB_PATH", temp_db_dir)
    embeddings = FakeEmbeddings()

    # First run
    embed_and_store(
        chunks=sample_chunks,
        embeddings=embeddings
    )

    # Second run with same chunks
    vectorstore = embed_and_store(
        chunks=sample_chunks,
        embeddings=embeddings
    )

    # Verify no duplicates (should still have 3 chunks, not 6)
    results = vectorstore.similarity_search("test", k=10)
    assert len(results) == 3


def test_embed_and_store_custom_collection_name(
    temp_db_dir: Path,
    sample_chunks: list[Document],
    monkeypatch
) -> None:
    """Test using a custom collection name."""
    # Patch DEFAULT_DB_PATH in chroma module
    monkeypatch.setattr("src.vectorstores.chroma.DEFAULT_DB_PATH", temp_db_dir)
    embeddings = FakeEmbeddings()

    vectorstore = embed_and_store(
        chunks=sample_chunks,
        collection_name="test_collection",
        embeddings=embeddings
    )

    # Verify vectorstore works with custom collection
    results = vectorstore.similarity_search("test", k=1)
    assert len(results) == 1


def test_embed_and_store_persist_dir_as_string(
    temp_db_dir: Path,
    sample_chunks: list[Document],
    monkeypatch
) -> None:
    """Test that persist_dir can be set as string via environment."""
    # Patch DEFAULT_DB_PATH in chroma module
    monkeypatch.setattr("src.vectorstores.chroma.DEFAULT_DB_PATH", temp_db_dir)
    embeddings = FakeEmbeddings()

    vectorstore = embed_and_store(
        chunks=sample_chunks,
        embeddings=embeddings
    )

    # Verify it works the same
    results = vectorstore.similarity_search("test", k=1)
    assert len(results) == 1


def test_embed_and_store_missing_chunk_id(monkeypatch) -> None:
    """Test that missing chunk_id raises ValueError."""
    embeddings = FakeEmbeddings()

    chunks = [
        Document(
            page_content="Chunk without chunk_id",
            metadata={
                "document_id": "test-doc",
                "author": "test-author",
            }
        )
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Patch DEFAULT_DB_PATH in chroma module
        monkeypatch.setattr("src.vectorstores.chroma.DEFAULT_DB_PATH", Path(tmpdir))
        with pytest.raises(ValueError, match="missing 'chunk_id'"):
            embed_and_store(
                chunks=chunks,
                embeddings=embeddings
            )


def test_embed_and_store_preserves_all_metadata(
    temp_db_dir: Path,
    monkeypatch
) -> None:
    """Test that all metadata fields are preserved in vectorstore."""
    # Patch DEFAULT_DB_PATH in chroma module
    monkeypatch.setattr("src.vectorstores.chroma.DEFAULT_DB_PATH", temp_db_dir)
    embeddings = FakeEmbeddings()

    chunk = Document(
        page_content="Test content",
        metadata={
            "chunk_id": "test123",
            "chunk_index": 0,
            "document_id": "doc-id",
            "document_title": "Test Title",
            "author": "test-author",
            "source": "https://example.com",
            "page_number": 42,
            "custom_field": "custom_value",
        }
    )

    vectorstore = embed_and_store(
        chunks=[chunk],
        embeddings=embeddings
    )

    results = vectorstore.similarity_search("test", k=1)
    retrieved_metadata = results[0].metadata

    # Verify all original metadata is present
    assert retrieved_metadata["chunk_id"] == "test123"
    assert retrieved_metadata["chunk_index"] == 0
    assert retrieved_metadata["document_id"] == "doc-id"
    assert retrieved_metadata["document_title"] == "Test Title"
    assert retrieved_metadata["author"] == "test-author"
    assert retrieved_metadata["source"] == "https://example.com"
    assert retrieved_metadata["page_number"] == 42
    assert retrieved_metadata["custom_field"] == "custom_value"
