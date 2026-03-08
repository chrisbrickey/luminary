"""Integration tests for retriever with real ChromaDB."""

import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.vectorstores.chroma import embed_and_store
from src.vectorstores.retriever import build_retriever
from tests.conftest import FakeEmbeddings


def test_retriever_end_to_end_round_trip() -> None:
    """Test full round-trip: embed documents then retrieve them.

    This integration test verifies that:
    1. Documents can be embedded and stored in ChromaDB
    2. A retriever can be built from the same database
    3. The retriever can successfully retrieve the stored documents
    4. All metadata is preserved through the round-trip
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir)
        embeddings = FakeEmbeddings()

        # Step 1: Create and embed sample documents
        original_chunks = [
            Document(
                page_content="Discussion of tolerance and reason in Enlightenment thought.",
                metadata={
                    "chunk_id": "test_chunk_001",
                    "chunk_index": 0,
                    "document_id": "test-document-1",
                    "document_title": "Test Philosophical Text",
                    "author": "test-author",
                    "source": "https://example.com/test1",
                    "page_number": 1,
                }
            ),
            Document(
                page_content="Analysis of natural rights and social contracts.",
                metadata={
                    "chunk_id": "test_chunk_002",
                    "chunk_index": 1,
                    "document_id": "test-document-1",
                    "document_title": "Test Philosophical Text",
                    "author": "test-author",
                    "source": "https://example.com/test1",
                    "page_number": 2,
                }
            ),
            Document(
                page_content="Exploration of liberty and equality principles.",
                metadata={
                    "chunk_id": "test_chunk_003",
                    "chunk_index": 0,
                    "document_id": "test-document-2",
                    "document_title": "Another Test Text",
                    "author": "test-author",
                    "source": "https://example.com/test2",
                    "page_number": 1,
                }
            ),
        ]

        # Step 2: Embed and store
        embed_and_store(
            chunks=original_chunks,
            persist_dir=db_path,
            embeddings=embeddings
        )

        # Step 3: Build retriever from same database
        retriever = build_retriever(
            persist_dir=db_path,
            embeddings=embeddings,
            k=5
        )

        # Step 4: Retrieve documents
        retrieved_docs = retriever.invoke("test query")

        # Verify we got documents back
        assert len(retrieved_docs) == 3

        # Verify all original chunk IDs are present
        retrieved_ids = {doc.metadata["chunk_id"] for doc in retrieved_docs}
        original_ids = {chunk.metadata["chunk_id"] for chunk in original_chunks}
        assert retrieved_ids == original_ids

        # Verify metadata is fully preserved
        for retrieved_doc in retrieved_docs:
            # Find corresponding original chunk
            original_chunk = next(
                c for c in original_chunks
                if c.metadata["chunk_id"] == retrieved_doc.metadata["chunk_id"]
            )

            # Check all metadata fields
            assert retrieved_doc.metadata["chunk_index"] == original_chunk.metadata["chunk_index"]
            assert retrieved_doc.metadata["document_id"] == original_chunk.metadata["document_id"]
            assert retrieved_doc.metadata["document_title"] == original_chunk.metadata["document_title"]
            assert retrieved_doc.metadata["author"] == original_chunk.metadata["author"]
            assert retrieved_doc.metadata["source"] == original_chunk.metadata["source"]
            assert retrieved_doc.metadata["page_number"] == original_chunk.metadata["page_number"]


def test_retriever_with_multiple_authors_integration() -> None:
    """Integration test for retriever with author filtering across full pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir)
        embeddings = FakeEmbeddings()

        # Create documents from two different authors
        chunks = [
            Document(
                page_content="First author's perspective on philosophy.",
                metadata={
                    "chunk_id": "author1_chunk001",
                    "chunk_index": 0,
                    "document_id": "author1-doc",
                    "author": "author-one",
                    "source": "https://example.com/author1",
                }
            ),
            Document(
                page_content="First author's continued discussion.",
                metadata={
                    "chunk_id": "author1_chunk002",
                    "chunk_index": 1,
                    "document_id": "author1-doc",
                    "author": "author-one",
                    "source": "https://example.com/author1",
                }
            ),
            Document(
                page_content="Second author's perspective on philosophy.",
                metadata={
                    "chunk_id": "author2_chunk001",
                    "chunk_index": 0,
                    "document_id": "author2-doc",
                    "author": "author-two",
                    "source": "https://example.com/author2",
                }
            ),
            Document(
                page_content="Second author's continued discussion.",
                metadata={
                    "chunk_id": "author2_chunk002",
                    "chunk_index": 1,
                    "document_id": "author2-doc",
                    "author": "author-two",
                    "source": "https://example.com/author2",
                }
            ),
        ]

        # Embed and store all chunks
        embed_and_store(
            chunks=chunks,
            persist_dir=db_path,
            embeddings=embeddings
        )

        # Test 1: Retrieve only author-one's documents
        retriever_author1 = build_retriever(
            persist_dir=db_path,
            embeddings=embeddings,
            k=5,
            author="author-one"
        )

        results_author1 = retriever_author1.invoke("philosophy")
        assert len(results_author1) == 2
        assert all(doc.metadata["author"] == "author-one" for doc in results_author1)

        # Test 2: Retrieve only author-two's documents
        retriever_author2 = build_retriever(
            persist_dir=db_path,
            embeddings=embeddings,
            k=5,
            author="author-two"
        )

        results_author2 = retriever_author2.invoke("philosophy")
        assert len(results_author2) == 2
        assert all(doc.metadata["author"] == "author-two" for doc in results_author2)

        # Test 3: Retrieve without filter (should get all)
        retriever_all = build_retriever(
            persist_dir=db_path,
            embeddings=embeddings,
            k=5,
            author=None
        )

        results_all = retriever_all.invoke("philosophy")
        assert len(results_all) == 4


def test_retriever_persistence_across_sessions() -> None:
    """Integration test verifying retriever works across separate sessions.

    This simulates the real-world scenario where:
    1. Documents are ingested and stored in one session
    2. Later, a retriever is created to query the stored data
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir)
        embeddings = FakeEmbeddings()

        # Session 1: Ingest documents
        chunks = [
            Document(
                page_content="Persistent content that should survive.",
                metadata={
                    "chunk_id": "persist_001",
                    "chunk_index": 0,
                    "document_id": "persistent-doc",
                    "author": "test-author",
                    "source": "https://example.com",
                }
            ),
        ]

        embed_and_store(
            chunks=chunks,
            persist_dir=db_path,
            embeddings=embeddings
        )

        # Session 2: Create new retriever (simulating separate execution)
        retriever = build_retriever(
            persist_dir=db_path,
            embeddings=embeddings,
            k=5
        )

        # Should retrieve the previously stored document
        results = retriever.invoke("persistent")
        assert len(results) == 1
        assert results[0].metadata["chunk_id"] == "persist_001"
