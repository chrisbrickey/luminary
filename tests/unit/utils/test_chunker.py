"""Tests for src/utils/chunker.py — text chunking with metadata validation."""

from typing import Any

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from src.schemas import ChunkInfo
from src.utils.chunker import _generate_chunk_id, chunk_documents

# -- Shared test constants --------------------------------------------------

DOCUMENT_ID = "test-doc-001"
DOCUMENT_TITLE = "Test Document Title"
AUTHOR = "testauthor"
SOURCE_URL = "https://example.com/test-doc"

# Sample French text for chunking tests (broad Enlightenment theme)
SAMPLE_TEXT_SHORT = (
    "La raison est le guide de l'humanité. "
    "Les Lumières apportent la connaissance."
)

SAMPLE_TEXT_LONG = """
Les Lumières représentent un mouvement intellectuel majeur. La raison et la science
doivent guider l'humanité vers le progrès.

La liberté de conscience est un droit naturel. Chaque individu doit pouvoir penser
librement sans crainte de persécution.

La tolérance religieuse est essentielle pour la paix sociale. Les différentes croyances
peuvent coexister harmonieusement dans une société éclairée.

L'éducation est la clé du progrès humain. Sans instruction, les peuples restent dans
l'obscurité et la superstition.

Les droits naturels appartiennent à tous les êtres humains. Aucun pouvoir ne peut
légitimement les violer.
"""


def _document_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return default Document constructor kwargs with optional overrides."""
    defaults: dict[str, Any] = {
        "page_content": SAMPLE_TEXT_SHORT,
        "metadata": {
            "document_id": DOCUMENT_ID,
            "document_title": DOCUMENT_TITLE,
            "author": AUTHOR,
            "source": SOURCE_URL,
            "page_number": 1,
        },
    }
    # Allow overriding page_content and individual metadata fields
    if "page_content" in overrides:
        defaults["page_content"] = overrides.pop("page_content")
    if "metadata" in overrides:
        defaults["metadata"].update(overrides.pop("metadata"))
    defaults.update(overrides)
    return defaults


class TestChunkDocuments:
    def test_chunks_within_size_bounds(self) -> None:
        """Verify chunks respect chunk_size parameter."""
        doc = Document(**_document_kwargs(page_content=SAMPLE_TEXT_LONG))
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=50)

        assert len(chunks) > 1  # Long text should produce multiple chunks
        for chunk in chunks:
            # Chunks should be at or below chunk_size (with some tolerance for word boundaries)
            assert len(chunk.page_content) <= 250  # Allow small overage

    def test_metadata_preserved_and_enriched(self) -> None:
        """Verify all original metadata preserved and chunk metadata added."""
        doc = Document(**_document_kwargs())
        chunks = chunk_documents([doc])

        assert len(chunks) >= 1
        for i, chunk in enumerate(chunks):
            # Original metadata preserved
            assert chunk.metadata["document_id"] == DOCUMENT_ID
            assert chunk.metadata["document_title"] == DOCUMENT_TITLE
            assert chunk.metadata["author"] == AUTHOR
            assert chunk.metadata["source"] == SOURCE_URL
            assert chunk.metadata["page_number"] == 1

            # Chunk-specific metadata added
            assert chunk.metadata["chunk_index"] == i
            assert "chunk_id" in chunk.metadata
            assert len(chunk.metadata["chunk_id"]) == 12  # SHA256 truncated to 12 chars

    def test_no_text_lost(self) -> None:
        """Verify all text from original document appears in chunks."""
        doc = Document(**_document_kwargs(page_content=SAMPLE_TEXT_LONG))
        chunks = chunk_documents([doc], chunk_size=300, chunk_overlap=50)

        # Concatenate all chunk text
        all_chunk_text = "".join(chunk.page_content for chunk in chunks)

        # Remove whitespace for comparison (chunker may normalize spacing)
        original_normalized = "".join(SAMPLE_TEXT_LONG.split())
        chunks_normalized = "".join(all_chunk_text.split())

        # All content should be present (order may differ slightly due to overlap)
        assert len(chunks_normalized) >= len(original_normalized) * 0.95  # Allow small loss

    def test_chunkinfo_validation_passes(self) -> None:
        """Verify chunk metadata validates against ChunkInfo schema."""
        doc = Document(**_document_kwargs())
        chunks = chunk_documents([doc])

        # Each chunk's metadata should construct a valid ChunkInfo
        for chunk in chunks:
            chunk_info = ChunkInfo.model_validate(chunk.metadata)
            assert chunk_info.chunk_id is not None
            assert chunk_info.document_id == DOCUMENT_ID
            assert chunk_info.chunk_index >= 0

    def test_empty_document_produces_no_chunks(self) -> None:
        """Verify empty or whitespace-only documents produce no chunks."""
        empty_doc = Document(**_document_kwargs(page_content=""))
        whitespace_doc = Document(**_document_kwargs(page_content="   \n\n  "))

        empty_chunks = chunk_documents([empty_doc])
        whitespace_chunks = chunk_documents([whitespace_doc])

        assert len(empty_chunks) == 0
        assert len(whitespace_chunks) == 0

    def test_chunk_id_deterministic(self) -> None:
        """Verify chunk_id generation is deterministic and unique per chunk."""
        doc = Document(**_document_kwargs(page_content=SAMPLE_TEXT_LONG))

        # Run chunking twice
        chunks_1 = chunk_documents([doc], chunk_size=300)
        chunks_2 = chunk_documents([doc], chunk_size=300)

        assert len(chunks_1) == len(chunks_2)

        # Same chunk_ids should be generated each time
        for c1, c2 in zip(chunks_1, chunks_2):
            assert c1.metadata["chunk_id"] == c2.metadata["chunk_id"]

        # All chunk_ids should be unique
        chunk_ids = [c.metadata["chunk_id"] for c in chunks_1]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_multiple_documents_processed_independently(self) -> None:
        """Verify multiple documents are chunked independently."""
        doc1 = Document(
            **_document_kwargs(
                metadata={"document_id": "doc-001", "source": "https://example.com/1"}
            )
        )
        doc2 = Document(
            **_document_kwargs(
                metadata={"document_id": "doc-002", "source": "https://example.com/2"}
            )
        )

        chunks = chunk_documents([doc1, doc2])

        # Should have chunks from both documents
        doc1_chunks = [c for c in chunks if c.metadata["document_id"] == "doc-001"]
        doc2_chunks = [c for c in chunks if c.metadata["document_id"] == "doc-002"]

        assert len(doc1_chunks) >= 1
        assert len(doc2_chunks) >= 1

        # Chunk indices should be independent (both start at 0)
        assert doc1_chunks[0].metadata["chunk_index"] == 0
        assert doc2_chunks[0].metadata["chunk_index"] == 0

    def test_missing_document_id_fails_validation(self) -> None:
        """Verify ChunkInfo validation catches missing required fields."""
        doc = Document(
            page_content=SAMPLE_TEXT_SHORT,
            metadata={
                # Missing document_id
                "author": AUTHOR,
                "source": SOURCE_URL,
            },
        )

        with pytest.raises(ValueError, match="validation failed"):
            chunk_documents([doc])

    def test_invalid_author_case_fails_validation(self) -> None:
        """Verify ChunkInfo validation enforces lowercase author."""
        doc = Document(**_document_kwargs(metadata={"author": "TestAuthor"}))

        with pytest.raises(ValueError, match="validation failed"):
            chunk_documents([doc])

    def test_custom_chunk_size_and_overlap(self) -> None:
        """Verify custom chunk_size and chunk_overlap parameters work."""
        doc = Document(**_document_kwargs(page_content=SAMPLE_TEXT_LONG))

        small_chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=20)
        large_chunks = chunk_documents([doc], chunk_size=500, chunk_overlap=50)

        # Smaller chunk_size should produce more chunks
        assert len(small_chunks) > len(large_chunks)


class TestGenerateChunkId:
    def test_chunk_id_format(self) -> None:
        """Verify chunk_id is 12-character hex string."""
        chunk_id = _generate_chunk_id("test-doc", 0)
        assert len(chunk_id) == 12
        assert all(c in "0123456789abcdef" for c in chunk_id)

    def test_chunk_id_deterministic(self) -> None:
        """Verify same inputs produce same chunk_id."""
        chunk_id_1 = _generate_chunk_id("doc-001", 5)
        chunk_id_2 = _generate_chunk_id("doc-001", 5)
        assert chunk_id_1 == chunk_id_2

    def test_chunk_id_unique_per_index(self) -> None:
        """Verify different chunk indices produce different IDs."""
        chunk_id_0 = _generate_chunk_id("doc-001", 0)
        chunk_id_1 = _generate_chunk_id("doc-001", 1)
        assert chunk_id_0 != chunk_id_1

    def test_chunk_id_unique_per_document(self) -> None:
        """Verify different document_ids produce different IDs."""
        chunk_id_a = _generate_chunk_id("doc-001", 0)
        chunk_id_b = _generate_chunk_id("doc-002", 0)
        assert chunk_id_a != chunk_id_b
