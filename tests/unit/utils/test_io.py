"""Unit tests for IO utilities."""

import json
import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.utils.io import save_documents_to_disk


class TestSaveDocumentsToDisk:
    """Test save_documents_to_disk function."""

    def test_saves_documents_to_json_files(self) -> None:
        """Test that documents are correctly saved as JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create sample documents
            docs = [
                Document(
                    page_content="Content of page 1",
                    metadata={"page_number": 1, "author": "test"}
                ),
                Document(
                    page_content="Content of page 2",
                    metadata={"page_number": 2, "author": "test"}
                ),
            ]

            # Save documents
            saved_paths = save_documents_to_disk(docs, tmppath)

            assert len(saved_paths) == 2
            assert all(p.exists() for p in saved_paths)
            assert (tmppath / "page_01.json").exists()
            assert (tmppath / "page_02.json").exists()

            # Verify content
            with open(tmppath / "page_01.json") as f:
                data = json.load(f)
                assert data["page_content"] == "Content of page 1"
                assert data["metadata"]["page_number"] == 1
                assert data["metadata"]["author"] == "test"

    def test_creates_directory_if_not_exists(self) -> None:
        """Test that directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir) / "new" / "nested" / "dir"
            assert not tmppath.exists()

            doc = Document(
                page_content="Test content",
                metadata={"page_number": 1}
            )

            save_documents_to_disk([doc], tmppath)

            assert tmppath.exists()
            assert (tmppath / "page_01.json").exists()

    def test_raises_error_for_missing_page_number(self) -> None:
        """Test that error is raised if page_number is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            doc = Document(
                page_content="Test content",
                metadata={"author": "test"}  # Missing page_number
            )

            with pytest.raises(ValueError) as exc_info:
                save_documents_to_disk([doc], tmpdir)

            assert "missing 'page_number'" in str(exc_info.value)

    def test_handles_empty_document_list(self) -> None:
        """Test that empty document list is handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_paths = save_documents_to_disk([], tmpdir)
            assert saved_paths == []

    def test_formats_page_numbers_with_padding(self) -> None:
        """Test that page numbers are zero-padded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            docs = [
                Document(
                    page_content="Page 5",
                    metadata={"page_number": 5}
                ),
                Document(
                    page_content="Page 10",
                    metadata={"page_number": 10}
                ),
            ]

            save_documents_to_disk(docs, tmppath)

            assert (tmppath / "page_05.json").exists()
            assert (tmppath / "page_10.json").exists()

    def test_preserves_unicode_content(self) -> None:
        """Test that Unicode content is preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            doc = Document(
                page_content="Français: « liberté, égalité »",
                metadata={"page_number": 1, "title": "Les Lumières"}
            )

            save_documents_to_disk([doc], tmppath)

            with open(tmppath / "page_01.json", encoding="utf-8") as f:
                data = json.load(f)
                assert "Français" in data["page_content"]
                assert "« liberté, égalité »" in data["page_content"]
                assert data["metadata"]["title"] == "Les Lumières"
