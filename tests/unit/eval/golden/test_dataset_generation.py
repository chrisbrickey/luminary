"""Tests for golden dataset generation infrastructure.

This module tests the LLM-based golden dataset generation system that creates
high-quality evaluation examples by auto-discovering required fields and using
an LLM to make judgments about chunk relevance and expected values.
"""

from unittest.mock import Mock, patch

import pytest

from src.eval.golden.dataset_generation import (
    build_field_guidance,
    build_prompt,
    discover_required_fields,
    generate_golden_example_with_llm,
    retrieve_candidate_chunks,
)
from src.schemas.eval import GoldenExample


class TestDiscoverRequiredFields:
    """Test auto-discovery of required fields from schema and metrics."""

    def test_discover_required_fields_includes_schema_fields(self) -> None:
        """Auto-discovery finds fields from GoldenExample schema."""
        # GoldenExample has expected_chunk_ids and expected_source_titles
        discovered = discover_required_fields()

        # Should include fields from the schema (non-core fields)
        assert "expected_chunk_ids" in discovered
        assert "expected_source_titles" in discovered

    def test_discover_required_fields_includes_metric_fields(self) -> None:
        """Auto-discovery finds fields from METRIC_REGISTRY."""
        # METRIC_REGISTRY contains metrics that declare required_example_fields
        # For example, retrieval_relevance requires "expected_chunk_ids"
        discovered = discover_required_fields()

        # Should include fields that metrics declare as required
        assert "expected_chunk_ids" in discovered
        assert "expected_source_titles" in discovered

    def test_discover_required_fields_excludes_core_fields(self) -> None:
        """Auto-discovery excludes core input fields."""
        # Core fields (id, question, author, language) are inputs, not LLM judgments
        discovered = discover_required_fields()

        # Should NOT include core fields
        assert "id" not in discovered
        assert "question" not in discovered
        assert "author" not in discovered
        assert "language" not in discovered


class TestBuildFieldGuidance:
    """Test LLM guidance generation for each field type."""

    def test_build_field_guidance_includes_known_fields(self) -> None:
        """Guidance for known fields is specific and detailed."""
        # Test guidance for currently-implemented metric fields
        chunk_ids_guidance = build_field_guidance("expected_chunk_ids")
        source_titles_guidance = build_field_guidance("expected_source_titles")

        # Should contain specific guidance with examples
        assert len(chunk_ids_guidance) > 50  # Detailed guidance
        assert "chunk" in chunk_ids_guidance.lower()
        assert "expected_chunk_ids" in chunk_ids_guidance

        assert len(source_titles_guidance) > 50  # Detailed guidance
        assert "title" in source_titles_guidance.lower()
        assert "expected_source_titles" in source_titles_guidance

    def test_build_field_guidance_handles_unknown_fields(self) -> None:
        """Guidance for unknown fields provides generic fallback."""
        # Test with a field that doesn't have specific guidance
        unknown_guidance = build_field_guidance("some_future_field")

        # Should return generic fallback guidance
        assert len(unknown_guidance) > 0
        assert "some_future_field" in unknown_guidance


class TestBuildPrompt:
    """Test prompt assembly with auto-discovered fields."""

    def test_build_prompt_includes_all_required_fields(self) -> None:
        """Assembled prompt includes guidance for all discovered fields."""
        question = "What is tolerance?"
        author = "voltaire"
        required_fields = {"expected_chunk_ids", "expected_source_titles"}

        prompt = build_prompt(question, author, required_fields)

        # Should include the question and author
        assert question in prompt
        assert author in prompt

        # Should include guidance for all required fields
        assert "expected_chunk_ids" in prompt
        assert "expected_source_titles" in prompt

        # Should instruct LLM to return JSON
        assert "json" in prompt.lower() or "JSON" in prompt


class TestRetrieveCandidateChunks:
    """Test candidate chunk retrieval for LLM judgment."""

    def test_retrieve_candidate_chunks_returns_metadata(self) -> None:
        """Retrieval returns k=15 chunks with required metadata."""
        # Mock retriever
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = [
            Mock(
                metadata={
                    "chunk_id": f"chunk_{i}",
                    "document_title": f"Document {i}",
                },
                page_content=f"Content for chunk {i}",
            )
            for i in range(15)
        ]

        with patch("src.eval.golden.dataset_generation.build_retriever", return_value=mock_retriever):
            chunks = retrieve_candidate_chunks(
                question="What is tolerance?",
                author="voltaire",
                k=15,
            )

        # Should return 15 chunks (more than production k=5)
        assert len(chunks) == 15

        # Each chunk should have required metadata
        for i, chunk in enumerate(chunks):
            assert "chunk_id" in chunk
            assert "source" in chunk
            assert "text" in chunk
            assert chunk["chunk_id"] == f"chunk_{i}"
            assert chunk["source"] == f"Document {i}"
            assert chunk["text"] == f"Content for chunk {i}"


class TestGenerateGoldenExampleWithLLM:
    """Test LLM-based generation of validated GoldenExample instances."""

    def test_generate_golden_example_returns_valid_schema(self) -> None:
        """LLM generation returns validated GoldenExample."""
        # Mock LLM to return valid JSON
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(
            content='{"id": "test_tolerance_fr", "question": "Qu\'est-ce que la tolérance?", "author": "voltaire", "language": "fr", "expected_chunk_ids": ["chunk_123"], "expected_source_titles": ["Lettres philosophiques"]}'
        )

        # Mock retriever
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = [
            Mock(
                metadata={
                    "chunk_id": "chunk_123",
                    "document_title": "Lettres philosophiques",
                },
                page_content="Content about tolerance",
            )
        ]

        with patch("src.eval.golden.dataset_generation.build_retriever", return_value=mock_retriever):
            example = generate_golden_example_with_llm(
                question="Qu'est-ce que la tolérance?",
                author="voltaire",
                language="fr",
                llm=mock_llm,
                retriever=mock_retriever,
            )

        # Should return a valid GoldenExample instance
        assert isinstance(example, GoldenExample)
        assert example.id == "test_tolerance_fr"
        assert example.question == "Qu'est-ce que la tolérance?"
        assert example.author == "voltaire"
        assert example.language == "fr"
        assert example.expected_chunk_ids == ["chunk_123"]
        assert example.expected_source_titles == ["Lettres philosophiques"]

    def test_generate_golden_example_validates_schema(self) -> None:
        """Invalid LLM output raises validation error."""
        # Mock LLM to return invalid JSON (missing required fields)
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(
            content='{"id": "test_1", "question": "Test?"}'  # Missing author, language
        )

        # Mock retriever
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = []

        with patch("src.eval.golden.dataset_generation.build_retriever", return_value=mock_retriever):
            # Should raise validation error when LLM returns invalid JSON
            with pytest.raises(Exception):  # Pydantic ValidationError
                generate_golden_example_with_llm(
                    question="Test?",
                    author="voltaire",
                    language="fr",
                    llm=mock_llm,
                    retriever=mock_retriever,
                )
