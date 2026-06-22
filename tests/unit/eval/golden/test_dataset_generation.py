"""Unit tests for golden dataset generation infrastructure.

This module tests the LLM-based golden dataset generation system that creates
high-quality evaluation examples by auto-discovering required fields and using
an LLM to make judgments about chunk relevance and expected values.

Generation runs two LLM passes:
- Pass 1: produce the JSON example with primary keywords (no synonyms)
- Pass 2: produce 2-4 single-word synonyms per primary, grounded in the same chunks
"""

from unittest.mock import Mock, patch

import pytest

from tests.fake_authors import FAKE_AUTHOR_A
from src.eval.golden.dataset_generation import (
    build_field_guidance,
    build_prompt,
    build_synonyms_prompt,
    discover_required_fields,
    generate_golden_example_with_llm,
    generate_synonyms_with_llm,
    retrieve_candidate_chunks,
)
from src.schemas.eval import GoldenExample

# -- Pytest fixtures --------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_authors(mock_author_configs):
    """Apply author mocking to all tests in this file."""


# -- Shared test constants --------------------------------------------------

AUTHOR = FAKE_AUTHOR_A
LANGUAGE_FR = "fr"
LANGUAGE_EN = "en"

QUESTION_FR = "Quelle est la question?"
QUESTION_EN = "What is the question?"

PRIMARY_A = "alpha"
PRIMARY_B = "beta"

SAMPLE_CHUNK = {
    "chunk_id": "chunk_123",
    "source": "Sample Document",
    "text": "Content about sample topic.",
}


def _build_pass1_json(primaries: list[str]) -> str:
    """Render a minimal GoldenExample JSON string with the given primaries."""
    primary_entries = ", ".join(
        f'{{"primary": "{p}"}}' for p in primaries
    )
    return (
        '{"id": "test_example_fr", '
        f'"question": "{QUESTION_FR}", '
        f'"author": "{AUTHOR}", '
        f'"language": "{LANGUAGE_FR}", '
        '"expected_chunk_ids": ["chunk_123"], '
        '"expected_source_titles": ["Sample Document"], '
        f'"expected_keywords": [{primary_entries}]'
        '}'
    )


def _mock_chunk(chunk_id: str = "chunk_123", title: str = "Sample Document",
                content: str = "Content about sample topic.") -> Mock:
    return Mock(
        metadata={"chunk_id": chunk_id, "document_title": title},
        page_content=content,
    )


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

    def test_build_field_guidance_for_keywords_omits_default_empty_synonyms(self) -> None:
        """Pass-1 guidance must not instruct the LLM to default synonyms to []."""
        guidance = build_field_guidance("expected_keywords")

        assert "default" not in guidance.lower() or "[]" not in guidance


class TestBuildPrompt:
    """Test prompt assembly with auto-discovered fields."""

    def test_build_prompt_includes_all_required_fields(self) -> None:
        """Assembled prompt includes guidance for all discovered fields."""
        required_fields = {"expected_chunk_ids", "expected_source_titles"}

        prompt = build_prompt(QUESTION_EN, AUTHOR, required_fields)

        # Should include the question and author
        assert QUESTION_EN in prompt
        assert AUTHOR in prompt

        # Should include guidance for all required fields
        assert "expected_chunk_ids" in prompt
        assert "expected_source_titles" in prompt

        # Should instruct LLM to return JSON
        assert "json" in prompt.lower() or "JSON" in prompt


class TestBuildSynonymsPrompt:
    """Tests for the pass-2 synonyms prompt assembly."""

    def test_includes_each_primary_in_output_template(self) -> None:
        """Each primary appears as a key in the example output object."""
        prompt = build_synonyms_prompt(
            primaries=[PRIMARY_A, PRIMARY_B],
            language=LANGUAGE_EN,
            chunks=[SAMPLE_CHUNK],
        )

        assert f'"{PRIMARY_A}"' in prompt
        assert f'"{PRIMARY_B}"' in prompt

    def test_includes_language_and_chunks(self) -> None:
        """Prompt mentions the requested language and embeds the candidate chunks."""
        prompt = build_synonyms_prompt(
            primaries=[PRIMARY_A],
            language=LANGUAGE_FR,
            chunks=[SAMPLE_CHUNK],
        )

        assert f"**Language**: {LANGUAGE_FR}" in prompt
        assert SAMPLE_CHUNK["chunk_id"] in prompt
        assert SAMPLE_CHUNK["text"] in prompt

    def test_forbids_multi_word_synonyms(self) -> None:
        """Pass-2 prompt instructs the LLM to emit single-word synonyms only."""
        prompt = build_synonyms_prompt(
            primaries=[PRIMARY_A],
            language=LANGUAGE_EN,
            chunks=[SAMPLE_CHUNK],
        )

        # Either an explicit "SINGLE word" requirement or a "no spaces" rule is acceptable.
        assert "single word" in prompt.lower() or "single-word" in prompt.lower()


class TestRetrieveCandidateChunks:
    """Test candidate chunk retrieval for LLM judgment."""

    def test_retrieve_candidate_chunks_returns_metadata(self) -> None:
        """Retrieval returns k=15 chunks with required metadata."""
        # Mock retriever
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = [
            _mock_chunk(chunk_id=f"chunk_{i}", title=f"Document {i}", content=f"Content for chunk {i}")
            for i in range(15)
        ]

        with patch(
            "src.eval.golden.dataset_generation.build_retriever",
            return_value=mock_retriever,
        ):
            chunks = retrieve_candidate_chunks(
                question=QUESTION_EN,
                author=AUTHOR,
                k=15,
            )

        # Should return 15 chunks (more than production k=5)
        assert len(chunks) == 15

        # Each chunk should have required metadata
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_id"] == f"chunk_{i}"
            assert chunk["source"] == f"Document {i}"
            assert chunk["text"] == f"Content for chunk {i}"


class TestGenerateSynonymsWithLLM:
    """Tests for the pass-2 synonym generator."""

    def test_returns_synonyms_keyed_by_primary(self) -> None:
        """LLM JSON keyed by primary is parsed into a {primary: [synonyms]} dict."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(
            content=f'{{"{PRIMARY_A}": ["syn_a1", "syn_a2"], "{PRIMARY_B}": ["syn_b1", "syn_b2"]}}'
        )

        result = generate_synonyms_with_llm(
            primaries=[PRIMARY_A, PRIMARY_B],
            language=LANGUAGE_EN,
            chunks=[SAMPLE_CHUNK],
            llm=mock_llm,
        )

        assert result == {
            PRIMARY_A: ["syn_a1", "syn_a2"],
            PRIMARY_B: ["syn_b1", "syn_b2"],
        }

    def test_filters_multi_word_synonyms(self) -> None:
        """Multi-word entries are dropped silently; single words are kept."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(
            content=f'{{"{PRIMARY_A}": ["syn_a1", "two words", "syn_a2"]}}'
        )

        result = generate_synonyms_with_llm(
            primaries=[PRIMARY_A],
            language=LANGUAGE_EN,
            chunks=[SAMPLE_CHUNK],
            llm=mock_llm,
        )

        assert result == {PRIMARY_A: ["syn_a1", "syn_a2"]}

    def test_drops_primary_from_its_own_synonym_list(self) -> None:
        """The primary itself (case-insensitive) is never carried as a synonym."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(
            content=f'{{"{PRIMARY_A}": ["{PRIMARY_A.capitalize()}", "syn_a1"]}}'
        )

        result = generate_synonyms_with_llm(
            primaries=[PRIMARY_A],
            language=LANGUAGE_EN,
            chunks=[SAMPLE_CHUNK],
            llm=mock_llm,
        )

        assert result == {PRIMARY_A: ["syn_a1"]}

    def test_caps_synonyms_per_primary(self) -> None:
        """No more than four synonyms are returned per primary, even if the LLM returns more."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(
            content=f'{{"{PRIMARY_A}": ["a", "b", "c", "d", "e", "f"]}}'
        )

        result = generate_synonyms_with_llm(
            primaries=[PRIMARY_A],
            language=LANGUAGE_EN,
            chunks=[SAMPLE_CHUNK],
            llm=mock_llm,
        )

        assert result[PRIMARY_A] == ["a", "b", "c", "d"]

    def test_missing_primary_key_returns_empty_list(self) -> None:
        """Primaries omitted by the LLM are mapped to []."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(
            content=f'{{"{PRIMARY_A}": ["syn_a1"]}}'  # PRIMARY_B missing
        )

        result = generate_synonyms_with_llm(
            primaries=[PRIMARY_A, PRIMARY_B],
            language=LANGUAGE_EN,
            chunks=[SAMPLE_CHUNK],
            llm=mock_llm,
        )

        assert result == {
            PRIMARY_A: ["syn_a1"],
            PRIMARY_B: [],
        }

    def test_empty_primaries_skips_llm_call(self) -> None:
        """No primaries → no LLM call; returns empty dict."""
        mock_llm = Mock()

        result = generate_synonyms_with_llm(
            primaries=[],
            language=LANGUAGE_EN,
            chunks=[SAMPLE_CHUNK],
            llm=mock_llm,
        )

        assert result == {}
        mock_llm.invoke.assert_not_called()

    def test_invalid_json_raises(self) -> None:
        """Malformed LLM JSON raises ValueError."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="not json")

        with pytest.raises(ValueError, match="invalid JSON"):
            generate_synonyms_with_llm(
                primaries=[PRIMARY_A],
                language=LANGUAGE_EN,
                chunks=[SAMPLE_CHUNK],
                llm=mock_llm,
            )

    def test_non_object_json_raises(self) -> None:
        """Top-level JSON that is not an object (e.g., array) is rejected."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content='["syn_a1", "syn_a2"]')

        with pytest.raises(ValueError, match="non-object"):
            generate_synonyms_with_llm(
                primaries=[PRIMARY_A],
                language=LANGUAGE_EN,
                chunks=[SAMPLE_CHUNK],
                llm=mock_llm,
            )


class TestGenerateGoldenExampleWithLLM:
    """Test LLM-based generation of validated GoldenExample instances."""

    def test_generate_golden_example_returns_valid_schema_when_no_primaries(self) -> None:
        """When pass 1 emits zero primaries, no pass-2 call happens and the example validates."""
        # Mock LLM to return valid JSON
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content=_build_pass1_json(primaries=[]))

        # Mock retriever
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = [_mock_chunk()]

        with patch(
            "src.eval.golden.dataset_generation.build_retriever",
            return_value=mock_retriever,
        ):
            example = generate_golden_example_with_llm(
                question=QUESTION_FR,
                author=AUTHOR,
                language=LANGUAGE_FR,
                llm=mock_llm,
                retriever=mock_retriever,
            )

        # Should return a valid GoldenExample instance
        assert isinstance(example, GoldenExample)
        assert example.id == "test_example_fr"
        assert example.question == QUESTION_FR
        assert example.author == AUTHOR
        assert example.language == LANGUAGE_FR
        assert example.expected_chunk_ids == ["chunk_123"]
        assert example.expected_source_titles == ["Sample Document"]
        assert example.expected_keywords == []
        # No primaries → only the pass-1 LLM call happens.
        assert mock_llm.invoke.call_count == 1

    def test_generate_golden_example_runs_second_pass_for_synonyms(self) -> None:
        """When pass 1 emits primaries, pass 2 runs and synonyms are merged into the example."""
        pass1_content = _build_pass1_json(primaries=[PRIMARY_A, PRIMARY_B])
        pass2_content = (
            f'{{"{PRIMARY_A}": ["syn_a1", "syn_a2"], '
            f'"{PRIMARY_B}": ["syn_b1", "syn_b2"]}}'
        )

        mock_llm = Mock()
        mock_llm.invoke.side_effect = [
            Mock(content=pass1_content),
            Mock(content=pass2_content),
        ]

        mock_retriever = Mock()
        mock_retriever.invoke.return_value = [_mock_chunk()]

        with patch(
            "src.eval.golden.dataset_generation.build_retriever",
            return_value=mock_retriever,
        ):
            example = generate_golden_example_with_llm(
                question=QUESTION_FR,
                author=AUTHOR,
                language=LANGUAGE_FR,
                llm=mock_llm,
                retriever=mock_retriever,
            )

        assert isinstance(example, GoldenExample)
        assert [k.primary for k in example.expected_keywords] == [PRIMARY_A, PRIMARY_B]
        assert example.expected_keywords[0].synonyms == ["syn_a1", "syn_a2"]
        assert example.expected_keywords[1].synonyms == ["syn_b1", "syn_b2"]
        assert mock_llm.invoke.call_count == 2

    def test_generate_golden_example_validates_schema(self) -> None:
        """Invalid pass-1 JSON raises (missing required fields)."""
        # Mock LLM to return invalid JSON (missing required fields)
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(
            content='{"id": "test_1", "question": "Test?"}'  # Missing author, language
        )

        # Mock retriever
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = []

        with patch(
            "src.eval.golden.dataset_generation.build_retriever",
            return_value=mock_retriever,
        ):
            # Should raise validation error when LLM returns invalid JSON
            with pytest.raises(Exception):  # Pydantic ValidationError
                generate_golden_example_with_llm(
                    question="Test?",
                    author=AUTHOR,
                    language=LANGUAGE_FR,
                    llm=mock_llm,
                    retriever=mock_retriever,
                )
