"""Unit tests for src/eval/golden/rechunk.py.

The rechunk module produces a new `expected_chunk_ids` list for a GoldenExample
after the underlying corpus has been re-chunked. Tests cover:
- Prompt construction (chunk IDs and question surface, no keyword/synonym leak)
- LLM response handling
- Field preservation when merging new chunk IDs back into a GoldenExample
"""

import logging
from unittest.mock import Mock, patch

import pytest

from tests.fake_authors import FAKE_AUTHOR_A
from src.eval.golden.rechunk import (
    build_chunk_id_prompt,
    extract_json_object,
    rechunk_example,
    regenerate_expected_chunk_ids,
)
from src.schemas.eval import GoldenExample, KeywordEntry


# -- Pytest fixtures --------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_authors(mock_author_configs):
    """Apply author mocking to all tests in this file."""


# -- Shared test constants --------------------------------------------------

AUTHOR = FAKE_AUTHOR_A
LANGUAGE_EN = "en"

QUESTION = "What is the question?"
EXAMPLE_ID = "topic_en"

CHUNK_ID_A = "aaaa11112222"
CHUNK_ID_B = "bbbb33334444"
CHUNK_ID_C = "cccc55556666"

SAMPLE_CHUNK = {
    "chunk_id": CHUNK_ID_A,
    "source": "Sample Document",
    "text": "Content about the sample topic.",
}


def _mock_chunk(chunk_id: str, source: str = "Sample Document",
                text: str = "Sample text.") -> dict[str, str]:
    """Return a chunk dict in the shape returned by retrieve_candidate_chunks."""
    return {"chunk_id": chunk_id, "source": source, "text": text}


def _build_example(chunk_ids: list[str] | None = None,
                   source_titles: list[str] | None = None,
                   keywords: list[KeywordEntry] | None = None) -> GoldenExample:
    """Build a GoldenExample with reasonable defaults for merge-preservation tests."""
    return GoldenExample(
        id=EXAMPLE_ID,
        question=QUESTION,
        author=AUTHOR,
        language=LANGUAGE_EN,
        expected_chunk_ids=chunk_ids if chunk_ids is not None else [CHUNK_ID_A],
        expected_source_titles=source_titles if source_titles is not None else ["Original Title"],
        expected_keywords=keywords if keywords is not None else [
            KeywordEntry(primary="alpha", synonyms=["alpha_syn_1", "alpha_syn_2"]),
        ],
    )


def _mock_llm_returning(content: str) -> Mock:
    """Mock LLM whose invoke() returns a message with the given string content."""
    llm = Mock()
    llm.invoke.return_value = Mock(content=content)
    return llm


# -- Tests ------------------------------------------------------------------


class TestExtractJsonObject:
    """Tests for the markdown-fence-tolerant JSON extractor."""

    def test_returns_content_unchanged_when_no_fence(self) -> None:
        """Plain JSON is returned as-is (stripped)."""
        raw = '{"expected_chunk_ids": ["a", "b"]}'
        assert extract_json_object(raw) == raw

    def test_unwraps_json_code_fence(self) -> None:
        """```json ... ``` fences are stripped."""
        raw = '```json\n{"expected_chunk_ids": ["a"]}\n```'
        assert extract_json_object(raw) == '{"expected_chunk_ids": ["a"]}'

    def test_unwraps_bare_code_fence(self) -> None:
        """Fences without a language tag are also stripped."""
        raw = '```\n{"expected_chunk_ids": ["a"]}\n```'
        assert extract_json_object(raw) == '{"expected_chunk_ids": ["a"]}'


class TestBuildChunkIdPrompt:
    """Tests for prompt construction. The prompt is scoped to chunk_ids only."""

    def test_prompt_includes_question_and_author(self) -> None:
        """The prompt surfaces the question and author for LLM context."""
        prompt = build_chunk_id_prompt(QUESTION, AUTHOR, [SAMPLE_CHUNK])
        assert QUESTION in prompt
        assert AUTHOR in prompt

    def test_prompt_includes_every_candidate_chunk_id(self) -> None:
        """Each candidate chunk_id appears verbatim in the prompt."""
        chunks = [
            _mock_chunk(CHUNK_ID_A),
            _mock_chunk(CHUNK_ID_B),
            _mock_chunk(CHUNK_ID_C),
        ]
        prompt = build_chunk_id_prompt(QUESTION, AUTHOR, chunks)
        for chunk_id in (CHUNK_ID_A, CHUNK_ID_B, CHUNK_ID_C):
            assert chunk_id in prompt

    def test_prompt_scopes_output_to_expected_chunk_ids_only(self) -> None:
        """The prompt asks for only the expected_chunk_ids field, no keywords/synonyms/source titles."""
        prompt = build_chunk_id_prompt(QUESTION, AUTHOR, [SAMPLE_CHUNK])
        assert "expected_chunk_ids" in prompt
        # The prompt must NOT ask the LLM to also produce these other fields.
        assert "expected_source_titles" not in prompt
        assert "expected_keywords" not in prompt
        assert "synonyms" not in prompt

    def test_prompt_states_the_3_to_7_selection_range(self) -> None:
        """The prompt specifies how many chunks the LLM should select."""
        prompt = build_chunk_id_prompt(QUESTION, AUTHOR, [SAMPLE_CHUNK])
        assert "3-7" in prompt


class TestRegenerateExpectedChunkIds:
    """Tests for the single-pass LLM invocation.

    All tests patch retrieve_candidate_chunks at its usage site inside the
    rechunk module so no real vector store is touched.
    """

    def test_returns_ids_from_valid_llm_response(self) -> None:
        """Happy path: valid JSON with a list of chunk IDs is returned as-is."""
        llm = _mock_llm_returning(
            f'{{"expected_chunk_ids": ["{CHUNK_ID_A}", "{CHUNK_ID_B}"]}}'
        )
        example = _build_example()

        with patch(
            "src.eval.golden.rechunk.retrieve_candidate_chunks",
            return_value=[_mock_chunk(CHUNK_ID_A), _mock_chunk(CHUNK_ID_B)],
        ):
            result = regenerate_expected_chunk_ids(example=example, llm=llm)

        assert result == [CHUNK_ID_A, CHUNK_ID_B]
        assert llm.invoke.call_count == 1

    def test_unwraps_json_from_markdown_fence(self) -> None:
        """LLM responses wrapped in ```json ... ``` fences are parsed correctly."""
        llm = _mock_llm_returning(
            f'```json\n{{"expected_chunk_ids": ["{CHUNK_ID_A}"]}}\n```'
        )
        example = _build_example()

        with patch(
            "src.eval.golden.rechunk.retrieve_candidate_chunks",
            return_value=[_mock_chunk(CHUNK_ID_A)],
        ):
            result = regenerate_expected_chunk_ids(example=example, llm=llm)

        assert result == [CHUNK_ID_A]

    def test_empty_ids_returned_and_warning_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Empty list is accepted but logged as a warning for the caller to notice."""
        llm = _mock_llm_returning('{"expected_chunk_ids": []}')
        example = _build_example()

        with patch(
            "src.eval.golden.rechunk.retrieve_candidate_chunks",
            return_value=[_mock_chunk(CHUNK_ID_A)],
        ), caplog.at_level(logging.WARNING, logger="src.eval.golden.rechunk"):
            result = regenerate_expected_chunk_ids(example=example, llm=llm)

        assert result == []
        assert any(EXAMPLE_ID in rec.message and "empty" in rec.message.lower()
                   for rec in caplog.records)

    def test_malformed_json_raises_value_error(self) -> None:
        """Non-JSON LLM output raises ValueError with the example id in the message."""
        llm = _mock_llm_returning("this is not JSON")
        example = _build_example()

        with patch(
            "src.eval.golden.rechunk.retrieve_candidate_chunks",
            return_value=[_mock_chunk(CHUNK_ID_A)],
        ):
            with pytest.raises(ValueError, match=EXAMPLE_ID):
                regenerate_expected_chunk_ids(example=example, llm=llm)

    def test_missing_expected_chunk_ids_key_raises(self) -> None:
        """JSON object without the required field raises ValueError."""
        llm = _mock_llm_returning('{"some_other_field": ["a", "b"]}')
        example = _build_example()

        with patch(
            "src.eval.golden.rechunk.retrieve_candidate_chunks",
            return_value=[_mock_chunk(CHUNK_ID_A)],
        ):
            with pytest.raises(ValueError, match="expected_chunk_ids"):
                regenerate_expected_chunk_ids(example=example, llm=llm)

    def test_non_list_value_raises(self) -> None:
        """`expected_chunk_ids` must be a list, not a string or object."""
        llm = _mock_llm_returning('{"expected_chunk_ids": "not a list"}')
        example = _build_example()

        with patch(
            "src.eval.golden.rechunk.retrieve_candidate_chunks",
            return_value=[_mock_chunk(CHUNK_ID_A)],
        ):
            with pytest.raises(ValueError, match="non-string chunk_ids"):
                regenerate_expected_chunk_ids(example=example, llm=llm)

    def test_non_string_elements_raise(self) -> None:
        """List elements must all be strings."""
        llm = _mock_llm_returning('{"expected_chunk_ids": ["ok", 42, null]}')
        example = _build_example()

        with patch(
            "src.eval.golden.rechunk.retrieve_candidate_chunks",
            return_value=[_mock_chunk(CHUNK_ID_A)],
        ):
            with pytest.raises(ValueError, match="non-string chunk_ids"):
                regenerate_expected_chunk_ids(example=example, llm=llm)


class TestRechunkExample:
    """Tests for the merge step: only `expected_chunk_ids` may change."""

    def test_only_expected_chunk_ids_is_replaced(self) -> None:
        """Every other field on the example is preserved byte-for-byte."""
        original = _build_example(
            chunk_ids=[CHUNK_ID_A],
            source_titles=["Original Title A", "Original Title B"],
            keywords=[
                KeywordEntry(primary="alpha", synonyms=["a_syn_1", "a_syn_2"]),
                KeywordEntry(primary="beta", synonyms=["b_syn_1"]),
            ],
        )

        result = rechunk_example(original, [CHUNK_ID_B, CHUNK_ID_C])

        # Field under test is replaced.
        assert result.expected_chunk_ids == [CHUNK_ID_B, CHUNK_ID_C]

        # All other fields preserved.
        assert result.id == original.id
        assert result.question == original.question
        assert result.author == original.author
        assert result.language == original.language
        assert result.expected_source_titles == original.expected_source_titles
        assert result.expected_keywords == original.expected_keywords

    def test_returns_a_copy_not_the_same_instance(self) -> None:
        """Rechunking must not mutate the source example in place."""
        original = _build_example(chunk_ids=[CHUNK_ID_A])

        result = rechunk_example(original, [CHUNK_ID_B])

        assert result is not original
        assert original.expected_chunk_ids == [CHUNK_ID_A]  # unchanged

    def test_accepts_empty_new_ids(self) -> None:
        """An empty list is a legal input (LLM returned no relevant chunks)."""
        original = _build_example(chunk_ids=[CHUNK_ID_A])

        result = rechunk_example(original, [])

        assert result.expected_chunk_ids == []
        # Other fields still preserved.
        assert result.expected_source_titles == original.expected_source_titles
        assert result.expected_keywords == original.expected_keywords
