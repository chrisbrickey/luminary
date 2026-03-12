"""Unit tests for chat chain."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.chains.chat_chain import build_chain
from src.configs.common import DEFAULT_LLM_MODEL
from src.prompts.voltaire import build_voltaire_prompt

# Test constants
SAMPLE_QUESTION = "What is tolerance?"


@pytest.fixture
def sample_docs() -> dict[str, Document]:
    """Provide sample documents with varying metadata for testing."""
    return {
        "full": Document(
            page_content="Text about tolerance",
            metadata={
                "chunk_id": "abc123",
                "document_title": "Lettres philosophiques",
                "page_number": 5,
                "author": "voltaire",
                "source": "https://example.com/lettres",
            },
        ),
        "no_page": Document(
            page_content="More text",
            metadata={
                "chunk_id": "def456",
                "document_title": "Philosophical Letters",
                "author": "voltaire",
                "source": "https://example.com/letters",
            },
        ),
        "no_title": Document(
            page_content="Text without title",
            metadata={
                "chunk_id": "ghi789",
                "author": "voltaire",
                "source": "https://example.com/source",
            },
        ),
        "minimal": Document(
            page_content="Minimal metadata",
            metadata={
                "chunk_id": "jkl012",
            },
        ),
    }


class TestBuildChain:
    """Tests for build_chain function."""

    def test_chain_invokes_retriever_and_llm(
        self, sample_docs, mock_retriever_with_docs, mock_llm_with_response
    ) -> None:
        """Should wire retriever and LLM correctly."""
        # Build chain
        chain = build_chain(
            retriever=mock_retriever_with_docs([sample_docs["full"]]),
            prompt=build_voltaire_prompt(),
            llm=mock_llm_with_response("Response about tolerance"),
            detect_user_language=False,
        )

        # Invoke
        response = chain.invoke(SAMPLE_QUESTION)

        # Assertions
        assert response.text == "Response about tolerance"
        assert response.retrieved_passage_ids == ["abc123"]
        assert response.retrieved_source_titles == ["Lettres philosophiques, page 5"]
        assert response.language == "fr"
        assert len(response.retrieved_contexts) == 1
        assert response.retrieved_contexts[0] == "Text about tolerance"

    @pytest.mark.parametrize(
        "doc_key,expected_title",
        [
            ("full", "Lettres philosophiques, page 5"),
            ("no_page", "Philosophical Letters"),
            ("no_title", "https://example.com/source"),
            ("minimal", "unknown"),
        ],
    )
    def test_source_title_extraction(
        self, sample_docs, doc_key, expected_title, mock_retriever_with_docs, mock_llm_with_response
    ) -> None:
        """Should extract source titles correctly with various metadata levels."""
        chain = build_chain(
            retriever=mock_retriever_with_docs([sample_docs[doc_key]]),
            prompt=build_voltaire_prompt(),
            llm=mock_llm_with_response(),
        )

        response = chain.invoke(SAMPLE_QUESTION)
        assert response.retrieved_source_titles == [expected_title]

    def test_multiple_documents_with_varying_metadata(
        self, sample_docs, mock_retriever_with_docs, mock_llm_with_response
    ) -> None:
        """Should handle multiple documents with different metadata completeness."""
        docs = [sample_docs["full"], sample_docs["no_page"], sample_docs["no_title"]]
        chain = build_chain(
            retriever=mock_retriever_with_docs(docs),
            prompt=build_voltaire_prompt(),
            llm=mock_llm_with_response(),
        )

        response = chain.invoke(SAMPLE_QUESTION)
        assert response.retrieved_source_titles == [
            "Lettres philosophiques, page 5",
            "Philosophical Letters",
            "https://example.com/source",
        ]
        assert response.retrieved_passage_ids == ["abc123", "def456", "ghi789"]
        assert len(response.retrieved_contexts) == 3

    def test_chunk_id_extraction(
        self, sample_docs, mock_retriever_with_docs, mock_llm_with_response
    ) -> None:
        """Should extract chunk IDs into retrieved_passage_ids."""
        docs = [sample_docs["full"], sample_docs["no_page"]]
        chain = build_chain(
            retriever=mock_retriever_with_docs(docs),
            prompt=build_voltaire_prompt(),
            llm=mock_llm_with_response(),
        )

        response = chain.invoke(SAMPLE_QUESTION)
        assert response.retrieved_passage_ids == ["abc123", "def456"]

    def test_missing_chunk_id(self, mock_retriever_with_docs, mock_llm_with_response) -> None:
        """Should use 'unknown' for missing chunk_id."""
        doc_no_chunk_id = Document(page_content="test", metadata={})
        chain = build_chain(
            retriever=mock_retriever_with_docs([doc_no_chunk_id]),
            prompt=build_voltaire_prompt(),
            llm=mock_llm_with_response(),
        )

        response = chain.invoke(SAMPLE_QUESTION)
        assert response.retrieved_passage_ids == ["unknown"]

    def test_context_formatting(
        self, sample_docs, mock_retriever_with_docs, mock_llm_with_response
    ) -> None:
        """Should format context with source labels but NOT chunk IDs."""
        docs = [sample_docs["full"], sample_docs["no_page"]]
        mock_llm = mock_llm_with_response()

        chain = build_chain(
            retriever=mock_retriever_with_docs(docs),
            prompt=build_voltaire_prompt(),
            llm=mock_llm,
        )

        chain.invoke(SAMPLE_QUESTION)

        # Inspect what was passed to the LLM
        llm_call_args = mock_llm.invoke.call_args[0][0]
        context_str = str(llm_call_args)

        # Should include source labels without chunk IDs
        assert "[source: Lettres philosophiques, page 5]" in context_str
        assert "[source: Philosophical Letters]" in context_str
        assert "Text about tolerance" in context_str

        # Should NOT include chunk IDs in context (they're internal metadata only)
        assert "chunk_id: abc123" not in context_str
        assert "chunk_id: def456" not in context_str

    def test_prompt_does_not_instruct_chunk_id_citations(
        self, sample_docs, mock_retriever_with_docs, mock_llm_with_response
    ) -> None:
        """Should NOT instruct LLM to cite chunk IDs in responses."""
        docs = [sample_docs["full"]]
        mock_llm = mock_llm_with_response()

        chain = build_chain(
            retriever=mock_retriever_with_docs(docs),
            prompt=build_voltaire_prompt(),
            llm=mock_llm,
        )

        chain.invoke(SAMPLE_QUESTION)

        # Inspect the prompt template passed to the LLM
        llm_call_args = mock_llm.invoke.call_args[0][0]
        prompt_str = str(llm_call_args)

        # Prompt should NOT instruct LLM to include chunk_id in citations
        assert "chunk_id: xxx" not in prompt_str
        assert "| chunk_id:" not in prompt_str

        # Prompt should still instruct proper citation format (without chunk_id)
        assert "[source:" in prompt_str

    def test_prompt_instructs_citations_after_sentences(
        self, sample_docs, mock_retriever_with_docs, mock_llm_with_response
    ) -> None:
        """Should instruct LLM to place citations AFTER sentences, not before."""
        docs = [sample_docs["full"]]
        mock_llm = mock_llm_with_response()

        chain = build_chain(
            retriever=mock_retriever_with_docs(docs),
            prompt=build_voltaire_prompt(),
            llm=mock_llm,
        )

        chain.invoke(SAMPLE_QUESTION)

        # Inspect the prompt template passed to the LLM
        llm_call_args = mock_llm.invoke.call_args[0][0]
        prompt_str = str(llm_call_args).lower()

        # Prompt should instruct placing citations after the sentence/argument
        assert "après la phrase" in prompt_str or "after the sentence" in prompt_str

        # Prompt should explicitly forbid citations at the beginning
        assert "jamais de citation au début" in prompt_str or "never" in prompt_str and "beginning" in prompt_str

    def test_empty_retrieval(self, mock_retriever_with_docs, mock_llm_with_response) -> None:
        """Should handle empty retrieval gracefully."""
        chain = build_chain(
            retriever=mock_retriever_with_docs([]),
            prompt=build_voltaire_prompt(),
            llm=mock_llm_with_response("No relevant sources found"),
        )

        response = chain.invoke(SAMPLE_QUESTION)
        assert response.text == "No relevant sources found"
        assert response.retrieved_passage_ids == []
        assert response.retrieved_source_titles == []
        assert response.retrieved_contexts == []

    def test_uses_default_language(
        self, sample_docs, mock_retriever_with_docs, mock_llm_with_response
    ) -> None:
        """Should use default_language parameter when explicitly provided."""
        chain = build_chain(
            retriever=mock_retriever_with_docs([sample_docs["full"]]),
            prompt=build_voltaire_prompt(),
            llm=mock_llm_with_response("English response"),
            default_language="en",
        )

        response = chain.invoke(SAMPLE_QUESTION)
        assert response.language == "en"

    def test_defaults_to_language_from_author(
        self, sample_docs, mock_retriever_with_docs, mock_llm_with_response
    ) -> None:
        """Should use author's language when default_language not provided."""
        chain = build_chain(
            retriever=mock_retriever_with_docs([sample_docs["full"]]),
            prompt=build_voltaire_prompt(),
            llm=mock_llm_with_response("French response"),
            author="voltaire",
        )

        response = chain.invoke(SAMPLE_QUESTION)
        # "voltaire" author's language is "fr"
        assert response.language == "fr"

    def test_unknown_author_raises_error(self) -> None:
        """Should raise ValueError for unknown author."""
        with pytest.raises(ValueError, match="Unknown author: 'unknown'"):
            build_chain(author="unknown")

    @patch("src.chains.chat_chain.build_retriever")
    @patch("src.chains.chat_chain.ChatOllama")
    def test_build_chain_with_defaults_wires_components(
        self, mock_chat_ollama: Mock, mock_build_retriever: Mock, sample_docs
    ) -> None:
        """Should wire retriever, LLM, and prompt correctly when using defaults."""
        # Mock retriever
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = [sample_docs["full"]]
        mock_build_retriever.return_value = mock_retriever

        # Mock LLM
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.return_value = AIMessage(content="Test response")
        mock_chat_ollama.return_value = mock_llm_instance

        # Build chain with defaults
        chain = build_chain(persist_dir="test_db", author="voltaire")

        # Invoke
        response = chain.invoke(SAMPLE_QUESTION)

        # Verify components were instantiated
        mock_build_retriever.assert_called_once()
        call_kwargs = mock_build_retriever.call_args[1]
        assert call_kwargs["persist_dir"] == "test_db"
        assert call_kwargs["author"] == "voltaire"

        mock_chat_ollama.assert_called_once_with(model=DEFAULT_LLM_MODEL)

        # Verify response
        assert response.text == "Test response"

    def test_build_chain_with_injected_components(
        self, sample_docs, mock_retriever_with_docs, mock_llm_with_response
    ) -> None:
        """Should use injected components when provided (testing mode)."""
        mock_retriever = mock_retriever_with_docs([sample_docs["full"]])
        mock_llm = mock_llm_with_response("Injected response")

        # Build chain with all injected
        chain = build_chain(
            retriever=mock_retriever,
            llm=mock_llm,
            prompt=build_voltaire_prompt(),
            default_language="en",
        )

        response = chain.invoke(SAMPLE_QUESTION)

        # Verify injected components were used
        mock_retriever.invoke.assert_called_once_with(SAMPLE_QUESTION)
        mock_llm.invoke.assert_called_once()
        assert response.text == "Injected response"
        assert response.language == "en"
