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

# Generic test data constants
AUTHOR = "test-author"
CHUNK_ID_1 = "test_chunk_001"
CHUNK_ID_2 = "test_chunk_002"
CHUNK_ID_3 = "test_chunk_003"
CHUNK_ID_4 = "test_chunk_004"
DOC_TITLE_FULL = "Sample Document A"
DOC_TITLE_NO_PAGE = "Sample Document B"
PAGE_NUMBER = 42
SOURCE_URL_FULL = "https://example.com/doc-a"
SOURCE_URL_NO_PAGE = "https://example.com/doc-b"
SOURCE_URL_NO_TITLE = "https://example.com/doc-c"
CONTENT_FULL = "Sample text content for testing"
CONTENT_NO_PAGE = "More sample text"
CONTENT_NO_TITLE = "Additional sample text"
CONTENT_MINIMAL = "Minimal test content"


@pytest.fixture
def sample_docs() -> dict[str, Document]:
    """Provide sample documents with varying metadata for testing."""
    return {
        "full": Document(
            page_content=CONTENT_FULL,
            metadata={
                "chunk_id": CHUNK_ID_1,
                "document_title": DOC_TITLE_FULL,
                "page_number": PAGE_NUMBER,
                "author": AUTHOR,
                "source": SOURCE_URL_FULL,
            },
        ),
        "no_page": Document(
            page_content=CONTENT_NO_PAGE,
            metadata={
                "chunk_id": CHUNK_ID_2,
                "document_title": DOC_TITLE_NO_PAGE,
                "author": AUTHOR,
                "source": SOURCE_URL_NO_PAGE,
            },
        ),
        "no_title": Document(
            page_content=CONTENT_NO_TITLE,
            metadata={
                "chunk_id": CHUNK_ID_3,
                "author": AUTHOR,
                "source": SOURCE_URL_NO_TITLE,
            },
        ),
        "minimal": Document(
            page_content=CONTENT_MINIMAL,
            metadata={
                "chunk_id": CHUNK_ID_4,
            },
        ),
    }


class TestBuildChain:
    """Tests for build_chain function."""

    class TestMetadataExtraction:
        """Tests for extracting metadata from documents."""

        @pytest.mark.parametrize(
            "doc_key,expected_title",
            [
                ("full", f"{DOC_TITLE_FULL}, page {PAGE_NUMBER}"),
                ("no_page", DOC_TITLE_NO_PAGE),
                ("no_title", SOURCE_URL_NO_TITLE),
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
                f"{DOC_TITLE_FULL}, page {PAGE_NUMBER}",
                DOC_TITLE_NO_PAGE,
                SOURCE_URL_NO_TITLE,
            ]
            assert response.retrieved_passage_ids == [CHUNK_ID_1, CHUNK_ID_2, CHUNK_ID_3]
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
            assert response.retrieved_passage_ids == [CHUNK_ID_1, CHUNK_ID_2]

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

    class TestContextFormatting:
        """Tests for formatting context passed to LLM."""

        def test_context_formatting(
            self, sample_docs, mock_retriever_with_docs, mock_llm_with_response
        ) -> None:
            """Should format context with source labels AFTER content without chunkIDs."""

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

            # Should include source labels AFTER content (not before)
            content1_idx = context_str.find(CONTENT_FULL)
            source1_idx = context_str.find(f"[source: {DOC_TITLE_FULL}, page {PAGE_NUMBER}]")
            assert content1_idx != -1 and source1_idx != -1
            assert content1_idx < source1_idx, "Source label should appear AFTER content"

            content2_idx = context_str.find(CONTENT_NO_PAGE)
            source2_idx = context_str.find(f"[source: {DOC_TITLE_NO_PAGE}]")
            assert content2_idx != -1 and source2_idx != -1
            assert content2_idx < source2_idx, "Source label should appear AFTER content"

            # Should NOT include chunk IDs in context (they're internal metadata only)
            assert f"chunk_id: {CHUNK_ID_1}" not in context_str
            assert f"chunk_id: {CHUNK_ID_2}" not in context_str

    class TestPromptIntegration:
        """Tests for prompt integration in the chain (prompt-agnostic)."""

        def test_chain_uses_provided_prompt(
            self, sample_docs, mock_retriever_with_docs, mock_llm_with_response
        ) -> None:
            """Should use the provided prompt template when invoking the chain."""
            docs = [sample_docs["full"]]
            mock_llm = mock_llm_with_response()
            custom_prompt = build_voltaire_prompt()

            chain = build_chain(
                retriever=mock_retriever_with_docs(docs),
                prompt=custom_prompt,
                llm=mock_llm,
            )

            chain.invoke(SAMPLE_QUESTION)

            # Verify LLM was invoked (prompt was used)
            mock_llm.invoke.assert_called_once()
            # Verify the call included messages (indicating prompt was applied)
            llm_call_args = mock_llm.invoke.call_args[0][0]
            assert llm_call_args is not None
            assert len(llm_call_args) > 0

        def test_chain_passes_context_to_prompt(
            self, sample_docs, mock_retriever_with_docs, mock_llm_with_response
        ) -> None:
            """Should pass retrieved context to the prompt."""
            docs = [sample_docs["full"]]
            mock_llm = mock_llm_with_response()

            chain = build_chain(
                retriever=mock_retriever_with_docs(docs),
                prompt=build_voltaire_prompt(),
                llm=mock_llm,
            )

            chain.invoke(SAMPLE_QUESTION)

            # Verify the retrieved document content was passed to the LLM
            llm_call_args = mock_llm.invoke.call_args[0][0]
            messages_str = str(llm_call_args)
            # The document content should appear somewhere in the formatted messages
            assert CONTENT_FULL in messages_str

    class TestLanguageDetection:
        """Tests for language detection logic."""

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

        def test_defaults_to_application_language(
            self, sample_docs, mock_retriever_with_docs, mock_llm_with_response
        ) -> None:
            """Should use DEFAULT_RESPONSE_LANGUAGE when default_language not provided."""
            from src.configs.common import DEFAULT_RESPONSE_LANGUAGE

            chain = build_chain(
                retriever=mock_retriever_with_docs([sample_docs["full"]]),
                prompt=build_voltaire_prompt(),
                llm=mock_llm_with_response("French response"),
                author="voltaire",
            )

            response = chain.invoke(SAMPLE_QUESTION)
            # Should use application-level default
            assert response.language == DEFAULT_RESPONSE_LANGUAGE

    class TestComponentWiring:
        """Tests for dependency injection and component integration."""

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
            assert response.retrieved_passage_ids == [CHUNK_ID_1]
            assert response.retrieved_source_titles == [f"{DOC_TITLE_FULL}, page {PAGE_NUMBER}"]
            assert response.language == "en"
            assert len(response.retrieved_contexts) == 1
            assert response.retrieved_contexts[0] == CONTENT_FULL

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

    class TestEdgeCases:
        """Tests for error handling and edge cases."""

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

        def test_unknown_author_raises_error(self) -> None:
            """Should raise ValueError for unknown author."""
            with pytest.raises(ValueError, match="Unknown author: 'unknown'"):
                build_chain(author="unknown")
