"""Unit tests for chat chain."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.chains.chat_chain import build_chain
from src.configs.common import DEFAULT_LLM_MODEL, DEFAULT_RESPONSE_LANGUAGE

# Test constants
SAMPLE_QUESTION = "What do you think is the most important question in philosophy?"

# Generic test data constants
AUTHOR = "voltaire"
CHUNK_ID_1 = "test_chunk_001"
CHUNK_ID_2 = "test_chunk_002"
CHUNK_ID_3 = "test_chunk_003"
CHUNK_ID_4 = "test_chunk_004"
DOC_TITLE_FULL = "Sample Document A"
DOC_TITLE_NO_PAGE = "Sample Document B"
PAGE_NUMBER = 42
SAMPLE_RESPONSE = "It seems to me that"
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
                author=AUTHOR,
                retriever=mock_retriever_with_docs([sample_docs[doc_key]]),
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
                author=AUTHOR,
                retriever=mock_retriever_with_docs(docs),
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
                author=AUTHOR,
                retriever=mock_retriever_with_docs(docs),
                llm=mock_llm_with_response(),
            )

            response = chain.invoke(SAMPLE_QUESTION)
            assert response.retrieved_passage_ids == [CHUNK_ID_1, CHUNK_ID_2]

        def test_missing_chunk_id(self, mock_retriever_with_docs, mock_llm_with_response) -> None:
            """Should use 'unknown' for missing chunk_id."""
            doc_no_chunk_id = Document(page_content="test", metadata={})
            chain = build_chain(
                author=AUTHOR,
                retriever=mock_retriever_with_docs([doc_no_chunk_id]),
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
                author=AUTHOR,
                retriever=mock_retriever_with_docs(docs),
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

    class TestPersonaPromptBehavior:
        """Tests that verify the persona prompts are correctly integrated."""

        def test_chain_invokes_llm_with_author_prompt(
            self, sample_docs, mock_retriever_with_docs, mock_llm_with_response
        ) -> None:
            """Should invoke LLM with author's persona prompt from registry."""
            docs = [sample_docs["full"]]
            mock_llm = mock_llm_with_response()

            chain = build_chain(
                author=AUTHOR,
                retriever=mock_retriever_with_docs(docs),
                llm=mock_llm,
            )

            chain.invoke(SAMPLE_QUESTION)

            # Verify LLM was invoked (prompt was used)
            mock_llm.invoke.assert_called_once()

            # Verify the call included messages (indicating prompt was applied)
            llm_call_args = mock_llm.invoke.call_args[0][0]
            assert llm_call_args is not None
            assert len(llm_call_args) > 0

        def test_chain_includes_context_in_formatted_prompt(
            self, sample_docs, mock_retriever_with_docs, mock_llm_with_response
        ) -> None:
            """Should include retrieved context in the formatted prompt."""
            docs = [sample_docs["full"]]
            mock_llm = mock_llm_with_response()

            chain = build_chain(
                author=AUTHOR,
                retriever=mock_retriever_with_docs(docs),
                llm=mock_llm,
            )

            chain.invoke(SAMPLE_QUESTION)

            # Verify the retrieved document content was passed to the LLM
            llm_call_args = mock_llm.invoke.call_args[0][0]
            messages_str = str(llm_call_args)

            # The document content should appear somewhere in the formatted messages
            assert CONTENT_FULL in messages_str

    class TestComponentWiring:
        """Tests for dependency injection and component integration."""

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
            mock_llm_instance.invoke.return_value = AIMessage(content=SAMPLE_RESPONSE)
            mock_chat_ollama.return_value = mock_llm_instance

            # Build chain with defaults
            chain = build_chain(author="voltaire")

            # Invoke the chain
            response = chain.invoke(SAMPLE_QUESTION)

            # Verify all default components (author, retriever, llm, language) were recorded or instantiated.
            call_kwargs = mock_build_retriever.call_args[1]
            assert call_kwargs["author"] == "voltaire"
            mock_chat_ollama.assert_called_once_with(model=DEFAULT_LLM_MODEL)
            assert response.language == DEFAULT_RESPONSE_LANGUAGE

        def test_build_chain_accepts_custom_components(
            self, sample_docs, mock_retriever_with_docs, mock_llm_with_response
        ) -> None:
            """Should accept and use custom retriever and LLM parameters."""

            mock_retriever = mock_retriever_with_docs([sample_docs["full"]])
            mock_llm = mock_llm_with_response(SAMPLE_RESPONSE)

            chain = build_chain(
                author=AUTHOR,
                retriever=mock_retriever,
                llm=mock_llm,
            )

            # Use language parameter in invoke
            german_code = "de"
            response = chain.invoke(SAMPLE_QUESTION, language=german_code)

            # Verify custom components were used
            mock_retriever.invoke.assert_called_once_with(SAMPLE_QUESTION)
            mock_llm.invoke.assert_called_once()
            assert response.language == german_code

        def test_chain_end_to_end_with_documents(
            self, sample_docs, mock_retriever_with_docs, mock_llm_with_response
        ) -> None:
            """Should correctly extract and structure document metadata into response."""

            chain = build_chain(
                author=AUTHOR,
                retriever=mock_retriever_with_docs([sample_docs["full"]]),
                llm=mock_llm_with_response(SAMPLE_RESPONSE),
            )

            response = chain.invoke(SAMPLE_QUESTION)

            # Verify complete response structure
            assert response.text == SAMPLE_RESPONSE
            assert response.retrieved_passage_ids == [CHUNK_ID_1]
            assert response.retrieved_source_titles == [f"{DOC_TITLE_FULL}, page {PAGE_NUMBER}"]
            assert response.retrieved_contexts == [CONTENT_FULL]
            assert len(response.retrieved_contexts) == 1

    class TestEdgeCases:
        """Tests for error handling and edge cases."""

        def test_empty_retrieval(self, mock_retriever_with_docs, mock_llm_with_response) -> None:
            """Should handle empty retrieval gracefully."""
            chain = build_chain(
                author=AUTHOR,
                retriever=mock_retriever_with_docs([]),
                llm=mock_llm_with_response("No relevant sources found"),
            )

            response = chain.invoke(SAMPLE_QUESTION)
            assert response.text == "No relevant sources found"
            assert response.retrieved_passage_ids == []
            assert response.retrieved_source_titles == []
            assert response.retrieved_contexts == []

        def test_unsupported_author_raises_error(self) -> None:
            """Should raise ValueError for unsupported author."""

            unsupported_author = "unsupported author"
            with pytest.raises(ValueError, match=f"Unknown author: '{unsupported_author}'"):
                build_chain(author=unsupported_author)
