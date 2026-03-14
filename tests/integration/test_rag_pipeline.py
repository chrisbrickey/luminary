"""Integration tests for RAG pipeline components.

This file tests the integration of real components:
- ChromaDB vectorstore (embed_and_store)
- Retriever (build_retriever)
- Chat chain (build_chain)
- Prompts (build_voltaire_prompt)

External dependencies (LLM, embeddings) are faked to avoid network calls.
"""

from pathlib import Path

import pytest

from src.chains.chat_chain import build_chain
from src.prompts.voltaire import build_voltaire_prompt
from src.schemas import ChatResponse
from src.vectorstores.chroma import embed_and_store
from src.vectorstores.retriever import build_retriever
from tests.conftest import FakeChatModel, FakeEmbeddings

# Test constants
DEFAULT_K = 5
TEST_QUESTION = "What do you think about tolerance?"


# =============================================================================
# Vectorstore Operations (storage, retrieval, filtering, persistence)
# =============================================================================


def test_vectorstore_round_trip_storage_and_retrieval(
    tmp_path: Path, make_test_document
) -> None:
    """Test full round-trip: embed documents then retrieve them.

    This integration test verifies that:
    1. Documents can be embedded and stored in ChromaDB
    2. A retriever can be built from the same database
    3. The retriever can successfully retrieve the stored documents
    4. All metadata is preserved through the round-trip
    """
    db_path = tmp_path / "chroma"
    embeddings = FakeEmbeddings()

    # Create sample documents
    original_chunks = [
        make_test_document(
            content="Discussion of tolerance and reason in Enlightenment thought.",
            chunk_id="test_chunk_001",
            chunk_index=0,
            doc_id="test-document-1",
            title="Test Philosophical Text",
            author="test-author",
            page_number=1,
        ),
        make_test_document(
            content="Analysis of natural rights and social contracts.",
            chunk_id="test_chunk_002",
            chunk_index=1,
            doc_id="test-document-1",
            title="Test Philosophical Text",
            author="test-author",
            page_number=2,
        ),
        make_test_document(
            content="Exploration of liberty and equality principles.",
            chunk_id="test_chunk_003",
            chunk_index=0,
            doc_id="test-document-2",
            title="Another Test Text",
            author="test-author",
            page_number=1,
        ),
    ]

    # Embed and store
    embed_and_store(chunks=original_chunks, persist_dir=db_path, embeddings=embeddings)

    # Build retriever from same database
    retriever = build_retriever(persist_dir=db_path, embeddings=embeddings, k=5)

    # Retrieve documents
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
            c
            for c in original_chunks
            if c.metadata["chunk_id"] == retrieved_doc.metadata["chunk_id"]
        )

        # Check all metadata fields
        assert retrieved_doc.metadata["chunk_index"] == original_chunk.metadata["chunk_index"]
        assert retrieved_doc.metadata["document_id"] == original_chunk.metadata["document_id"]
        assert (
            retrieved_doc.metadata["document_title"]
            == original_chunk.metadata["document_title"]
        )
        assert retrieved_doc.metadata["author"] == original_chunk.metadata["author"]
        assert retrieved_doc.metadata["source"] == original_chunk.metadata["source"]
        assert retrieved_doc.metadata["page_number"] == original_chunk.metadata["page_number"]


def test_vectorstore_author_filtering(tmp_path: Path, make_test_document) -> None:
    """Test retriever author filtering across full pipeline.

    Verifies that:
    1. Documents from multiple authors can be stored together
    2. Retriever correctly filters by author metadata
    3. No cross-contamination between author filters
    """
    db_path = tmp_path / "chroma"
    embeddings = FakeEmbeddings()

    # Create documents from two different authors
    chunks = [
        make_test_document(
            content="First author's perspective on philosophy.",
            chunk_id="author1_chunk001",
            chunk_index=0,
            doc_id="author1-doc",
            author="author-one",
            page_number=None,
        ),
        make_test_document(
            content="First author's continued discussion.",
            chunk_id="author1_chunk002",
            chunk_index=1,
            doc_id="author1-doc",
            author="author-one",
            page_number=None,
        ),
        make_test_document(
            content="Second author's perspective on philosophy.",
            chunk_id="author2_chunk001",
            chunk_index=0,
            doc_id="author2-doc",
            author="author-two",
            page_number=None,
        ),
        make_test_document(
            content="Second author's continued discussion.",
            chunk_id="author2_chunk002",
            chunk_index=1,
            doc_id="author2-doc",
            author="author-two",
            page_number=None,
        ),
    ]

    # Embed and store all chunks
    embed_and_store(chunks=chunks, persist_dir=db_path, embeddings=embeddings)

    # Test 1: Retrieve only author-one's documents
    retriever_author1 = build_retriever(
        persist_dir=db_path, embeddings=embeddings, k=5, author="author-one"
    )

    results_author1 = retriever_author1.invoke("philosophy")
    assert len(results_author1) == 2
    assert all(doc.metadata["author"] == "author-one" for doc in results_author1)

    # Test 2: Retrieve only author-two's documents
    retriever_author2 = build_retriever(
        persist_dir=db_path, embeddings=embeddings, k=5, author="author-two"
    )

    results_author2 = retriever_author2.invoke("philosophy")
    assert len(results_author2) == 2
    assert all(doc.metadata["author"] == "author-two" for doc in results_author2)

    # Test 3: Retrieve without filter (should get all)
    retriever_all = build_retriever(
        persist_dir=db_path, embeddings=embeddings, k=5, author=None
    )

    results_all = retriever_all.invoke("philosophy")
    assert len(results_all) == 4


def test_vectorstore_persistence_across_sessions(
    tmp_path: Path, make_test_document
) -> None:
    """Test retriever works across separate sessions.

    This simulates the real-world scenario where:
    1. Documents are ingested and stored in one session
    2. Later, a retriever is created to query the stored data
    """
    db_path = tmp_path / "chroma"
    embeddings = FakeEmbeddings()

    # Session 1: Ingest documents
    chunks = [
        make_test_document(
            content="Persistent content that should survive.",
            chunk_id="persist_001",
            chunk_index=0,
            doc_id="persistent-doc",
            author="test-author",
            page_number=None,
        ),
    ]

    embed_and_store(chunks=chunks, persist_dir=db_path, embeddings=embeddings)

    # Session 2: Create new retriever (simulating separate execution)
    retriever = build_retriever(persist_dir=db_path, embeddings=embeddings, k=5)

    # Should retrieve the previously stored document
    results = retriever.invoke("persistent")
    assert len(results) == 1
    assert results[0].metadata["chunk_id"] == "persist_001"


# =============================================================================
# Full RAG Chain (vectorstore + retriever + chain + prompt)
# =============================================================================


@pytest.fixture
def integration_chain_setup(tmp_path: Path) -> pytest.fixture:
    """Set up ChromaDB, embeddings, and helper function for full RAG chain tests."""
    embeddings = FakeEmbeddings()
    db_path = tmp_path / "chroma"

    def build_test_chain(
        chunks,
        author="voltaire",
        default_language="fr",
        k=DEFAULT_K,
        detect_user_language=True,
    ):
        """Build a complete chain with real ChromaDB and fake LLM.

        Args:
            chunks: Documents to embed and store
            author: Author key for filtering (default: "voltaire")
            default_language: Fallback language when detection disabled (default: "fr")
            k: Number of documents to retrieve (default: DEFAULT_K)
            detect_user_language: Enable language detection from question (default: True)
        """
        embed_and_store(chunks=chunks, persist_dir=db_path, embeddings=embeddings)
        retriever = build_retriever(
            persist_dir=db_path, embeddings=embeddings, k=k, author=author
        )
        return build_chain(
            retriever=retriever,
            llm=FakeChatModel(),
            prompt=build_voltaire_prompt(),
            default_language=default_language,
            detect_user_language=detect_user_language,
        )

    return build_test_chain


def test_full_rag_chain_with_retrieval(
    integration_chain_setup, make_test_document
) -> None:
    """Test complete RAG pipeline: storage → retrieval → chain → response.

    This integration test verifies:
    1. Documents are embedded and stored in ChromaDB
    2. Retriever successfully fetches relevant documents
    3. Prompt is formatted correctly with retrieved context
    4. Chain returns a valid ChatResponse with all expected fields
    5. Source titles include page numbers when available
    6. Language detection detects question language and responds accordingly
    """
    # Create fixture documents
    chunks = [
        make_test_document(
            content="Tolerance is essential for civil society and peaceful coexistence.",
            chunk_id="test_001",
            chunk_index=0,
            doc_id="lettres-philosophiques",
            title="Lettres philosophiques",
            page_number=5,
        ),
        make_test_document(
            content="Religious tolerance prevents fanaticism and promotes reason.",
            chunk_id="test_002",
            chunk_index=1,
            doc_id="lettres-philosophiques",
            title="Lettres philosophiques",
            page_number=6,
        ),
        make_test_document(
            content="The English model demonstrates religious pluralism in practice.",
            chunk_id="test_003",
            chunk_index=2,
            doc_id="lettres-philosophiques",
            title="Lettres philosophiques",
            page_number=7,
        ),
    ]

    # Build complete chain with language detection enabled
    chain = integration_chain_setup(chunks, detect_user_language=True)

    # Test 1: English question → English response
    response_en = chain.invoke("What do you think about tolerance?")

    # Verify response structure
    assert isinstance(response_en, ChatResponse)
    assert isinstance(response_en.text, str)
    assert len(response_en.text) > 0

    # Verify language detection worked for English
    assert response_en.language == "en"

    # Verify retrieved metadata
    assert len(response_en.retrieved_passage_ids) == 3
    assert len(response_en.retrieved_contexts) == 3
    assert len(response_en.retrieved_source_titles) == 3

    # Verify chunk IDs match what we stored
    assert set(response_en.retrieved_passage_ids) == {"test_001", "test_002", "test_003"}

    # Verify source titles include page numbers
    assert "Lettres philosophiques, page 5" in response_en.retrieved_source_titles
    assert "Lettres philosophiques, page 6" in response_en.retrieved_source_titles
    assert "Lettres philosophiques, page 7" in response_en.retrieved_source_titles

    # Verify contexts match original content
    contexts_set = set(response_en.retrieved_contexts)
    assert (
        "Tolerance is essential for civil society and peaceful coexistence." in contexts_set
    )

    # Test 2: French question → French response
    response_fr = chain.invoke("Que pensez-vous de la tolérance?")

    # Verify language detection worked for French
    assert response_fr.language == "fr"

    # Verify same retrieval quality (same number of chunks retrieved)
    assert len(response_fr.retrieved_passage_ids) == 3
    assert len(response_fr.retrieved_contexts) == 3
    assert len(response_fr.retrieved_source_titles) == 3


def test_rag_chain_handles_missing_metadata(
    integration_chain_setup, make_test_document
) -> None:
    """Test RAG chain with documents having varying metadata completeness.

    Verifies that:
    1. Chain gracefully handles missing page numbers
    2. Chain falls back to source URL when title is missing
    3. All documents are still retrieved and processed correctly
    4. Explicit language mode works (detection disabled)
    """
    # Create documents with varying metadata
    chunks = [
        # Full metadata with page number
        make_test_document(
            content="Content with full metadata.",
            chunk_id="full_001",
            chunk_index=0,
            doc_id="doc1",
            title="Complete Document",
            page_number=10,
        ),
        # No page number
        make_test_document(
            content="Content without page number.",
            chunk_id="partial_001",
            chunk_index=0,
            doc_id="doc2",
            title="Partial Document",
            page_number=None,
        ),
        # No title (should fall back to source URL)
        make_test_document(
            content="Content with minimal metadata.",
            chunk_id="minimal_001",
            chunk_index=0,
            doc_id="doc3",
            title="",
            page_number=None,
        ),
    ]

    # Build chain with explicit English language (detection disabled)
    chain = integration_chain_setup(
        chunks, default_language="en", detect_user_language=False
    )

    # Invoke
    response = chain.invoke("test question")

    # Verify all source titles are present with appropriate fallbacks
    assert len(response.retrieved_source_titles) == 3
    assert "Complete Document, page 10" in response.retrieved_source_titles
    assert "Partial Document" in response.retrieved_source_titles
    # Empty title should fall back to source URL
    assert "https://example.com/doc3" in response.retrieved_source_titles

    # Verify explicit language mode (uses default_language, not detection)
    assert response.language == "en"


def test_rag_chain_does_not_expose_chunk_ids_to_llm(
    tmp_path: Path, make_test_document
) -> None:
    """Test that chunk IDs never reach the LLM in the full RAG pipeline.

    This integration test verifies:
    1. Documents are stored with chunk_id metadata
    2. Chunk IDs are available in the response object for --show-chunks
    3. But chunk IDs are NOT included in the context sent to the LLM
    4. And chunk IDs are NOT in the prompt instructions

    Chunk IDs are internal metadata for debugging only.
    """
    from unittest.mock import Mock

    db_path = tmp_path / "chroma"
    embeddings = FakeEmbeddings()

    # Create documents with known chunk IDs
    chunks = [
        make_test_document(
            content="First passage about philosophy.",
            chunk_id="secret_id_001",
            chunk_index=0,
            doc_id="test-doc",
            title="Test Document",
            page_number=1,
        ),
        make_test_document(
            content="Second passage about philosophy.",
            chunk_id="secret_id_002",
            chunk_index=1,
            doc_id="test-doc",
            title="Test Document",
            page_number=2,
        ),
    ]

    # Embed and store
    embed_and_store(chunks=chunks, persist_dir=db_path, embeddings=embeddings)

    # Build retriever
    retriever = build_retriever(
        persist_dir=db_path, embeddings=embeddings, k=5, author="voltaire"
    )

    # Build chain with mock LLM to inspect what it receives
    mock_llm = Mock()
    mock_llm.invoke.return_value = Mock(content="Mock response")

    # Build chain with detection disabled to test explicit language mode
    chain = build_chain(
        retriever=retriever,
        llm=mock_llm,
        prompt=build_voltaire_prompt(),
        default_language="fr",
        detect_user_language=False,  # Explicit language mode
    )

    # Invoke chain
    response = chain.invoke("test question")

    # VERIFY: Chunk IDs ARE available in the response metadata (for --show-chunks)
    assert "secret_id_001" in response.retrieved_passage_ids
    assert "secret_id_002" in response.retrieved_passage_ids

    # VERIFY: Chunk IDs are NOT in the LLM input
    llm_call_args = mock_llm.invoke.call_args[0][0]
    llm_input_str = str(llm_call_args)

    # Should NOT contain chunk IDs
    assert "secret_id_001" not in llm_input_str
    assert "secret_id_002" not in llm_input_str
    assert "chunk_id:" not in llm_input_str

    # Should still contain proper source formatting (without chunk IDs)
    assert "[source: Test Document, page 1]" in llm_input_str
    assert "[source: Test Document, page 2]" in llm_input_str

    # Should contain the actual content
    assert "First passage about philosophy." in llm_input_str
    assert "Second passage about philosophy." in llm_input_str
