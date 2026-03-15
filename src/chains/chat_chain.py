"""RAG chat chain with author-specific prompts."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import ChatOllama

from src.configs.authors import AUTHOR_CONFIGS, DEFAULT_AUTHOR
from src.configs.common import (
    DEFAULT_DB_PATH,
    DEFAULT_LLM_MODEL,
    DEFAULT_RESPONSE_LANGUAGE,
)
from src.schemas import ChatResponse
from src.vectorstores.retriever import build_retriever


def build_chain(
    persist_dir: Path | str = DEFAULT_DB_PATH,
    author: str = DEFAULT_AUTHOR,
    retriever: VectorStoreRetriever | None = None,
    llm: BaseChatModel | None = None,
    prompt: ChatPromptTemplate | None = None,
    language: str | None = None,
    detect_user_language: bool = True,
) -> Runnable[str, ChatResponse]:
    """Build a RAG chat chain with sensible defaults or custom components.

    This method supports both production use (with defaults) and testing (with injection):
    - Production: build_chain() or build_chain(author="gouges")
    - Testing: build_chain(retriever=mock, llm=mock, prompt=mock)

    Args:
        persist_dir: Path to ChromaDB vectorstore (default: DEFAULT_DB_PATH)
        author: Author key for filtering and prompt selection (default: DEFAULT_AUTHOR)
        retriever: LangChain retriever (default: builds from persist_dir + author)
        llm: Language model (default: ChatOllama with DEFAULT_LLM_MODEL)
        prompt: Chat prompt template (default: builds from author registry)
        language: Response language ISO code (default: DEFAULT_RESPONSE_LANGUAGE)
        detect_user_language: Whether to detect question language (default: True)

    Returns:
        Runnable that takes a question string and returns a ChatResponse

    Raises:
        ValueError: If author is not registered in AUTHOR_CONFIGS
    """
    # Validate author
    if author not in AUTHOR_CONFIGS:
        raise ValueError(
            f"Unknown author: {author!r}. "
            f"Valid authors: {list(AUTHOR_CONFIGS.keys())}"
        )

    # Get author-specific configuration
    config = AUTHOR_CONFIGS[author]
    prompt_factory = config.prompt_factory

    # Build retriever if not provided
    if retriever is None:
        retriever = build_retriever(persist_dir=persist_dir, author=author)

    # Build prompt if not provided
    if prompt is None:
        prompt = prompt_factory()

    # Default LLM
    if llm is None:
        llm = ChatOllama(model=DEFAULT_LLM_MODEL)

    # Default language from application config
    if language is None:
        language = DEFAULT_RESPONSE_LANGUAGE

    def _run(question: str) -> ChatResponse:
        """Internal function that executes the RAG pipeline.

        Args:
            question: User's question

        Returns:
            ChatResponse with answer and metadata
        """
        # Retrieve relevant documents
        docs = retriever.invoke(question)

        # Format context with source labels
        context = _format_docs_with_titles(docs)

        # Format the prompt
        # Use language parameter directly for now. We will add detection of user's language.
        formatted_prompt = prompt.format_messages(
            context=context,
            question=question,
            language=language,
        )

        # Invoke LLM
        response = llm.invoke(formatted_prompt)

        # Extract text from response
        if hasattr(response, "content"):
            content = response.content
            response_text = str(content) if not isinstance(content, str) else content
        else:
            response_text = str(response)

        # Extract metadata
        chunk_ids = _extract_chunk_ids(docs)
        contexts = [doc.page_content for doc in docs]
        source_titles = _extract_source_titles(docs)

        return ChatResponse(
            text=response_text,
            retrieved_passage_ids=chunk_ids,
            retrieved_contexts=contexts,
            retrieved_source_titles=source_titles,
            language=language,
        )

    # Return the runnable
    class ChatChainRunnable(Runnable[str, ChatResponse]):
        """Wrapper to make _run conform to Runnable protocol."""

        def invoke(self, input: str, config: dict | None = None) -> ChatResponse:  # type: ignore[override]
            """Execute the chain."""
            return _run(input)

    return ChatChainRunnable()


def _format_docs_with_titles(docs: list[Document]) -> str:
    """Format documents with source labels.

    Each document is labeled with: [source: {title}, page {page_number}]

    Args:
        docs: List of retrieved documents

    Returns:
        Formatted context string
    """
    if not docs:
        return ""

    formatted_chunks = []
    for doc in docs:
        metadata = doc.metadata

        # Build source label
        document_title = metadata.get("document_title")
        page_number = metadata.get("page_number")

        if document_title and page_number is not None:
            source_label = f"{document_title}, page {page_number}"
        elif document_title:
            source_label = document_title
        else:
            source_label = metadata.get("source", "unknown")

        # Format: content\n[source: title, page N]
        formatted_chunks.append(
            f"{doc.page_content}\n[source: {source_label}]"
        )

    return "\n\n".join(formatted_chunks)


def _extract_chunk_ids(docs: list[Document]) -> list[str]:
    """Extract chunk IDs from documents.

    Args:
        docs: List of retrieved documents

    Returns:
        List of chunk_id values
    """
    return [doc.metadata.get("chunk_id", "unknown") for doc in docs]


def _extract_source_titles(docs: list[Document]) -> list[str]:
    """Extract human-readable source titles from documents.

    Combines document_title + page_number when available.
    Fallback order: title-only -> source URL -> "unknown"

    Args:
        docs: List of retrieved documents

    Returns:
        List of formatted source titles
    """
    titles = []
    for doc in docs:
        metadata = doc.metadata
        document_title = metadata.get("document_title")
        page_number = metadata.get("page_number")

        if document_title and page_number is not None:
            titles.append(f"{document_title}, page {page_number}")
        elif document_title:
            titles.append(document_title)
        elif "source" in metadata:
            titles.append(metadata["source"])
        else:
            titles.append("unknown")

    return titles
