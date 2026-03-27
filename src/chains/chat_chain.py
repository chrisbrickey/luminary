"""RAG chat chain. Orchestrates retrieval, context formatting with labels,
and LLM call including persona prompts. Returns ChatResponse.

The chat chain uses a builder pattern with closure and implements the Runnable protocol.
- build_chain() is a factory that creates and configures a ChatChainRunnable instance
- inner _run() function captures dependencies (retriever, llm, prompt) in a closure
- This creates a pre-configured, stateful callable object.

Key benefit: Build once (expensive: load models, connect to DB),
             then invoke many times (cheap: just run the chain).
             The closure captures all the heavy setup, so each invocation is lightweight.

Instructions for callers:
  1. Call build_chain() once to construct a ChatChainRunnable configured for a specific author.
  2. Then call runnable.invoke(user_input, detected_language) repeatedly with different questions
"""

from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import ChatOllama

from src.configs.authors import AUTHOR_CONFIGS, DEFAULT_AUTHOR
from src.configs.common import DEFAULT_CHAT_MODEL, DEFAULT_RESPONSE_LANGUAGE
from src.schemas import ChatResponse
from src.vectorstores.retriever import build_retriever

def build_chain(
    author: str = DEFAULT_AUTHOR,
    retriever: VectorStoreRetriever | None = None,
    llm: BaseChatModel | None = None,
) -> Runnable[str, ChatResponse]:
    """Build a RAG chat chain that takes custom components. If custom components are
    not injected (e.g. for testing), this method constructs sensible defaults.

    This method supports both production use (with defaults) and testing (with injection):
    - Production: build_chain() or build_chain(author="gouges")
    - Testing: build_chain(retriever=mock, llm=mock, prompt=mock)

    Args:
        author: Author key for filtering and prompt selection (default: DEFAULT_AUTHOR)
        retriever: LangChain retriever (default: builds from author)
        llm: Language model (default: ChatOllama with DEFAULT_CHAT_MODEL)

    Returns:
        Runnable that takes string from user input and returns a ChatResponse (via invoke method)

    Raises:
        ValueError: If author is not registered in AUTHOR_CONFIGS
    """

    # Validate author and retrieve that philosopher's prompt
    if author not in AUTHOR_CONFIGS:
        raise ValueError(f"Unknown author: {author!r}. Valid authors: {list(AUTHOR_CONFIGS.keys())}")
    config = AUTHOR_CONFIGS[author]
    prompt = config.prompt_factory()

    # Build retriever if not provided
    if retriever is None:
        retriever = build_retriever(author=author)

    # Assign default LLM if not provided
    if llm is None:
        llm = ChatOllama(model=DEFAULT_CHAT_MODEL)

    def _run(user_input: str, language: str) -> ChatResponse:
        """Internal function that executes the RAG pipeline.

        Args:
            user_input: user input
            language: ISO 639-1 language code (e.g., "en", "fr")

        Returns:
            ChatResponse with answer and metadata
        """
        # Retrieve relevant embedded documents
        docs = retriever.invoke(user_input)

        # Format context with source labels
        context = _format_docs_with_titles(docs)

        # Format the prompt with the specified language
        formatted_prompt = prompt.format_messages(
            context=context,
            question=user_input,
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
        """Wrapper to make _run conform to Runnable protocol.
        Detected language code should be passed in as an additional argument
        to adhere to Runnable interface."""

        def invoke(
            self,
            input: str,  # noqa: A002 - 'input' shadows built-in, but required by Runnable protocol
            config: RunnableConfig | None = None,
            **kwargs: Any,
        ) -> ChatResponse:
            """Invoke the chain with optional language parameter.

            The language code should be passed explicitly from the caller
            as an additional agument for testing or after language is detected.
            It defaults to DEFAULT_RESPONSE_LANGUAGE if not provided.

            Args:
                input: string from user (required by Runnable protocol)
                config: optional LangChain config (unused, for Runnable compatibility)
                **kwargs: additional arguments, including:
                    language: ISO 639-1 language code (e.g., "en", "fr")

            Returns:
                ChatResponse with answer and metadata
            """
            # Parse detected language code from kwargs; falls back to default
            language = kwargs.get("language", DEFAULT_RESPONSE_LANGUAGE)

            return _run(input, language=language)

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
