"""Streamlit web UI for chatting with Enlightenment philosophers."""

import os
from pathlib import Path
from typing import cast

# Disable database telemetry before any imports
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import streamlit as st
from langchain_core.runnables import RunnableConfig

from src.chains.chat_chain import build_chain
from src.configs.authors import AUTHOR_CONFIGS, DEFAULT_AUTHOR
from src.configs.common import DEFAULT_DB_PATH
from src.i18n import get_message
from src.i18n.keys import (
    ERROR_CHAIN_NOT_INITIALIZED,
    ERROR_GENERATING_RESPONSE,
    SOURCES_ITEM_SEPARATOR_WEB,
    SOURCES_LABEL_WEB,
    SOURCES_NONE,
    SOURCES_SUFFIX_WEB,
)
from src.schemas import ChatResponse
from src.utils.language import detect_language, get_reflecting_message
from src.utils.ollama_health import check_ollama_available


def deduplicate_sources(response: ChatResponse) -> list[str]:
    """Deduplicate source titles while preserving order.

    Args:
        response: ChatResponse with retrieved_source_titles

    Returns:
        List of deduplicated source titles in order of first appearance
    """
    seen = set()
    deduplicated = []
    for title in response.retrieved_source_titles:
        if title not in seen:
            seen.add(title)
            deduplicated.append(title)
    return deduplicated


def format_sources_caption(response: ChatResponse, language: str = "fr") -> str:
    """Format sources as a compact caption.

    Args:
        response: ChatResponse with retrieved_source_titles
        language: ISO 639-1 language code for localized formatting

    Returns:
        Formatted sources string for display as caption
    """
    sources = deduplicate_sources(response)
    if not sources:
        none_label = get_message(SOURCES_NONE, language)
        label = get_message(SOURCES_LABEL_WEB, language)
        suffix = get_message(SOURCES_SUFFIX_WEB, language)
        return f"{label} {none_label}{suffix}"

    label = get_message(SOURCES_LABEL_WEB, language)
    separator = get_message(SOURCES_ITEM_SEPARATOR_WEB, language)
    suffix = get_message(SOURCES_SUFFIX_WEB, language)
    sources_list = separator.join(sources)
    return f"{label} {sources_list}{suffix}"


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "current_author" not in st.session_state:
        st.session_state.current_author = DEFAULT_AUTHOR
    if "current_db_path" not in st.session_state:
        st.session_state.current_db_path = str(DEFAULT_DB_PATH)


def rebuild_chain_if_needed(db_path: str, author: str) -> None:
    """Rebuild chain if configuration has changed.

    Args:
        db_path: Path to database directory
        author: Author key (e.g., "voltaire")
    """
    if (
        st.session_state.chain is None
        or st.session_state.current_author != author
        or st.session_state.current_db_path != db_path
    ):
        try:
            # Check Ollama availability
            check_ollama_available()

            # Build chain
            st.session_state.chain = build_chain(persist_dir=db_path, author=author)
            st.session_state.current_author = author
            st.session_state.current_db_path = db_path

            # Clear message history when switching authors
            st.session_state.messages = []

        except ValueError as e:
            error_msg = f"Configuration error: {e}"
            st.error(error_msg)
            st.session_state.chain = None
        except RuntimeError as e:
            error_msg = f"Ollama error: {e}"
            st.error(error_msg)
            st.session_state.chain = None
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            st.error(error_msg)
            st.session_state.chain = None


def main() -> None:
    """Run the Streamlit chat UI."""
    st.set_page_config(
        page_title="Luminary - Debate Enlightenment Thinkers",
        page_icon="💡",
        layout="centered",
    )

    st.title("💡 Luminary")
    st.caption("Debate Enlightenment Thinkers")

    # Initialize session state
    initialize_session_state()

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        db_path = st.text_input(
            "Database Path",
            value=str(DEFAULT_DB_PATH),
            help="Path to the database directory containing embedded documents",
        )

        # Get available authors
        available_authors = sorted(AUTHOR_CONFIGS.keys())
        author_index = (
            available_authors.index(st.session_state.current_author)
            if st.session_state.current_author in available_authors
            else 0
        )

        author = st.selectbox(
            "Philosopher",
            options=available_authors,
            index=author_index,
            help="Select which philosopher to chat with",
            format_func=lambda x: x.capitalize(),
        )

        st.divider()
        st.caption("Responses are generated using retrieval-augmented generation (RAG) with historical texts.")

    # Rebuild chain if configuration changed
    rebuild_chain_if_needed(db_path, author)

    # Display chat messages
    for message in st.session_state.messages:
        avatar = "💬" if message["role"] == "user" else "🪶"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if "sources" in message:
                st.caption(message["sources"])

    # Chat input
    if prompt := st.chat_input(
        f"Ask {author.capitalize()} a question in French or English..."
    ):
        # Check if chain is available
        if st.session_state.chain is None:
            # Use default language for error message
            _, default_lang, _ = AUTHOR_CONFIGS.get(author, (None, "fr", ""))
            error_msg = get_message(ERROR_CHAIN_NOT_INITIALIZED, default_lang)
            st.error(error_msg)
            return

        # Display user message
        with st.chat_message("user", avatar="💬"):
            st.markdown(prompt)

        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Detect language and get localized reflecting message
        detected_lang = detect_language(prompt, default="fr")
        reflecting_msg = get_reflecting_message(detected_lang, verbose=False)

        # Generate response
        with st.chat_message("assistant", avatar="🪶"):
            with st.spinner(reflecting_msg):
                try:
                    response: ChatResponse = st.session_state.chain.invoke(
                        prompt, config=cast(RunnableConfig, {"language": detected_lang})
                    )
                    st.markdown(response.text)
                    sources_caption = format_sources_caption(response, language=detected_lang)
                    st.caption(sources_caption)

                    # Add assistant message to history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response.text,
                            "sources": sources_caption,
                        }
                    )
                except Exception as e:
                    error_msg = get_message(ERROR_GENERATING_RESPONSE, detected_lang, error=str(e))
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"*{error_msg}*"}
                    )


if __name__ == "__main__":
    main()
