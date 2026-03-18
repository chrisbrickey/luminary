"""Streamlit web UI for chatting with Enlightenment philosophers."""

import os

# Disable database telemetry before any imports
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import streamlit as st

from src.chains.chat_chain import build_chain
from src.configs.authors import AUTHOR_CONFIGS, DEFAULT_AUTHOR
from src.configs.common import DEFAULT_RESPONSE_LANGUAGE
from src.i18n import get_message
from src.i18n.keys import (
    ERROR_CHAIN_NOT_INITIALIZED,
    ERROR_GENERATING_RESPONSE,
    STATUS_REFLECTING,
)
from src.schemas import ChatResponse
from src.utils.formatting import format_sources
from src.utils.language import detect_language
from src.utils.ollama_health import check_ollama_available

author_avatar: str = "🪶"
user_avatar: str = "👤"
product_emoji: str = "💡"

def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "current_author" not in st.session_state:
        st.session_state.current_author = DEFAULT_AUTHOR
    if "show_exit_message" not in st.session_state:
        st.session_state.show_exit_message = None


def initialize_or_rebuild_chain(author: str) -> None:
    """Build the chat chain prior to invoking it and save it to the state
    so that it can be later invoked.

    This method will be called again if the chain needs to be rebuilt
    (e.g. if user changes selection of author in the browser).

    Args:
        author: name of the selected author
    """
    if (
        st.session_state.chain is None
        or st.session_state.current_author != author
    ):
        try:
            # Check Ollama availability
            check_ollama_available()

            # Build chain
            st.session_state.chain = build_chain(author=author)
            st.session_state.current_author = author

            # Clear message history when switching authors
            st.session_state.messages = []

        except ValueError as e:
            st.error(f"Configuration error: {e}")
            st.session_state.chain = None
        except RuntimeError as e:
            st.error(f"Ollama error: {e}")
            st.session_state.chain = None
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.session_state.chain = None


def main() -> None:
    """Run the Streamlit chat UI."""
    st.set_page_config(
        page_title="Luminary - Debate Enlightenment Thinkers",
        page_icon=product_emoji,
        layout="centered",
    )

    st.title(f"{product_emoji} Luminary")
    st.caption("Debate Enlightenment Thinkers")

    # Initialize session state
    initialize_session_state()

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

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

        # Clear conversation button
        if st.button("Clear conversation"):
            exit_msg = AUTHOR_CONFIGS[author].exit_message
            st.session_state.messages = []
            st.session_state.show_exit_message = exit_msg
            st.rerun()

        st.divider()
        st.caption("Responses are generated using retrieval-augmented generation (RAG) with historical texts.")

    initialize_or_rebuild_chain(author)

    # Show exit message if conversation was just cleared
    if st.session_state.show_exit_message:
        st.toast(st.session_state.show_exit_message, icon=author_avatar)
        st.session_state.show_exit_message = None

    # Display chat messages
    for message in st.session_state.messages:
        avatar = user_avatar if message["role"] == "user" else author_avatar
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
            st.error(get_message(ERROR_CHAIN_NOT_INITIALIZED, DEFAULT_RESPONSE_LANGUAGE))
            return

        # Display user message
        with st.chat_message("user", avatar=user_avatar):
            st.markdown(prompt)

        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Detect language before generating response
        detected_lang = detect_language(prompt)

        # Generate response with localized spinner and detected language
        with st.chat_message("assistant", avatar=author_avatar):
            with st.spinner(get_message(STATUS_REFLECTING, detected_lang)):
                try:

                    # Invoke the chat chain, which calls the LLM, with the prompt and detected language
                    response: ChatResponse = st.session_state.chain.invoke(
                        prompt, language=detected_lang
                    )
                    st.markdown(response.text)

                    # Format sources including some hard-coded strings localized by detected language
                    sources_caption = format_sources(response, detected_lang)
                    st.markdown(sources_caption)

                    # Add assistant message to history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response.text,
                            "sources": sources_caption,
                        }
                    )
                except Exception as e:
                    error_msg = get_message(
                        ERROR_GENERATING_RESPONSE, detected_lang, error=str(e)
                    )
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"*{error_msg}*"}
                    )


if __name__ == "__main__":
    main()
