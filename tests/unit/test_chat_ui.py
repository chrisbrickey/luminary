"""Unit tests for Streamlit chat UI."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.prompts import ChatPromptTemplate

from chat_ui import (
    initialize_session_state,
    main,
    rebuild_chain_if_needed,
)
from src.configs.authors import AUTHOR_CONFIGS, AuthorConfig, DEFAULT_AUTHOR
from src.schemas import ChatResponse


# --- Test constants ---


def _fake_gouges_prompt_factory() -> ChatPromptTemplate:
    """Fake prompt factory for test author gouges."""
    return ChatPromptTemplate.from_messages([("system", "You are {author}"), ("human", "{question}")])


# Mock AuthorConfig for test author gouges
GOUGES_CONFIG = AuthorConfig(
    prompt_factory=_fake_gouges_prompt_factory,
    exit_message="Adieu - Olympe",
)


# --- Test fixtures ---


class SessionStateMock(dict):
    """Mock for Streamlit session_state that supports both dict and attribute access."""

    def __getattr__(self, key: str) -> object:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'SessionStateMock' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: object) -> None:
        self[key] = value


@pytest.fixture
def mock_author_configs_with_gouges():
    """Patch AUTHOR_CONFIGS with real and test authors."""
    # Include both real voltaire config and fake gouges config
    test_configs = {**AUTHOR_CONFIGS, "gouges": GOUGES_CONFIG}
    with patch("chat_ui.AUTHOR_CONFIGS", test_configs):
        yield


# --- Test session state initialization ---


@patch("chat_ui.st")
def test_initialize_session_state_empty(mock_st: Mock) -> None:
    """Test session state initialization when state is empty."""
    mock_st.session_state = SessionStateMock()

    initialize_session_state()

    assert mock_st.session_state["messages"] == []
    assert mock_st.session_state["chain"] is None
    assert mock_st.session_state["current_author"] == DEFAULT_AUTHOR
    assert mock_st.session_state["show_exit_message"] is None


@patch("chat_ui.st")
def test_initialize_session_state_existing(mock_st: Mock) -> None:
    """Test session state initialization preserves existing values."""
    existing_messages = [{"role": "user", "content": "test"}]
    mock_chain = Mock()

    mock_st.session_state = SessionStateMock(
        {
            "messages": existing_messages,
            "chain": mock_chain,
            "current_author": "gouges",
        }
    )

    initialize_session_state()

    # Should not overwrite existing values
    assert mock_st.session_state["messages"] == existing_messages
    assert mock_st.session_state["chain"] == mock_chain
    assert mock_st.session_state["current_author"] == "gouges"


# --- Test chain rebuilding ---


@patch("chat_ui.check_ollama_available")
@patch("chat_ui.build_chain")
@patch("chat_ui.st")
def test_rebuild_chain_if_needed_first_time(
    mock_st: Mock, mock_build_chain: Mock, mock_check_ollama: Mock
) -> None:
    """Test chain is built on first call with default configs."""
    mock_st.session_state = SessionStateMock(
        {
            "chain": None,
            "current_author": DEFAULT_AUTHOR,
            "messages": [],
        }
    )
    mock_chain = Mock()
    mock_build_chain.return_value = mock_chain

    rebuild_chain_if_needed(DEFAULT_AUTHOR)

    mock_check_ollama.assert_called_once()
    mock_build_chain.assert_called_once_with(author=DEFAULT_AUTHOR)
    assert mock_st.session_state["chain"] == mock_chain


@patch("chat_ui.check_ollama_available")
@patch("chat_ui.build_chain")
@patch("chat_ui.st")
def test_rebuild_chain_if_needed_author_changed(
    mock_st: Mock, mock_build_chain: Mock, mock_check_ollama: Mock
) -> None:
    """Test chain is rebuilt when author changes."""
    old_chain = Mock()
    mock_st.session_state = SessionStateMock(
        {
            "chain": old_chain,
            "current_author": "voltaire",
            "messages": [{"role": "user", "content": "old message"}],
        }
    )
    new_chain = Mock()
    mock_build_chain.return_value = new_chain

    rebuild_chain_if_needed("gouges")

    mock_check_ollama.assert_called_once()
    mock_build_chain.assert_called_once_with(author="gouges")
    assert mock_st.session_state["chain"] == new_chain
    assert mock_st.session_state["current_author"] == "gouges"
    # Messages should be cleared when switching authors
    assert mock_st.session_state["messages"] == []


@patch("chat_ui.check_ollama_available")
@patch("chat_ui.build_chain")
@patch("chat_ui.st")
def test_rebuild_chain_if_needed_no_change(
    mock_st: Mock, mock_build_chain: Mock, mock_check_ollama: Mock
) -> None:
    """Test chain is not rebuilt when default configs remain unchanged."""
    existing_chain = Mock()
    mock_st.session_state = SessionStateMock(
        {
            "chain": existing_chain,
            "current_author": DEFAULT_AUTHOR,
            "messages": [{"role": "user", "content": "existing message"}],
        }
    )

    rebuild_chain_if_needed(DEFAULT_AUTHOR)

    # Should not rebuild
    mock_check_ollama.assert_not_called()
    mock_build_chain.assert_not_called()
    assert mock_st.session_state["chain"] == existing_chain
    # Messages should be preserved
    assert len(mock_st.session_state["messages"]) == 1


@patch("chat_ui.check_ollama_available")
@patch("chat_ui.build_chain")
@patch("chat_ui.st")
def test_rebuild_chain_value_error(
    mock_st: Mock, mock_build_chain: Mock, mock_check_ollama: Mock
) -> None:
    """Test chain rebuild handles ValueError from build_chain."""
    mock_st.session_state = SessionStateMock(
        {
            "chain": None,
            "current_author": DEFAULT_AUTHOR,
            "messages": [],
        }
    )
    mock_st.error = Mock()
    mock_build_chain.side_effect = ValueError("Invalid author")

    rebuild_chain_if_needed("invalid_author")

    mock_st.error.assert_called_once()
    assert "Configuration error" in mock_st.error.call_args[0][0]
    assert mock_st.session_state["chain"] is None


@patch("chat_ui.check_ollama_available")
@patch("chat_ui.build_chain")
@patch("chat_ui.st")
def test_rebuild_chain_runtime_error(
    mock_st: Mock, mock_build_chain: Mock, mock_check_ollama: Mock
) -> None:
    """Test chain rebuild handles RuntimeError from check_ollama_available."""
    mock_st.session_state = SessionStateMock(
        {
            "chain": None,
            "current_author": DEFAULT_AUTHOR,
            "messages": [],
        }
    )
    mock_st.error = Mock()
    mock_check_ollama.side_effect = RuntimeError("Ollama not running")

    rebuild_chain_if_needed(DEFAULT_AUTHOR)

    mock_st.error.assert_called_once()
    assert "Ollama error" in mock_st.error.call_args[0][0]
    assert mock_st.session_state["chain"] is None


@patch("chat_ui.check_ollama_available")
@patch("chat_ui.build_chain")
@patch("chat_ui.st")
def test_rebuild_chain_unexpected_error(
    mock_st: Mock, mock_build_chain: Mock, mock_check_ollama: Mock
) -> None:
    """Test chain rebuild handles unexpected exceptions."""
    mock_st.session_state = SessionStateMock(
        {
            "chain": None,
            "current_author": DEFAULT_AUTHOR,
            "messages": [],
        }
    )
    mock_st.error = Mock()
    mock_build_chain.side_effect = Exception("Unexpected error")

    rebuild_chain_if_needed(DEFAULT_AUTHOR)

    mock_st.error.assert_called_once()
    assert "Unexpected error" in mock_st.error.call_args[0][0]
    assert mock_st.session_state["chain"] is None


# --- Test main UI ---


@patch("chat_ui.rebuild_chain_if_needed")
@patch("chat_ui.initialize_session_state")
@patch("chat_ui.st")
def test_main_with_defaults(
    mock_st: Mock, mock_init: Mock, mock_rebuild: Mock
) -> None:
    """Test main() uses production default configs."""
    mock_st.session_state = SessionStateMock(
        {
            "messages": [],
            "chain": None,
            "current_author": DEFAULT_AUTHOR,
            "show_exit_message": None,
        }
    )
    mock_st.chat_input.return_value = None  # No user input
    mock_st.selectbox.return_value = DEFAULT_AUTHOR
    mock_st.button.return_value = False

    main()

    # Verify rebuild_chain_if_needed called with default values
    mock_rebuild.assert_called_once_with(DEFAULT_AUTHOR)

    # Verify available authors come from production config
    selectbox_call = mock_st.selectbox.call_args
    available_authors = selectbox_call[1]["options"]
    assert available_authors == sorted(AUTHOR_CONFIGS.keys())

    # Verify default author is used as initial selection
    author_index = selectbox_call[1]["index"]
    assert available_authors[author_index] == DEFAULT_AUTHOR


@patch("chat_ui.rebuild_chain_if_needed")
@patch("chat_ui.initialize_session_state")
@patch("chat_ui.st")
def test_main_renders_title(
    mock_st: Mock, mock_init: Mock, mock_rebuild: Mock
) -> None:
    """Test that main renders the title and caption with correct text."""
    mock_st.session_state = SessionStateMock(
        {
            "messages": [],
            "chain": None,
            "current_author": DEFAULT_AUTHOR,
            "show_exit_message": None,
        }
    )
    mock_st.chat_input.return_value = None  # No user input
    mock_st.selectbox.return_value = DEFAULT_AUTHOR
    mock_st.button.return_value = False

    main()

    # Assert on the specific strings visible on page load
    mock_st.title.assert_called_once_with("💡 Luminary")

    # Verify caption calls include the main subtitle
    caption_calls = [call[0][0] for call in mock_st.caption.call_args_list]
    assert "Debate Enlightenment Thinkers" in caption_calls

    # Verify sidebar caption about RAG is shown with exact text
    expected_rag_caption = ("Responses are generated using retrieval-augmented generation (RAG) with historical texts.")
    assert expected_rag_caption in caption_calls

    # Verify chat input placeholder text uses default author
    expected_placeholder = f"Ask {DEFAULT_AUTHOR.capitalize()} a question in French or English..."
    mock_st.chat_input.assert_called_once_with(expected_placeholder)


@patch("chat_ui.rebuild_chain_if_needed")
@patch("chat_ui.initialize_session_state")
@patch("chat_ui.st")
def test_main_initializes_session_state(
    mock_st: Mock, mock_init: Mock, mock_rebuild: Mock
) -> None:
    """Test that main calls initialize_session_state."""
    mock_st.session_state = SessionStateMock(
        {
            "messages": [],
            "chain": None,
            "current_author": DEFAULT_AUTHOR,
            "show_exit_message": None,
        }
    )
    mock_st.chat_input.return_value = None
    mock_st.selectbox.return_value = DEFAULT_AUTHOR
    mock_st.button.return_value = False

    main()

    mock_init.assert_called_once()


@patch("chat_ui.rebuild_chain_if_needed")
@patch("chat_ui.initialize_session_state")
@patch("chat_ui.st")
def test_main_sidebar_config(
    mock_st: Mock, mock_init: Mock, mock_rebuild: Mock
) -> None:
    """Test that main renders sidebar configuration."""
    mock_st.session_state = SessionStateMock(
        {
            "messages": [],
            "chain": None,
            "current_author": DEFAULT_AUTHOR,
            "show_exit_message": None,
        }
    )
    mock_st.chat_input.return_value = None
    mock_st.selectbox.return_value = DEFAULT_AUTHOR
    mock_st.button.return_value = False

    # Mock sidebar context manager
    mock_sidebar = MagicMock()
    mock_st.sidebar.__enter__ = Mock(return_value=mock_sidebar)
    mock_st.sidebar.__exit__ = Mock(return_value=False)

    main()

    # Verify selectbox was called (it's called within sidebar context)
    mock_st.selectbox.assert_called_once()


@patch("chat_ui.rebuild_chain_if_needed")
@patch("chat_ui.initialize_session_state")
@patch("chat_ui.st")
def test_main_rebuilds_chain(
    mock_st: Mock, mock_init: Mock, mock_rebuild: Mock, mock_author_configs_with_gouges
) -> None:
    """Test that main rebuilds chain with sidebar config."""
    mock_st.session_state = SessionStateMock(
        {
            "messages": [],
            "chain": None,
            "current_author": "voltaire",
            "show_exit_message": None,
        }
    )
    mock_st.chat_input.return_value = None
    mock_st.selectbox.return_value = "gouges"
    mock_st.button.return_value = False

    main()

    mock_rebuild.assert_called_once_with("gouges")


@patch("chat_ui.rebuild_chain_if_needed")
@patch("chat_ui.initialize_session_state")
@patch("chat_ui.st")
def test_main_clear_conversation_button_shows_exit_message(
    mock_st: Mock, mock_init: Mock, mock_rebuild: Mock, mock_author_configs_with_gouges
) -> None:
    """Test that Clear conversation button stores exit message in session state for display after rerun."""
    mock_st.session_state = SessionStateMock(
        {
            "messages": [{"role": "user", "content": "Previous message"}],
            "chain": Mock(),
            "current_author": "gouges",
            "show_exit_message": None,
        }
    )
    mock_st.chat_input.return_value = None
    mock_st.selectbox.return_value = "gouges"
    mock_st.button.return_value = True  # Trigger "Clear conversation" button
    mock_st.toast = Mock()
    mock_st.rerun = Mock(side_effect=Exception("Rerun called"))  # Prevent actual rerun

    # Run - should raise due to rerun
    try:
        main()
    except Exception as e:
        if str(e) != "Rerun called":
            raise

    # Verify exit message was stored in session state (not displayed immediately)
    assert mock_st.session_state["show_exit_message"] == "Adieu - Olympe"

    # Verify messages were cleared
    assert mock_st.session_state["messages"] == []

    # Toast should not be called during button click (it's called after rerun)
    mock_st.toast.assert_not_called()


@patch("chat_ui.rebuild_chain_if_needed")
@patch("chat_ui.initialize_session_state")
@patch("chat_ui.st")
def test_main_shows_exit_message_toast_after_rerun(
    mock_st: Mock, mock_init: Mock, mock_rebuild: Mock, mock_author_configs_with_gouges
) -> None:
    """Test that exit message toast is displayed after rerun when flag is set."""
    mock_st.session_state = SessionStateMock(
        {
            "messages": [],
            "chain": Mock(),
            "current_author": "gouges",
            "show_exit_message": "Adieu - Olympe",  # Flag set from previous run
        }
    )
    mock_st.chat_input.return_value = None
    mock_st.selectbox.return_value = "gouges"
    mock_st.button.return_value = False  # Button not clicked this time
    mock_st.toast = Mock()

    main()

    # Verify toast was called with the exit message
    mock_st.toast.assert_called_once_with("Adieu - Olympe", icon="🪶")

    # Verify flag was cleared after displaying
    assert mock_st.session_state["show_exit_message"] is None


@patch("chat_ui.rebuild_chain_if_needed")
@patch("chat_ui.initialize_session_state")
@patch("chat_ui.st")
def test_main_displays_existing_messages(
    mock_st: Mock, mock_init: Mock, mock_rebuild: Mock
) -> None:
    """Test that main displays existing messages from session state."""
    mock_st.session_state = SessionStateMock(
        {
            "messages": [
                {"role": "user", "content": "Question 1"},
                {
                    "role": "assistant",
                    "content": "Answer 1",
                    "sources": "*Sources: Source A*",
                },
            ],
            "chain": Mock(),
            "current_author": DEFAULT_AUTHOR,
            "show_exit_message": None,
        }
    )
    mock_st.chat_input.return_value = None
    mock_st.selectbox.return_value = DEFAULT_AUTHOR
    mock_st.button.return_value = False  # Don't trigger "Clear conversation" button

    # Mock chat_message context manager
    mock_chat_message = MagicMock()
    mock_st.chat_message.return_value.__enter__ = Mock(return_value=mock_chat_message)
    mock_st.chat_message.return_value.__exit__ = Mock(return_value=False)

    main()

    # Verify chat_message was called for each message with correct avatars
    assert mock_st.chat_message.call_count == 2

    # First call should be user message with speech bubble avatar
    first_call = mock_st.chat_message.call_args_list[0]
    assert first_call[0][0] == "user"
    assert first_call[1]["avatar"] == "💬"

    # Second call should be assistant message with feather avatar
    second_call = mock_st.chat_message.call_args_list[1]
    assert second_call[0][0] == "assistant"
    assert second_call[1]["avatar"] == "🪶"

    # Verify markdown was called for message content
    assert mock_st.markdown.call_count >= 2


@patch("chat_ui.rebuild_chain_if_needed")
@patch("chat_ui.initialize_session_state")
@patch("chat_ui.st")
def test_main_no_input_returns_early(
    mock_st: Mock, mock_init: Mock, mock_rebuild: Mock
) -> None:
    """Test that main returns early when no input is provided."""
    mock_st.session_state = SessionStateMock(
        {
            "messages": [],
            "chain": Mock(),
            "current_author": DEFAULT_AUTHOR,
            "show_exit_message": None,
        }
    )
    mock_st.chat_input.return_value = None  # No input
    mock_st.selectbox.return_value = DEFAULT_AUTHOR
    mock_st.button.return_value = False

    main()

    # Chain should not be invoked
    mock_st.session_state["chain"].invoke.assert_not_called()


@patch("chat_ui.rebuild_chain_if_needed")
@patch("chat_ui.initialize_session_state")
@patch("chat_ui.st")
def test_main_processes_user_input(
    mock_st: Mock, mock_init: Mock, mock_rebuild: Mock
) -> None:
    """Test that main processes user input and invokes chain."""
    mock_chain = Mock()
    mock_response = ChatResponse(
        text="Test response",
        retrieved_passage_ids=["id1"],
        retrieved_contexts=["context1"],
        retrieved_source_titles=["Source A"],
        language="fr",
    )
    mock_chain.invoke.return_value = mock_response

    mock_st.session_state = SessionStateMock(
        {
            "messages": [],
            "chain": mock_chain,
            "current_author": DEFAULT_AUTHOR,
            "show_exit_message": None,
        }
    )
    mock_st.chat_input.return_value = "What is tolerance?"
    mock_st.selectbox.return_value = DEFAULT_AUTHOR
    mock_st.button.return_value = False

    # Mock chat_message and spinner context managers
    mock_chat_message = MagicMock()
    mock_st.chat_message.return_value.__enter__ = Mock(return_value=mock_chat_message)
    mock_st.chat_message.return_value.__exit__ = Mock(return_value=False)

    mock_spinner = MagicMock()
    mock_st.spinner.return_value.__enter__ = Mock(return_value=mock_spinner)
    mock_st.spinner.return_value.__exit__ = Mock(return_value=False)

    main()

    # Verify chain was invoked
    mock_chain.invoke.assert_called_once_with("What is tolerance?")

    # Verify chat_message was called with correct avatars
    assert mock_st.chat_message.call_count == 2

    # First call should be user message with speech bubble avatar
    first_call = mock_st.chat_message.call_args_list[0]
    assert first_call[0][0] == "user"
    assert first_call[1]["avatar"] == "💬"

    # Second call should be assistant message with feather avatar
    second_call = mock_st.chat_message.call_args_list[1]
    assert second_call[0][0] == "assistant"
    assert second_call[1]["avatar"] == "🪶"

    # Verify messages were added to session state
    assert len(mock_st.session_state["messages"]) == 2
    assert mock_st.session_state["messages"][0]["role"] == "user"
    assert mock_st.session_state["messages"][0]["content"] == "What is tolerance?"
    assert mock_st.session_state["messages"][1]["role"] == "assistant"
    assert mock_st.session_state["messages"][1]["content"] == "Test response"


@patch("chat_ui.rebuild_chain_if_needed")
@patch("chat_ui.initialize_session_state")
@patch("chat_ui.st")
def test_main_shows_sources_caption(
    mock_st: Mock, mock_init: Mock, mock_rebuild: Mock
) -> None:
    """Test that main displays sources caption with response."""
    mock_chain = Mock()
    mock_response = ChatResponse(
        text="Test response",
        retrieved_passage_ids=["id1", "id2"],
        retrieved_contexts=["context1", "context2"],
        retrieved_source_titles=["Source A", "Source B"],
        language="fr",
    )
    mock_chain.invoke.return_value = mock_response

    mock_st.session_state = SessionStateMock(
        {
            "messages": [],
            "chain": mock_chain,
            "current_author": DEFAULT_AUTHOR,
            "show_exit_message": None,
        }
    )
    mock_st.chat_input.return_value = "Test question"
    mock_st.selectbox.return_value = DEFAULT_AUTHOR
    mock_st.button.return_value = False

    # Mock context managers
    mock_chat_message = MagicMock()
    mock_st.chat_message.return_value.__enter__ = Mock(return_value=mock_chat_message)
    mock_st.chat_message.return_value.__exit__ = Mock(return_value=False)

    mock_spinner = MagicMock()
    mock_st.spinner.return_value.__enter__ = Mock(return_value=mock_spinner)
    mock_st.spinner.return_value.__exit__ = Mock(return_value=False)

    main()

    # Verify markdown was called with properly formatted sources
    markdown_calls = [call[0][0] for call in mock_st.markdown.call_args_list]

    # Find the sources call (should contain bold markers and bullets)
    sources_calls = [call for call in markdown_calls if "**Sources" in call]
    assert len(sources_calls) == 1, f"Expected 1 sources call, got {len(sources_calls)}"

    # Uses DEFAULT_RESPONSE_LANGUAGE (en), so no space before colon
    assert sources_calls[0] == "**Sources:**\n- Source A\n- Source B"


@patch("chat_ui.rebuild_chain_if_needed")
@patch("chat_ui.initialize_session_state")
@patch("chat_ui.st")
def test_main_chain_not_initialized_error(
    mock_st: Mock, mock_init: Mock, mock_rebuild: Mock
) -> None:
    """Test that main shows error when chain is not initialized."""
    mock_st.session_state = SessionStateMock(
        {
            "messages": [],
            "chain": None,  # Chain not initialized
            "current_author": DEFAULT_AUTHOR,
            "show_exit_message": None,
        }
    )
    mock_st.chat_input.return_value = "Test question"
    mock_st.selectbox.return_value = DEFAULT_AUTHOR
    mock_st.button.return_value = False
    mock_st.error = Mock()

    main()

    # Verify error was shown
    mock_st.error.assert_called_once()
    assert "not initialized" in mock_st.error.call_args[0][0]


@patch("chat_ui.rebuild_chain_if_needed")
@patch("chat_ui.initialize_session_state")
@patch("chat_ui.st")
def test_main_chain_invocation_error(
    mock_st: Mock, mock_init: Mock, mock_rebuild: Mock
) -> None:
    """Test that main handles chain invocation errors gracefully."""
    mock_chain = Mock()
    mock_chain.invoke.side_effect = Exception("Chain error")

    mock_st.session_state = SessionStateMock(
        {
            "messages": [],
            "chain": mock_chain,
            "current_author": DEFAULT_AUTHOR,
            "show_exit_message": None,
        }
    )
    mock_st.chat_input.return_value = "Test question"
    mock_st.selectbox.return_value = DEFAULT_AUTHOR
    mock_st.button.return_value = False
    mock_st.error = Mock()

    # Mock context managers
    mock_chat_message = MagicMock()
    mock_st.chat_message.return_value.__enter__ = Mock(return_value=mock_chat_message)
    mock_st.chat_message.return_value.__exit__ = Mock(return_value=False)

    mock_spinner = MagicMock()
    mock_st.spinner.return_value.__enter__ = Mock(return_value=mock_spinner)
    mock_st.spinner.return_value.__exit__ = Mock(return_value=False)

    main()

    # Verify error was shown
    mock_st.error.assert_called_once()
    assert "Error generating response" in mock_st.error.call_args[0][0]

    # Verify error message was added to chat history
    assert len(mock_st.session_state["messages"]) == 2
    assert mock_st.session_state["messages"][1]["role"] == "assistant"
    assert "Error generating response" in mock_st.session_state["messages"][1]["content"]
