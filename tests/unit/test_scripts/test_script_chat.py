"""Unit tests for scripts/chat.py."""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from langchain_core.prompts import ChatPromptTemplate

from src.configs.authors import AuthorConfig, DEFAULT_AUTHOR
from src.configs.common import DEFAULT_DB_PATH
from src.schemas import ChatResponse

# Test constants (for use in tests - not necessarily the actual defaults)
TEST_AUTHOR = "condorcet"
TEST_DB_PATH = "data/chroma_db"
TEST_QUESTION = "What progress do you foresee for humanity?"
TEST_RESPONSE_TEXT = "Humanity will advance indefinitely through reason, science, and moral improvement."
TEST_CHUNK_IDS = ["xyz789abc123", "def456ghi789"]
TEST_CONTEXTS = [
    "Context from first chunk about scientific advancement.",
    "Context from second chunk about women's rights.",
]
TEST_SOURCE_TITLES = [
    "Esquisse d'un tableau historique, Page 12",
    "Esquisse d'un tableau historique, Page 9",
    "Esquisse d'un tableau historique, Page 12" # duplicate
]


def _fake_prompt_factory() -> ChatPromptTemplate:
    """Fake prompt factory for test author."""
    return ChatPromptTemplate.from_messages([("system", "You are {author}"), ("human", "{question}")])


# Mock AuthorConfig for test author
TEST_AUTHOR_CONFIG = AuthorConfig(
    prompt_factory=_fake_prompt_factory,
    exit_message="Au revoir - Condorcet",
)


def create_mock_response(
    text: str = TEST_RESPONSE_TEXT,
    chunk_ids: list[str] | None = None,
    contexts: list[str] | None = None,
    source_titles: list[str] | None = None,
    language: str = "fr",
) -> ChatResponse:
    """Create a mock ChatResponse for testing.

    Args:
        text: Response text
        chunk_ids: List of chunk IDs (None = use defaults)
        contexts: List of retrieved contexts (None = use defaults)
        source_titles: List of source titles (None = use defaults)
        language: Response language

    Returns:
        ChatResponse instance
    """
    return ChatResponse(
        text=text,
        retrieved_passage_ids=TEST_CHUNK_IDS if chunk_ids is None else chunk_ids,
        retrieved_contexts=TEST_CONTEXTS if contexts is None else contexts,
        retrieved_source_titles=TEST_SOURCE_TITLES if source_titles is None else source_titles,
        language=language,
    )


class TestDeduplicateSources:
    """Test deduplicate_sources() helper function."""

    def test_deduplicate_preserves_order(self) -> None:
        """Test that deduplication preserves first appearance order."""
        from scripts.chat import deduplicate_sources

        response = create_mock_response()
        result = deduplicate_sources(response)

        # Should have 2 unique titles in order of first appearance
        assert result == [
            "Esquisse d'un tableau historique, Page 12",
            "Esquisse d'un tableau historique, Page 9",
        ]

    def test_empty_sources(self) -> None:
        """Test deduplication with empty sources list."""
        from scripts.chat import deduplicate_sources

        response = create_mock_response(source_titles=[])
        result = deduplicate_sources(response)

        assert result == []

    def test_all_unique_sources(self) -> None:
        """Test deduplication when all sources are unique."""
        from scripts.chat import deduplicate_sources

        unique_sources = ["Source A", "Source B", "Source C"]
        response = create_mock_response(source_titles=unique_sources)
        result = deduplicate_sources(response)

        assert result == unique_sources


class TestFormatSourcesFooter:
    """Test format_sources_footer() helper function."""

    def test_format_with_sources(self) -> None:
        """Test formatting with deduplicated sources."""
        from scripts.chat import format_sources_footer

        response = create_mock_response()
        result = format_sources_footer(response)

        expected = (
            "\nSources:\n"
            "  - Esquisse d'un tableau historique, Page 12\n"
            "  - Esquisse d'un tableau historique, Page 9"
        )
        assert result == expected

    def test_format_with_no_sources(self) -> None:
        """Test formatting when no sources available."""
        from scripts.chat import format_sources_footer

        response = create_mock_response(source_titles=[])
        result = format_sources_footer(response)

        assert result == "\nSources: none"


class TestFormatChunksOutput:
    """Test format_chunks_output() helper function."""

    def test_format_with_chunks(self) -> None:
        """Test formatting with retrieved chunks."""
        from scripts.chat import format_chunks_output

        response = create_mock_response()
        result = format_chunks_output(response)

        # Test structure
        assert result.startswith("\nRetrieved chunks:")

        # Test that each chunk ID appears in brackets with its context
        assert f"[{TEST_CHUNK_IDS[0]}]" in result
        assert f"[{TEST_CHUNK_IDS[1]}]" in result
        assert TEST_CONTEXTS[0] in result
        assert TEST_CONTEXTS[1] in result

        # Test ordering: first chunk ID should appear before second chunk ID
        first_chunk_pos = result.index(f"[{TEST_CHUNK_IDS[0]}]")
        second_chunk_pos = result.index(f"[{TEST_CHUNK_IDS[1]}]")
        assert first_chunk_pos < second_chunk_pos

        # Test that context appears after its corresponding ID
        first_context_pos = result.index(TEST_CONTEXTS[0])
        assert first_chunk_pos < first_context_pos

    def test_format_with_no_chunks(self) -> None:
        """Test formatting when no chunks retrieved."""
        from scripts.chat import format_chunks_output

        response = create_mock_response(chunk_ids=[], contexts=[])
        result = format_chunks_output(response)

        assert result == "\nRetrieved chunks: none"


@pytest.fixture(autouse=True)
def mock_author_configs():
    """Patch AUTHOR_CONFIGS with test author for all tests."""
    with patch("scripts.chat.AUTHOR_CONFIGS", {TEST_AUTHOR: TEST_AUTHOR_CONFIG}):
        yield


class TestRunInteractiveChat:
    """Test run_interactive_chat() function."""

    @patch("scripts.chat.build_chain")
    @patch("scripts.chat.check_ollama_available")
    @patch("builtins.input")
    def test_welcome_text_displays(
        self,
        mock_input: MagicMock,
        mock_ollama: MagicMock,
        mock_build_chain: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that welcome text is displayed correctly."""
        from scripts.chat import run_interactive_chat

        # Setup mocks
        mock_ollama.return_value = None
        mock_chain = MagicMock()
        mock_build_chain.return_value = mock_chain
        mock_input.side_effect = ["quit"]

        # Run
        run_interactive_chat(
            db_path=Path(TEST_DB_PATH),
            author=TEST_AUTHOR,
            show_chunks=False,
            verbose=False,
        )

        # Capture output
        captured = capsys.readouterr()

        # Verify welcome text
        assert "Welcome to Luminary!" in captured.out
        assert "You are now chatting with Condorcet." in captured.out
        assert "Type 'quit' or press Ctrl+C to exit." in captured.out

    @patch("scripts.chat.build_chain")
    @patch("scripts.chat.check_ollama_available")
    @patch("builtins.input")
    def test_user_prompt_is_you_vous(
        self,
        mock_input: MagicMock,
        mock_ollama: MagicMock,
        mock_build_chain: MagicMock,
    ) -> None:
        """Test that user prompt is 'You/Vous:'."""
        from scripts.chat import run_interactive_chat

        # Setup mocks
        mock_ollama.return_value = None
        mock_chain = MagicMock()
        mock_build_chain.return_value = mock_chain
        mock_input.side_effect = ["quit"]

        # Run
        run_interactive_chat(
            db_path=Path(TEST_DB_PATH),
            author=TEST_AUTHOR,
            show_chunks=False,
            verbose=False,
        )

        # Verify input was called with correct prompt
        mock_input.assert_called_with("You/Vous: ")

    @patch("scripts.chat.build_chain")
    @patch("scripts.chat.check_ollama_available")
    @patch("builtins.input")
    def test_author_prompt_is_capitalized_name(
        self,
        mock_input: MagicMock,
        mock_ollama: MagicMock,
        mock_build_chain: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that author response is prefixed with capitalized author name."""
        from scripts.chat import run_interactive_chat

        # Setup mocks
        mock_ollama.return_value = None
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = create_mock_response()
        mock_build_chain.return_value = mock_chain
        mock_input.side_effect = [TEST_QUESTION, "quit"]

        # Run
        run_interactive_chat(
            db_path=Path(TEST_DB_PATH),
            author=TEST_AUTHOR,
            show_chunks=False,
            verbose=False,
        )

        # Capture output
        captured = capsys.readouterr()

        # Verify author name is capitalized and used as prompt
        assert "Condorcet:" in captured.out
        assert TEST_RESPONSE_TEXT in captured.out

    @patch("scripts.chat.build_chain")
    @patch("scripts.chat.check_ollama_available")
    @patch("builtins.input")
    def test_basic_question_answer_flow(
        self,
        mock_input: MagicMock,
        mock_ollama: MagicMock,
        mock_build_chain: MagicMock,
    ) -> None:
        """Test basic question-answer flow with quit."""
        from scripts.chat import run_interactive_chat

        # Setup mocks
        mock_ollama.return_value = None
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = create_mock_response()
        mock_build_chain.return_value = mock_chain
        mock_input.side_effect = [TEST_QUESTION, "quit"]

        # Run
        run_interactive_chat(
            db_path=Path(TEST_DB_PATH),
            author=TEST_AUTHOR,
            show_chunks=False,
            verbose=False,
        )

        # Verify Ollama check
        mock_ollama.assert_called_once()

        # Verify chain built with correct params
        mock_build_chain.assert_called_once_with(
            persist_dir=TEST_DB_PATH, author=TEST_AUTHOR
        )

        # Verify chain invoked once
        mock_chain.invoke.assert_called_once_with(TEST_QUESTION)

    @patch("scripts.chat.build_chain")
    @patch("scripts.chat.check_ollama_available")
    @patch("builtins.input")
    def test_show_chunks_flag(
        self,
        mock_input: MagicMock,
        mock_ollama: MagicMock,
        mock_build_chain: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that --show-chunks displays retrieved chunks."""
        from scripts.chat import run_interactive_chat

        # Setup mocks
        mock_ollama.return_value = None
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = create_mock_response()
        mock_build_chain.return_value = mock_chain
        mock_input.side_effect = [TEST_QUESTION, "quit"]

        # Run with show_chunks=True
        run_interactive_chat(
            db_path=Path(TEST_DB_PATH),
            author=TEST_AUTHOR,
            show_chunks=True,
            verbose=False,
        )

        # Capture output
        captured = capsys.readouterr()

        # Verify chunks are displayed
        assert "Retrieved chunks:" in captured.out
        assert TEST_CHUNK_IDS[0] in captured.out
        assert TEST_CONTEXTS[0] in captured.out

    @patch("scripts.chat.build_chain")
    @patch("scripts.chat.check_ollama_available")
    @patch("builtins.input")
    def test_chunks_not_shown_by_default(
        self,
        mock_input: MagicMock,
        mock_ollama: MagicMock,
        mock_build_chain: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that chunks are not shown without --show-chunks flag."""
        from scripts.chat import run_interactive_chat

        # Setup mocks
        mock_ollama.return_value = None
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = create_mock_response()
        mock_build_chain.return_value = mock_chain
        mock_input.side_effect = [TEST_QUESTION, "quit"]

        # Run with show_chunks=False
        run_interactive_chat(
            db_path=Path(TEST_DB_PATH),
            author=TEST_AUTHOR,
            show_chunks=False,
            verbose=False,
        )

        # Capture output
        captured = capsys.readouterr()

        # Verify chunks are NOT displayed
        assert "Retrieved chunks:" not in captured.out

    @patch("scripts.chat.build_chain")
    @patch("scripts.chat.check_ollama_available")
    @patch("builtins.input")
    def test_chunk_ids_not_in_response_output(
        self,
        mock_input: MagicMock,
        mock_ollama: MagicMock,
        mock_build_chain: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that chunk IDs never appear in response text (only in --show-chunks mode)."""
        from scripts.chat import run_interactive_chat

        # Setup mocks with response that should NOT contain chunk IDs
        mock_ollama.return_value = None
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = create_mock_response()
        mock_build_chain.return_value = mock_chain
        mock_input.side_effect = [TEST_QUESTION, "quit"]

        # Run with show_chunks=False
        run_interactive_chat(
            db_path=Path(TEST_DB_PATH),
            author=TEST_AUTHOR,
            show_chunks=False,
            verbose=False,
        )

        # Capture output
        captured = capsys.readouterr()

        # Verify response text is displayed
        assert TEST_RESPONSE_TEXT in captured.out

        # Verify chunk IDs from metadata are NOT in the main output
        # (they should only appear in the --show-chunks section)
        lines_before_chunks_section = captured.out.split("Sources:")[0]
        for chunk_id in TEST_CHUNK_IDS:
            assert chunk_id not in lines_before_chunks_section

    @patch("scripts.chat.build_chain")
    @patch("scripts.chat.check_ollama_available")
    @patch("builtins.input")
    def test_sources_always_displayed(
        self,
        mock_input: MagicMock,
        mock_ollama: MagicMock,
        mock_build_chain: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that sources footer is always displayed."""
        from scripts.chat import run_interactive_chat

        # Setup mocks
        mock_ollama.return_value = None
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = create_mock_response()
        mock_build_chain.return_value = mock_chain
        mock_input.side_effect = [TEST_QUESTION, "quit"]

        # Run
        run_interactive_chat(
            db_path=Path(TEST_DB_PATH),
            author=TEST_AUTHOR,
            show_chunks=False,
            verbose=False,
        )

        # Capture output
        captured = capsys.readouterr()

        # Verify sources are displayed and deduplicated
        assert "Sources:" in captured.out
        assert " Esquisse d'un tableau historique, Page 12" in captured.out
        assert " Esquisse d'un tableau historique, Page 9" in captured.out

    @patch("scripts.chat.build_chain")
    @patch("scripts.chat.check_ollama_available")
    @patch("builtins.input")
    def test_quit_command_exits(
        self,
        mock_input: MagicMock,
        mock_ollama: MagicMock,
        mock_build_chain: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that 'quit' command exits gracefully and displays exit message."""
        from scripts.chat import run_interactive_chat

        # Setup mocks
        mock_ollama.return_value = None
        mock_chain = MagicMock()
        mock_build_chain.return_value = mock_chain
        mock_input.side_effect = ["quit"]

        # Run
        run_interactive_chat(
            db_path=Path(TEST_DB_PATH),
            author=TEST_AUTHOR,
            show_chunks=False,
            verbose=False,
        )

        # Verify chain was never invoked
        mock_chain.invoke.assert_not_called()

        # Verify exit message appears in output
        captured = capsys.readouterr()
        assert "Au revoir - Condorcet" in captured.out

    @patch("scripts.chat.build_chain")
    @patch("scripts.chat.check_ollama_available")
    @patch("builtins.input")
    def test_keyboard_interrupt_exits(
        self,
        mock_input: MagicMock,
        mock_ollama: MagicMock,
        mock_build_chain: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that Ctrl+C (KeyboardInterrupt) exits gracefully and displays exit message."""
        from scripts.chat import run_interactive_chat

        # Setup mocks
        mock_ollama.return_value = None
        mock_chain = MagicMock()
        mock_build_chain.return_value = mock_chain
        mock_input.side_effect = KeyboardInterrupt()

        # Run - should not raise
        run_interactive_chat(
            db_path=Path(TEST_DB_PATH),
            author=TEST_AUTHOR,
            show_chunks=False,
            verbose=False,
        )

        # Verify chain was never invoked
        mock_chain.invoke.assert_not_called()

        # Verify exit message appears in output
        captured = capsys.readouterr()
        assert "Au revoir - Condorcet" in captured.out

    @patch("scripts.chat.build_chain")
    @patch("scripts.chat.check_ollama_available")
    @patch("builtins.input")
    def test_eof_error_exits(
        self,
        mock_input: MagicMock,
        mock_ollama: MagicMock,
        mock_build_chain: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that EOFError exits gracefully and displays exit message."""
        from scripts.chat import run_interactive_chat

        # Setup mocks
        mock_ollama.return_value = None
        mock_chain = MagicMock()
        mock_build_chain.return_value = mock_chain
        mock_input.side_effect = EOFError()

        # Run - should not raise
        run_interactive_chat(
            db_path=Path(TEST_DB_PATH),
            author=TEST_AUTHOR,
            show_chunks=False,
            verbose=False,
        )

        # Verify chain was never invoked
        mock_chain.invoke.assert_not_called()

        # Verify exit message appears in output
        captured = capsys.readouterr()
        assert "Au revoir - Condorcet" in captured.out

    @patch("scripts.chat.build_chain")
    @patch("scripts.chat.check_ollama_available")
    @patch("builtins.input")
    def test_empty_question_skipped(
        self,
        mock_input: MagicMock,
        mock_ollama: MagicMock,
        mock_build_chain: MagicMock,
    ) -> None:
        """Test that empty questions are skipped."""
        from scripts.chat import run_interactive_chat

        # Setup mocks
        mock_ollama.return_value = None
        mock_chain = MagicMock()
        mock_build_chain.return_value = mock_chain
        mock_input.side_effect = ["", "   ", "quit"]

        # Run
        run_interactive_chat(
            db_path=Path(TEST_DB_PATH),
            author=TEST_AUTHOR,
            show_chunks=False,
            verbose=False,
        )

        # Verify chain was never invoked for empty questions
        mock_chain.invoke.assert_not_called()

    @patch("scripts.chat.build_chain")
    @patch("scripts.chat.check_ollama_available")
    @patch("builtins.input")
    def test_multiple_questions(
        self,
        mock_input: MagicMock,
        mock_ollama: MagicMock,
        mock_build_chain: MagicMock,
    ) -> None:
        """Test multiple questions in sequence."""
        from scripts.chat import run_interactive_chat

        # Setup mocks
        mock_ollama.return_value = None
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = create_mock_response()
        mock_build_chain.return_value = mock_chain

        questions = ["Question 1?", "Question 2?", "Question 3?", "quit"]
        mock_input.side_effect = questions

        # Run
        run_interactive_chat(
            db_path=Path(TEST_DB_PATH),
            author=TEST_AUTHOR,
            show_chunks=False,
            verbose=False,
        )

        # Verify chain invoked 3 times (not for quit)
        assert mock_chain.invoke.call_count == 3
        mock_chain.invoke.assert_has_calls(
            [call("Question 1?"), call("Question 2?"), call("Question 3?")]
        )

    @patch("scripts.chat.build_chain")
    @patch("scripts.chat.check_ollama_available")
    @patch("builtins.input")
    def test_ollama_check_called(
        self,
        mock_input: MagicMock,
        mock_ollama: MagicMock,
        mock_build_chain: MagicMock,
    ) -> None:
        """Test that Ollama availability is checked."""
        from scripts.chat import run_interactive_chat

        # Setup mocks
        mock_ollama.return_value = None
        mock_chain = MagicMock()
        mock_build_chain.return_value = mock_chain
        mock_input.side_effect = ["quit"]

        # Run
        run_interactive_chat(
            db_path=Path(TEST_DB_PATH),
            author=TEST_AUTHOR,
            show_chunks=False,
            verbose=False,
        )

        # Verify Ollama check was called
        mock_ollama.assert_called_once()

    @patch("scripts.chat.check_ollama_available")
    def test_ollama_not_available_raises(
        self,
        mock_ollama: MagicMock,
    ) -> None:
        """Test that RuntimeError is raised if Ollama not available."""
        from scripts.chat import run_interactive_chat

        # Setup mock to raise
        mock_ollama.side_effect = RuntimeError("Ollama is not running")

        # Run - should raise
        with pytest.raises(RuntimeError, match="Ollama is not running"):
            run_interactive_chat(
                db_path=Path(TEST_DB_PATH),
                author=TEST_AUTHOR,
                show_chunks=False,
                verbose=False,
            )

    @patch("scripts.chat.build_chain")
    @patch("scripts.chat.check_ollama_available")
    @patch("builtins.input")
    def test_invalid_author_raises(
        self,
        mock_input: MagicMock,
        mock_ollama: MagicMock,
        mock_build_chain: MagicMock,
    ) -> None:
        """Test that ValueError is raised for invalid author."""
        from scripts.chat import run_interactive_chat

        # Setup mocks
        mock_ollama.return_value = None
        mock_build_chain.side_effect = ValueError("Unknown author")
        mock_input.side_effect = ["quit"]

        # Run - should raise
        with pytest.raises(ValueError, match="Unknown author"):
            run_interactive_chat(
                db_path=Path(TEST_DB_PATH),
                author="invalid_author",
                show_chunks=False,
                verbose=False,
            )

    @patch("scripts.chat.build_chain")
    @patch("scripts.chat.check_ollama_available")
    @patch("builtins.input")
    def test_chain_error_continues_loop(
        self,
        mock_input: MagicMock,
        mock_ollama: MagicMock,
        mock_build_chain: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that errors during chain invocation don't exit loop."""
        from scripts.chat import run_interactive_chat

        # Setup mocks
        mock_ollama.return_value = None
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = [
            Exception("Chain error"),
            create_mock_response(),
        ]
        mock_build_chain.return_value = mock_chain
        mock_input.side_effect = ["Question 1?", "Question 2?", "quit"]

        # Run - should not raise
        run_interactive_chat(
            db_path=Path(TEST_DB_PATH),
            author=TEST_AUTHOR,
            show_chunks=False,
            verbose=False,
        )

        # Verify error message logged
        assert "Error: Chain error" in caplog.text

        # Verify chain invoked twice (error + success)
        assert mock_chain.invoke.call_count == 2

    @patch("scripts.chat.build_chain")
    @patch("scripts.chat.check_ollama_available")
    @patch("builtins.input")
    def test_verbose_flag_enables_debug_logging(
        self,
        mock_input: MagicMock,
        mock_ollama: MagicMock,
        mock_build_chain: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that verbose flag enables debug logging."""
        from scripts.chat import run_interactive_chat

        # Setup mocks
        mock_ollama.return_value = None
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = create_mock_response()
        mock_build_chain.return_value = mock_chain
        mock_input.side_effect = [TEST_QUESTION, "quit"]

        # Run with verbose=True
        run_interactive_chat(
            db_path=Path(TEST_DB_PATH),
            author=TEST_AUTHOR,
            show_chunks=False,
            verbose=True,
        )

        # Verify debug messages logged
        assert "Verbose logging enabled" in caplog.text
        assert "Invoking chain with question:" in caplog.text


class TestMain:
    """Test main() function."""

    @patch("scripts.chat.run_interactive_chat")
    def test_default_arguments(
        self,
        mock_run: MagicMock,
    ) -> None:
        """Test main() with default arguments."""
        with patch("sys.argv", ["chat.py"]):
            from scripts.chat import main

            main()

        # Verify run_interactive_chat called with defaults
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[1]["db_path"] == Path(DEFAULT_DB_PATH)
        assert call_args[1]["author"] == DEFAULT_AUTHOR
        assert call_args[1]["show_chunks"] is False
        assert call_args[1]["verbose"] is False

    @patch("scripts.chat.run_interactive_chat")
    def test_custom_db_path(
        self,
        mock_run: MagicMock,
    ) -> None:
        """Test main() with custom --db path."""
        custom_db = "custom/db"
        with patch("sys.argv", ["chat.py", "--db", custom_db]):
            from scripts.chat import main

            main()

        # Verify custom db path used
        call_args = mock_run.call_args
        assert call_args[1]["db_path"] == Path(custom_db)

    @patch("scripts.chat.run_interactive_chat")
    def test_custom_author(
        self,
        mock_run: MagicMock,
    ) -> None:
        """Test main() with custom --author."""
        custom_author = "gouges"
        with patch("sys.argv", ["chat.py", "--author", custom_author]):
            from scripts.chat import main

            main()

        # Verify custom author used
        call_args = mock_run.call_args
        assert call_args[1]["author"] == custom_author

    @patch("scripts.chat.run_interactive_chat")
    def test_show_chunks_flag(
        self,
        mock_run: MagicMock,
    ) -> None:
        """Test main() with --show-chunks flag."""
        with patch("sys.argv", ["chat.py", "--show-chunks"]):
            from scripts.chat import main

            main()

        # Verify show_chunks=True
        call_args = mock_run.call_args
        assert call_args[1]["show_chunks"] is True

    @patch("scripts.chat.run_interactive_chat")
    def test_verbose_flag(
        self,
        mock_run: MagicMock,
    ) -> None:
        """Test main() with --verbose flag."""
        with patch("sys.argv", ["chat.py", "--verbose"]):
            from scripts.chat import main

            main()

        # Verify verbose=True
        call_args = mock_run.call_args
        assert call_args[1]["verbose"] is True

    @patch("scripts.chat.run_interactive_chat")
    def test_all_flags_combined(
        self,
        mock_run: MagicMock,
    ) -> None:
        """Test main() with all flags combined."""
        custom_db = "custom/db"
        custom_author = "gouges"
        with patch(
            "sys.argv",
            [
                "chat.py",
                "--db",
                custom_db,
                "--author",
                custom_author,
                "--show-chunks",
                "--verbose",
            ],
        ):
            from scripts.chat import main

            main()

        # Verify all arguments passed correctly
        call_args = mock_run.call_args
        assert call_args[1]["db_path"] == Path(custom_db)
        assert call_args[1]["author"] == custom_author
        assert call_args[1]["show_chunks"] is True
        assert call_args[1]["verbose"] is True

    @patch("scripts.chat.run_interactive_chat")
    def test_exception_exits_with_code_1(
        self,
        mock_run: MagicMock,
    ) -> None:
        """Test that exceptions cause exit with code 1."""
        mock_run.side_effect = RuntimeError("Ollama not running")

        with patch("sys.argv", ["chat.py"]):
            from scripts.chat import main

            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1
