"""Unit tests for Voltaire prompt."""

import pytest
from langchain_core.prompts import ChatPromptTemplate

from src.configs.common import ENGLISH_ISO_CODE
from src.prompts.voltaire import build_voltaire_prompt

# Test constants
SAMPLE_CONTEXT = "Sample passage from works"
SAMPLE_QUESTION = "What is your view?"
SAMPLE_LANGUAGE = ENGLISH_ISO_CODE


@pytest.fixture
def prompt() -> ChatPromptTemplate:
    """Provide Voltaire prompt for testing."""
    return build_voltaire_prompt()


@pytest.fixture
def system_template(prompt: ChatPromptTemplate) -> str:
    """Extract system message template from prompt."""
    return prompt.messages[0].prompt.template


class TestBuildVoltairePrompt:
    """Tests for build_voltaire_prompt function."""

    class TestStructure:
        """Tests for prompt structure and templating."""

        def test_returns_chat_prompt_template(self, prompt: ChatPromptTemplate) -> None:
            """Should return a ChatPromptTemplate instance."""
            assert isinstance(prompt, ChatPromptTemplate)

        def test_has_required_input_variables(self, prompt: ChatPromptTemplate) -> None:
            """Should require context, question, and language variables."""
            assert set(prompt.input_variables) == {"context", "question", "language"}

        def test_formats_with_all_variables(self, prompt: ChatPromptTemplate) -> None:
            """Should format successfully with all required variables."""
            formatted = prompt.format_messages(
                context=SAMPLE_CONTEXT,
                question=SAMPLE_QUESTION,
                language=SAMPLE_LANGUAGE,
            )
            assert len(formatted) == 2
            assert SAMPLE_CONTEXT in formatted[0].content
            assert SAMPLE_LANGUAGE in formatted[0].content
            assert formatted[1].content == SAMPLE_QUESTION

    class TestCitationRules:
        """Tests for citation instructions."""

        def test_requires_source_citations(self, system_template: str) -> None:
            """Should instruct to cite sources in [source: ...] format."""
            template_lower = system_template.lower()
            # Check citation requirement
            assert any(keyword in template_lower for keyword in ["mandatory", "always cite", "must cite"])
            # Check format specification
            assert "[source:" in system_template

        def test_instructs_citation_placement(self, system_template: str) -> None:
            """Should instruct citations after sentences, never at beginning."""
            template_lower = system_template.lower()
            # After sentence instruction
            assert "after" in template_lower
            # Never at beginning instruction
            assert "never" in template_lower and "beginning" in template_lower

        def test_does_not_mention_chunk_ids(self, system_template: str) -> None:
            """Should NOT instruct to include chunk_id in citations."""
            assert "chunk_id" not in system_template.lower()

    class TestContentGuidance:
        """Tests for content fidelity and style rules."""

        def test_requires_textual_fidelity(self, system_template: str) -> None:
            """Should instruct to base responses only on provided passages."""
            template_lower = system_template.lower()
            assert any(keyword in template_lower for keyword in ["exclusively", "only", "nothing"])

        def test_instructs_language_and_concision(self, system_template: str) -> None:
            """Should instruct response language and concise format."""
            assert "{language}" in system_template
            template_lower = system_template.lower()
            assert any(keyword in template_lower for keyword in ["concis", "short", "paragraph"])

        def test_instructs_natural_integration(self, system_template: str) -> None:
            """Should instruct to paraphrase/integrate sources naturally."""
            template_lower = system_template.lower()
            assert any(keyword in template_lower for keyword in ["paraphrase", "integrate", "natural"])

    class TestPersona:
        """Tests for Voltaire persona."""

        def test_establishes_persona_and_context(self, system_template: str) -> None:
            """Should establish Voltaire persona and Enlightenment context."""
            template_lower = system_template.lower()
            assert "voltaire" in template_lower
            assert "enlightenment" in template_lower or "lumières" in template_lower
            assert "wit" and "irony" and "clarity" in template_lower

        def test_provides_examples(self, system_template: str) -> None:
            """Should provide correct and incorrect citation examples."""
            # Visual markers for examples
            assert "✓" and "CORRECT" in system_template
            assert "✗" and "INCORRECT" in system_template
