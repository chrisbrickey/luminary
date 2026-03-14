"""Unit tests for authors config module."""

import pytest
from langchain_core.prompts import ChatPromptTemplate

from src.configs.authors import AUTHOR_CONFIGS, DEFAULT_AUTHOR, AuthorConfig

# Test constants
_EXPECTED_CONFIG_TUPLE_LENGTH = 3
_ISO_639_1_LENGTH = 2


def test_author_configs_registry() -> None:
    """Test that AUTHOR_CONFIGS is a non-empty dict."""
    assert isinstance(AUTHOR_CONFIGS, dict)
    assert len(AUTHOR_CONFIGS) > 0


def test_author_config_structure() -> None:
    """Test that each author config has the correct structure and valid language codes."""
    for author, config in AUTHOR_CONFIGS.items():
        # Validate config is a 3-element tuple
        assert isinstance(config, tuple), f"Config for {author} must be a tuple"
        assert (
            len(config) == _EXPECTED_CONFIG_TUPLE_LENGTH
        ), f"Config for {author} must have {_EXPECTED_CONFIG_TUPLE_LENGTH} elements"

        prompt_factory, language, goodbye_msg = config

        # Validate types
        assert callable(prompt_factory), f"Prompt factory for {author} must be callable"
        assert isinstance(language, str), f"Language for {author} must be a string"
        assert isinstance(goodbye_msg, str), f"Goodbye message for {author} must be a string"

        # Validate language code is ISO 639-1 format
        assert (
            len(language) == _ISO_639_1_LENGTH
        ), f"Language code for {author} must be {_ISO_639_1_LENGTH} characters"
        assert language.islower(), f"Language code for {author} must be lowercase"
        assert language.isalpha(), f"Language code for {author} must be alphabetic"

        # Validate goodbye message is non-empty
        assert len(goodbye_msg) > 0, f"Goodbye message for {author} must not be empty"


@pytest.mark.parametrize("author", list(AUTHOR_CONFIGS.keys()))
def test_prompt_factory_returns_valid_template(author: str) -> None:
    """Test that each author's prompt factory returns ChatPromptTemplate."""
    prompt_factory, _, _ = AUTHOR_CONFIGS[author]
    prompt = prompt_factory()
    assert isinstance(
        prompt, ChatPromptTemplate
    ), f"Prompt factory for {author} must return ChatPromptTemplate"


def test_default_author() -> None:
    """Test that DEFAULT_AUTHOR is voltaire and registered."""
    assert DEFAULT_AUTHOR == "voltaire"
    assert DEFAULT_AUTHOR in AUTHOR_CONFIGS
