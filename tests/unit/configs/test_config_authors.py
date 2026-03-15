"""Unit tests for authors config module."""

import pytest
from langchain_core.prompts import ChatPromptTemplate

from src.configs.authors import AUTHOR_CONFIGS, AuthorConfig, DEFAULT_AUTHOR


def test_author_configs_registry() -> None:
    """Test that AUTHOR_CONFIGS is a non-empty dict."""
    assert isinstance(AUTHOR_CONFIGS, dict)
    assert len(AUTHOR_CONFIGS) > 0


def test_author_config_structure() -> None:
    """Test that each author config has the correct structure."""
    for author, config in AUTHOR_CONFIGS.items():
        # Validate config is an AuthorConfig dataclass
        assert isinstance(config, AuthorConfig), f"Config for {author} must be an AuthorConfig"

        # Validate types
        assert callable(config.prompt_factory), f"Prompt factory for {author} must be callable"
        assert isinstance(config.exit_message, str), f"Exit message for {author} must be a string"

        # Validate exit message is non-empty
        assert len(config.exit_message) > 0, f"Exit message for {author} must be non-empty"


@pytest.mark.parametrize("author", list(AUTHOR_CONFIGS.keys()))
def test_prompt_factory_returns_valid_template(author: str) -> None:
    """Test that each author's prompt factory returns ChatPromptTemplate."""
    config = AUTHOR_CONFIGS[author]
    prompt = config.prompt_factory()
    assert isinstance(
        prompt, ChatPromptTemplate
    ), f"Prompt factory for {author} must return ChatPromptTemplate"


def test_default_author() -> None:
    """Test that DEFAULT_AUTHOR is voltaire and registered."""
    assert DEFAULT_AUTHOR == "voltaire"
    assert DEFAULT_AUTHOR in AUTHOR_CONFIGS
