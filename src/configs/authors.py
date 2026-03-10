"""Author configuration registry for prompt templates and default languages."""

from typing import Callable

from langchain_core.prompts import ChatPromptTemplate

from src.prompts.voltaire import build_voltaire_prompt

# Type alias for author configuration: (prompt_factory, default_language_code)
AuthorConfig = tuple[Callable[[], ChatPromptTemplate], str]

# Registry mapping author key -> (prompt_factory, default_language)
AUTHOR_CONFIGS: dict[str, AuthorConfig] = {
    "voltaire": (build_voltaire_prompt, "fr"),
}

# Default author for chat chain and CLI scripts
DEFAULT_AUTHOR = "voltaire"
