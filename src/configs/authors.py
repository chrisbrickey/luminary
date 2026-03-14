"""Author configuration registry for prompt templates, default languages, and farewell messages."""

from typing import Callable

from langchain_core.prompts import ChatPromptTemplate

from src.prompts.voltaire import build_voltaire_prompt

# Type alias for author configuration: (prompt_factory, default_language_code, goodbye_message)
# - prompt_factory: builds the LLM prompt template
# - default_language_code: fallback for UI language detection (NOT sent to LLM)
# - goodbye_message: author-specific farewell in their native language/style
AuthorConfig = tuple[Callable[[], ChatPromptTemplate], str, str]

# Registry mapping author key -> (prompt_factory, default_language, goodbye_message)
AUTHOR_CONFIGS: dict[str, AuthorConfig] = {
    "voltaire": (build_voltaire_prompt, "fr", "Au revoir, et cultivons notre jardin."),
}

# Default author for chat chain and CLI scripts
DEFAULT_AUTHOR = "voltaire"
