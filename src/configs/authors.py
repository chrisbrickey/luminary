"""Author configuration registry for prompt templates and metadata."""

from dataclasses import dataclass
from typing import Callable

from langchain_core.prompts import ChatPromptTemplate

from src.prompts.voltaire import build_voltaire_prompt

@dataclass(frozen=True)
class AuthorConfig:
    """Configuration for an author.

    Attributes:
        prompt_factory: Function that builds the author's ChatPromptTemplate
        exit_message: Farewell message shown when user exits the chat
    """

    prompt_factory: Callable[[], ChatPromptTemplate]
    exit_message: str


# Registry mapping author key -> AuthorConfig
AUTHOR_CONFIGS: dict[str, AuthorConfig] = {
    "voltaire": AuthorConfig(
        prompt_factory=build_voltaire_prompt,
        exit_message="Je vous embrasse - V",
    ),
}

# Default author for chat chain and CLI scripts
DEFAULT_AUTHOR = "voltaire"
