"""Interactive CLI chat interface for querying philosophers."""

import argparse
import logging
import os
import sys

# Disable ChromaDB telemetry before any imports that use ChromaDB
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from src.chains.chat_chain import build_chain
from src.configs.authors import AUTHOR_CONFIGS, DEFAULT_AUTHOR
from src.configs.common import DEFAULT_RESPONSE_LANGUAGE
from src.i18n import get_message
from src.i18n.keys import (
    CHAT_CHATTING_WITH,
    CHAT_EXIT_INSTRUCTIONS,
    CHAT_INPUT_PROMPT,
    ERROR_GENERIC,
    STATUS_REFLECTING,
)
from src.schemas import ChatResponse
from src.utils.formatting import format_sources
from src.utils.language import detect_language
from src.utils.logging import setup_cli_logging
from src.utils.ollama_health import check_ollama_available


def format_chunks_output(response: ChatResponse) -> str:
    """Format retrieved chunks with IDs and contexts when debugging.

    NB: This supports a unique feature for the CLI.
    This is called when the script is executed with --show-chunks flag.

    Args:
        response: ChatResponse with retrieved contexts and IDs

    Returns:
        Formatted chunks string
    """
    if not response.retrieved_passage_ids:
        return "\nRetrieved chunks: none"

    output = ["\nRetrieved chunks:"]
    for chunk_id, context in zip(
        response.retrieved_passage_ids, response.retrieved_contexts
    ):
        output.append(f"\n[{chunk_id}]")
        output.append(context)
    return "\n".join(output)


def run_interactive_chat(
    author: str,
    show_chunks: bool,
    verbose: bool,
) -> None:
    """Run interactive chat loop.

    Args:
        author: Author key (e.g., "voltaire")
        show_chunks: Whether to display retrieved chunks
        verbose: Whether to enable verbose logging

    Raises:
        ValueError: If author is not registered
        RuntimeError: If Ollama is not available
    """
    # Setup logging
    logger = setup_cli_logging(verbose=verbose)

    if verbose:
        logger.debug("Verbose logging enabled")

    # Suppress HTTP request logging from noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)

    # Check Ollama availability
    logger.debug("Checking Ollama availability...")
    check_ollama_available()

    # Build the chain
    logger.debug(f"Loading chain for author: {author}")
    chain = build_chain(author=author)

    # Get author's exit message
    exit_msg = AUTHOR_CONFIGS[author].exit_message

    # Interactive loop
    print("\n💡 Welcome to Luminary!")
    print(
        f"\n{get_message(CHAT_CHATTING_WITH, DEFAULT_RESPONSE_LANGUAGE, author=author.capitalize())}"
    )
    print(f"{get_message(CHAT_EXIT_INSTRUCTIONS, DEFAULT_RESPONSE_LANGUAGE)}\n")

    while True:
        try:
            question = input(
                get_message(CHAT_INPUT_PROMPT, DEFAULT_RESPONSE_LANGUAGE)
            ).strip()

            # Exit on quit command
            if question.lower() == "quit":
                print(f"\n{exit_msg}")
                break

            # Skip empty questions
            if not question:
                continue

            # 1. Show loading icon immediately
            print("\n⏳ ", end="", flush=True)

            # 2. Detect language
            detected_lang = detect_language(question)

            # 3. Show localized loading message
            print(get_message(STATUS_REFLECTING, detected_lang))

            # 4. Invoke chain with detected language
            if verbose:
                logger.debug(f"Invoking chain with question: {question}")
                logger.debug(f"Detected language: {detected_lang}")

            response: ChatResponse = chain.invoke(
                question, language=detected_lang
            )

            # Clear loading message and print out response
            print(f"\r{' ' * 80}\r", end="")
            print(f"\n{author.capitalize()}:\n{response.text}")

            # Print chunks if requested
            if show_chunks:
                print(format_chunks_output(response))

            # Print sources footer in detected language (if localization is supported)
            print(format_sources(response, detected_lang))
            print()

        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{exit_msg}")
            break
        except Exception as e:
            error_msg = get_message(
                ERROR_GENERIC, DEFAULT_RESPONSE_LANGUAGE, error=str(e)
            )
            logger.error(error_msg)
            if verbose:
                logger.exception("Full traceback:")


def main() -> None:
    """Parse arguments and run interactive chat."""
    parser = argparse.ArgumentParser(
        description="Interactive CLI chat with Enlightenment philosophers"
    )
    parser.add_argument(
        "--author",
        type=str,
        default=DEFAULT_AUTHOR,
        help=f"Author to query (default: {DEFAULT_AUTHOR})",
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Display retrieved chunks with IDs and contexts",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    try:
        run_interactive_chat(
            author=args.author,
            show_chunks=args.show_chunks,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
