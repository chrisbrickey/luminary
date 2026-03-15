"""Interactive CLI chat interface for querying philosophers."""

import argparse
import logging
import os
import sys
from pathlib import Path

# Disable ChromaDB telemetry before any imports that use ChromaDB
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from src.chains.chat_chain import build_chain
from src.configs.authors import AUTHOR_CONFIGS, DEFAULT_AUTHOR
from src.configs.common import DEFAULT_DB_PATH
from src.schemas import ChatResponse
from src.utils.ollama_health import check_ollama_available

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

# Suppress HTTP request logging from httpx, urllib3, and other noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)


def deduplicate_sources(response: ChatResponse) -> list[str]:
    """Deduplicate source titles while preserving order.

    Args:
        response: ChatResponse with retrieved_source_titles

    Returns:
        List of deduplicated source titles in order of first appearance
    """
    seen = set()
    deduplicated = []
    for title in response.retrieved_source_titles:
        if title not in seen:
            seen.add(title)
            deduplicated.append(title)
    return deduplicated


def format_sources_footer(response: ChatResponse) -> str:
    """Format the Sources: footer with deduplicated titles.

    Args:
        response: ChatResponse with retrieved_source_titles

    Returns:
        Formatted sources string
    """
    sources = deduplicate_sources(response)
    if not sources:
        return "\nSources: none"
    return "\nSources:\n" + "\n".join(f"  - {title}" for title in sources)


def format_chunks_output(response: ChatResponse) -> str:
    """Format retrieved chunks with IDs and contexts.

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
    db_path: Path,
    author: str,
    show_chunks: bool,
    verbose: bool,
) -> None:
    """Run interactive chat loop.

    Args:
        db_path: Path to ChromaDB directory
        author: Author key (e.g., "voltaire")
        show_chunks: Whether to display retrieved chunks
        verbose: Whether to enable verbose logging

    Raises:
        ValueError: If author is not registered
        RuntimeError: If Ollama is not available
    """
    # Set verbose logging if requested
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Check Ollama availability
    logger.debug("Checking Ollama availability...")
    check_ollama_available()

    # Build the chain
    logger.debug(f"Loading chain for author: {author}")
    logger.debug(f"Using database: {db_path}")
    chain = build_chain(persist_dir=str(db_path), author=author)

    # Get author's exit message
    exit_msg = AUTHOR_CONFIGS[author].exit_message

    # Interactive loop
    print("\nWelcome to Luminary!")
    print(f"\nYou are now chatting with {author.capitalize()}.")
    print(f"Type 'quit' or press Ctrl+C to exit.\n")

    while True:
        try:
            question = input("You/Vous: ").strip()

            # Exit on quit command
            if question.lower() == "quit":
                print(f"\n{exit_msg}")
                break

            # Skip empty questions
            if not question:
                continue

            # Show loading message
            print("\n⏳ Reflecting… (response time varies with retrieval corpus size and API latency)")

            # Invoke chain
            if verbose:
                logger.debug(f"Invoking chain with question: {question}")

            response: ChatResponse = chain.invoke(question)

            # Print response (overwrite loading message line)
            print(f"\r{' ' * 80}\r", end="")  # Clear the loading message
            print(f"\n{author.capitalize()}:\n{response.text}")

            # Print chunks if requested
            if show_chunks:
                print(format_chunks_output(response))

            # Always print sources footer
            print(format_sources_footer(response))
            print()  # Blank line for readability

        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{exit_msg}")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            if verbose:
                logger.exception("Full traceback:")


def main() -> None:
    """Parse arguments and run interactive chat."""
    parser = argparse.ArgumentParser(
        description="Interactive CLI chat with Enlightenment philosophers"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to ChromaDB directory (default: {DEFAULT_DB_PATH})",
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
            db_path=args.db,
            author=args.author,
            show_chunks=args.show_chunks,
            verbose=args.verbose,
        )
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
