"""CLI script for generating golden datasets using LLM-based independent evaluation.

This script uses a local LLM to independently judge which chunks are relevant,
what sources to expect, etc. This provides independent quality validation -
not just regression testing.

Why No Test Coverage?
The core logic is thoroughly tested in src/eval/golden/dataset_generation.py.
This script is just an argparse wrapper + user interaction (version prompts).
This is a one-off CLI script for dataset generation, not production code.

Usage:
    # Setup (one-time): Create .env file with your API key
    cp .env.example .env
    # Edit .env and add your ANTHROPIC_API_KEY

    # Generate using Anthropic (recommended - better JSON generation)
    uv run python src/eval/golden/scripts/generate_golden_dataset.py --config src/eval/golden/configs/voltaire_examples.json --provider anthropic

    # Alternatively, set API key via environment variable
    export ANTHROPIC_API_KEY='your-key-here'
    uv run python src/eval/golden/scripts/generate_golden_dataset.py --config src/eval/golden/configs/voltaire_examples.json --provider anthropic

    # Generate using Ollama (local, may struggle with complex JSON)
    uv run python src/eval/golden/scripts/generate_golden_dataset.py --config src/eval/golden/configs/voltaire_examples.json --provider ollama --model llama3

    # Use different model
    uv run python src/eval/golden/scripts/generate_golden_dataset.py --config src/eval/golden/configs/voltaire_examples.json --provider anthropic --model claude-opus-4-20250514

    # Specify version directly (non-interactive)
    uv run python src/eval/golden/scripts/generate_golden_dataset.py --config src/eval/golden/configs/voltaire_examples.json --provider anthropic --version 1.2

    # Enable verbose logging
    uv run python src/eval/golden/scripts/generate_golden_dataset.py --config src/eval/golden/configs/voltaire_examples.json --provider anthropic --verbose

Design principles:
- LLM-based independent evaluation (not circular)
- Temperature=0 for reproducible generations
- User controls versioning (decimal vs whole number increments)
- Auto-discovery of latest dataset for version bumping
- Multi-language by design (processes FR and EN examples together)
"""

import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from src.configs.eval import DEFAULT_GOLDEN_DATASET_PATH
from src.eval.golden.dataset_generation import generate_golden_example_with_llm
from src.eval.utils import discover_latest_golden_dataset, load_golden_dataset
from src.schemas.eval import GoldenDataset, GoldenExample
from src.utils.cli_helpers import check_ollama_or_exit
from src.utils.logging import setup_cli_logging
from src.vectorstores.retriever import build_retriever

logger = logging.getLogger(__name__)


def _parser_epilogue() -> str:
    epilogue = """
    Setup (first time):
      cp .env.example .env
      # Edit .env and add your ANTHROPIC_API_KEY

    Examples:
      # Generate with Anthropic (recommended)
      %(prog)s --config src/eval/golden/configs/voltaire_examples.json --provider anthropic

      # Generate with Ollama (local, may struggle with complex JSON)
      %(prog)s --config src/eval/golden/configs/voltaire_examples.json --provider ollama --model llama3

      # Specify version directly (non-interactive)
      %(prog)s --config src/eval/golden/configs/voltaire_examples.json --provider anthropic --version 1.2

      # Use different model
      %(prog)s --config src/eval/golden/configs/voltaire_examples.json --provider anthropic --model claude-opus-4-20250514

      # Save to custom directory
      %(prog)s --config src/eval/golden/configs/voltaire_examples.json --provider anthropic --output custom/output

      # Enable debug logging
      %(prog)s --config src/eval/golden/configs/voltaire_examples.json --provider anthropic --verbose
      """
    return epilogue


def parse_version(version_str: str) -> tuple[int, int]:
    """Parse semantic version string into (major, minor) tuple.

    Args:
        version_str: Version string like "1.0" or "2.3"

    Returns:
        Tuple of (major, minor) as integers

    Raises:
        ValueError: If version format is invalid
    """
    parts = version_str.split(".")
    if len(parts) != 2:
        raise ValueError(f"Invalid version format: {version_str}. Expected 'X.Y'")

    try:
        major = int(parts[0])
        minor = int(parts[1])
        return (major, minor)
    except ValueError as e:
        raise ValueError(f"Invalid version format: {version_str}. Parts must be integers.") from e


def initialize_llm(provider: str, model: str | None) -> BaseChatModel:
    """Initialize LLM based on provider.

    Args:
        provider: Either "ollama" or "anthropic"
        model: Model name (defaults to "mistral" for ollama, "claude-sonnet-4-5-20250929" for anthropic)

    Returns:
        Initialized LLM instance with temperature=0

    Raises:
        ValueError: If provider is invalid or required env vars are missing
    """
    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Set it with: export ANTHROPIC_API_KEY='your-key-here'"
            )

        model_name = model or "claude-sonnet-4-5-20250929"
        logger.info(f"Initializing Anthropic LLM: {model_name} (temperature=0)")
        return ChatAnthropic(model=model_name, temperature=0)  # type: ignore[call-arg]

    elif provider == "ollama":
        model_name = model or "mistral"
        logger.info(f"Initializing Ollama LLM: {model_name} (temperature=0, format=json)")
        return ChatOllama(model=model_name, temperature=0, format="json")

    else:
        raise ValueError(f"Invalid provider: {provider}. Must be 'ollama' or 'anthropic'")


def prompt_version_increment(current_version: str) -> str:
    """Prompt user to choose version increment type.

    Args:
        current_version: Current version string (e.g., "1.0")

    Returns:
        New version string based on user choice
    """
    major, minor = parse_version(current_version)

    decimal_version = f"{major}.{minor + 1}"
    whole_version = f"{major + 1}.0"

    print(f"\nCurrent version: {current_version}")
    print(f"  [1] Decimal increment → {decimal_version} (minor changes: new examples, tweaks)")
    print(f"  [2] Whole number increment → {whole_version} (major changes: new metrics, schema updates)")

    while True:
        choice = input("Select version increment [1/2]: ").strip()
        if choice == "1":
            return decimal_version
        elif choice == "2":
            return whole_version
        else:
            print("Invalid choice. Please enter 1 or 2.")


def main() -> int:
    """Main entry point for golden dataset generation CLI.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Generate golden dataset using LLM-based independent evaluation",
        epilog=_parser_epilogue(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to examples config JSON (required)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_GOLDEN_DATASET_PATH,
        help=f"Output directory for golden dataset (default: {DEFAULT_GOLDEN_DATASET_PATH})",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["ollama", "anthropic"],
        default="anthropic",
        help="LLM provider to use (default: anthropic, recommended for better JSON generation)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (default: mistral for ollama, claude-sonnet-4-5-20250929 for anthropic)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--version",
        type=str,
        help="Specify version directly (e.g., '1.2' or '2.0') to skip interactive prompt",
    )

    args = parser.parse_args()

    # Load environment variables from .env file
    load_dotenv()

    # Setup logging
    setup_cli_logging(verbose=args.verbose)

    # Check Ollama availability only if using Ollama provider
    if args.provider == "ollama":
        check_ollama_or_exit(logger)

    # Load config
    logger.info(f"Loading config from {args.config}")
    try:
        with args.config.open() as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        return 1
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        return 1

    if not config:
        logger.error("Config file is empty")
        return 1

    # Extract metadata from first example
    first_example = config[0]
    author = first_example["author"]
    scope = "persona"  # Default scope for now

    # Collect all unique authors from config
    authors = sorted(set(ex["author"] for ex in config))

    # Discover latest dataset for version bumping
    try:
        latest_path = discover_latest_golden_dataset(
            directory=args.output,
            scope=scope,
            authors=authors,
        )
        latest_dataset = load_golden_dataset(latest_path)
        current_version = latest_dataset.version
        logger.info(f"Found latest dataset: {latest_path.name} (version {current_version})")
    except FileNotFoundError:
        # No existing dataset, start at 1.0
        logger.warning("No existing dataset found. Starting at version 1.0")
        current_version = "0.0"  # Will be incremented to 1.0

    # Determine version (use --version flag if provided, otherwise prompt)
    if args.version:
        new_version = args.version
        logger.info(f"Using version from --version flag: {new_version}")
        # Validate version format
        try:
            parse_version(new_version)
        except ValueError as e:
            logger.error(str(e))
            return 1
    elif current_version == "0.0":
        new_version = "1.0"
        print(f"\nCreating initial dataset version: {new_version}")
    else:
        new_version = prompt_version_increment(current_version)

    # Initialize LLM with temperature=0 for deterministic generation
    try:
        llm = initialize_llm(
            provider=args.provider,
            model=args.model,
        )
    except ValueError as e:
        logger.error(str(e))
        return 1

    # Generate examples
    logger.info(f"Generating {len(config)} golden examples...")
    generated_examples: list[GoldenExample] = []

    for i, example_config in enumerate(config, 1):
        question = example_config["question"]
        author = example_config["author"]
        language = example_config["language"]

        logger.info(f"[{i}/{len(config)}] Generating: {question[:50]}...")

        try:
            # Build retriever for this author
            retriever = build_retriever(author=author, k=15)

            # Generate golden example
            golden_example = generate_golden_example_with_llm(
                question=question,
                author=author,
                language=language,
                llm=llm,
                retriever=retriever,
            )

            generated_examples.append(golden_example)
            logger.debug(f"Generated example ID: {golden_example.id}")

        except Exception as e:
            logger.error(f"Failed to generate example: {e}")
            logger.debug("Traceback:", exc_info=True)
            return 1

    # Create GoldenDataset
    today = date.today().isoformat()
    dataset = GoldenDataset(
        scope=scope,
        authors=authors,
        version=new_version,
        created_date=today,
        description=f"Golden dataset v{new_version} generated via LLM-based independent evaluation",
        examples=generated_examples,
    )

    # Save dataset
    filename = f"{dataset.identifier}.json"
    output_path = args.output / filename

    logger.info(f"Saving dataset to {output_path}")
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(dataset.model_dump(), f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save dataset: {e}")
        return 1

    # Print summary
    print("\n" + "=" * 70)
    print("DATASET GENERATED SUCCESSFULLY")
    print("=" * 70)
    print(f"  Scope: {dataset.scope}")
    print(f"  Authors: {', '.join(dataset.authors)}")
    print(f"  Version: {dataset.version}")
    print(f"  Date: {dataset.created_date}")
    print(f"  Examples: {len(dataset.examples)}")
    print(f"  Output: {output_path}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
