"""CLI script for re-generating ONLY `expected_chunk_ids` on an existing golden dataset.

Use this when the corpus was re-ingested with a different chunking parameters.
All other fields are copied verbatim from the source dataset.

There is no dedicated test for the script because it is primarily an argparse wrapper.
The core logic is extracted to rechunk.py, which is covered by unit testing.
Also, this script is a one-off CLI utility - not production code.

Usage:
    uv run python src/eval/golden/scripts/rechunk_expected_ids.py \\
        --source evals/golden/{filename}.json \\
        --version {version number}
        --provider ollama --model mistral
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from src.configs.eval import DEFAULT_GOLDEN_DATASET_PATH
from src.eval.golden.rechunk import rechunk_example, regenerate_expected_chunk_ids
from src.eval.utils import load_golden_dataset
from src.schemas.eval import GoldenDataset, GoldenExample
from src.utils.cli_helpers import check_ollama_or_exit
from src.utils.logging import setup_cli_logging

logger = logging.getLogger(__name__)


def _initialize_llm(provider: str, model: str | None) -> BaseChatModel:
    """Initialize LLM based on provider.

    Mirrors the initializer in generate_golden_dataset.py.
    Newer Anthropic models (e.g. Opus 4.7) no longer accept a temperature parameter,
    but Ollama is set to temperature=0 for determinism.
    """
    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Set it with: export ANTHROPIC_API_KEY='your-key-here'"
            )
        model_name = model or "claude-opus-4-7"
        logger.info(f"Initializing Anthropic LLM: {model_name}")
        return ChatAnthropic(model=model_name)  # type: ignore[call-arg]

    if provider == "ollama":
        model_name = model or "mistral"
        logger.info(f"Initializing Ollama LLM: {model_name} (temperature=0, format=json)")
        return ChatOllama(model=model_name, temperature=0, format="json")

    raise ValueError(f"Invalid provider: {provider}. Must be 'anthropic' or 'ollama'")


def main() -> int:
    """CLI entry point.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Re-generate expected_chunk_ids on an existing golden dataset after a "
            "chunk-size change. Preserves all other fields verbatim."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Path to the source golden dataset JSON",
    )
    parser.add_argument(
        "--version",
        required=True,
        type=str,
        help="New semantic version for the output dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_GOLDEN_DATASET_PATH,
        help=f"Output directory for the new dataset (default: {DEFAULT_GOLDEN_DATASET_PATH})",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["anthropic", "ollama"],
        default="anthropic",
        help="LLM provider (default: anthropic)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (default: claude-opus-4-7 for anthropic, mistral for ollama)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    load_dotenv()
    setup_cli_logging(verbose=args.verbose)

    if args.provider == "ollama":
        check_ollama_or_exit(logger)

    if not re.match(r"^\d+\.\d+$", args.version):
        logger.error(f"Invalid --version '{args.version}'. Expected 'X.Y'.")
        return 1

    try:
        source_dataset = load_golden_dataset(args.source)
    except FileNotFoundError:
        logger.error(f"Source dataset not found: {args.source}")
        return 1
    logger.info(
        f"Loaded source dataset {args.source.name} "
        f"(version {source_dataset.version}, {len(source_dataset.examples)} examples)"
    )

    try:
        llm = _initialize_llm(provider=args.provider, model=args.model)
    except ValueError as e:
        logger.error(str(e))
        return 1

    updated_examples: list[GoldenExample] = []
    for i, example in enumerate(source_dataset.examples, 1):
        logger.info(f"[{i}/{len(source_dataset.examples)}] Regenerating chunk_ids for {example.id}")
        try:
            new_ids = regenerate_expected_chunk_ids(example=example, llm=llm)
        except Exception as e:
            logger.error(f"Failed to regenerate chunk_ids for {example.id}: {e}")
            logger.debug("Traceback:", exc_info=True)
            return 1
        updated_examples.append(rechunk_example(example, new_ids))
        logger.debug(f"  {example.id}: {len(new_ids)} chunk_ids selected")

    today = date.today().isoformat()
    new_dataset = GoldenDataset(
        scope=source_dataset.scope,
        authors=source_dataset.authors,
        version=args.version,
        created_date=today,
        description=source_dataset.description,
        examples=updated_examples,
    )

    output_path = args.output / f"{new_dataset.identifier}.json"
    args.output.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(new_dataset.model_dump(), f, indent=2)

    print("\n" + "=" * 70)
    print("RE-CHUNKED DATASET WRITTEN")
    print("=" * 70)
    print(f"  Source:     {args.source}")
    print(f"  Output:     {output_path}")
    print(f"  Version:    {source_dataset.version} -> {new_dataset.version}")
    print(f"  Examples:   {len(new_dataset.examples)}")
    print(f"  Fields updated: expected_chunk_ids (per example), version, created_date")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
