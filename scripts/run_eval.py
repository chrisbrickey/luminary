"""CLI script for running evaluation harness.

This script provides a command-line interface to run the evaluation harness
against golden datasets, producing machine-readable JSON artifacts and
human-readable terminal output.

Usage:
    # Auto-discover latest golden dataset and run eval
    uv run python scripts/run_eval.py

    # Specify custom golden dataset
    uv run python scripts/run_eval.py --golden-path evals/golden/persona_voltaire_v1.0_2026-04-01.json

    # Specify custom output directory
    uv run python scripts/run_eval.py --output-path custom/output/dir

    # Enable verbose logging
    uv run python scripts/run_eval.py --verbose

Design principles:
- Auto-discovery by default: Discovers latest golden dataset and all registered metrics
- Multi-modal output: Persists JSON artifact + prints human-readable summary
- Clear error messages: Guides users to common fixes
- No hard-coded metrics: Works with any registered metrics via METRIC_REGISTRY
- Multilingual by default: Processes all languages (tests bilingual system as a whole)
"""

import argparse
import logging
import sys
from pathlib import Path

from src.chains.chat_chain import build_chain
from src.configs.eval import DEFAULT_EVAL_ARTIFACTS_PATH, DEFAULT_GOLDEN_DATASET_PATH
from src.eval.runner import run_eval
from src.eval.utils import discover_latest_golden_dataset, load_golden_dataset, save_eval_run
from src.schemas.eval import EvalRun, GoldenDataset
from src.utils.cli_helpers import check_ollama_or_exit
from src.utils.logging import setup_cli_logging

def _parser_epilogue() -> str:
    epilogue = """
    Examples:
      # Auto-discover latest dataset and run eval
      %(prog)s

      # Use specific golden dataset
      %(prog)s --golden-path evals/golden/persona_voltaire_v1.0_2026-04-01.json

      # Save to custom output directory
      %(prog)s --output-path custom/output

      # Override threshold for all metrics
      %(prog)s --threshold 0.9

      # Enable debug logging
      %(prog)s --verbose
      """

    return epilogue

def print_summary_table(eval_run: EvalRun) -> None:
    """Print human-readable summary table to stdout.
    This is a public method for re-useability and testability.

    Displays evaluation results in a formatted table with:
    - Dataset metadata (name, version, date)
    - System configuration (model, commit)
    - Overall scores for each metric with pass/fail status
    - By-language breakdown (if available)
    - Overall pass rate

    Args:
        eval_run: EvalRun object containing evaluation results
    """
    print("\n" + "=" * 70)
    print("EVALUATED DATASET:")
    print(f"  Identifier: {eval_run.dataset_identifier}")
    print(f"  Scope: {eval_run.dataset_scope}")
    print(f"  Authors: {', '.join(eval_run.dataset_authors)}")
    print(f"  Version: {eval_run.dataset_version}")
    print(f"  Date: {eval_run.dataset_date}")
    print(f"\nEVAL RUN TIMESTAMP: {eval_run.run_timestamp}")
    print(f"\nSYSTEM CONFIGURATION:")
    print(f"  Chat Model: {eval_run.system_version.get('chat_model', 'unknown')}")
    print(f"  Git Commit: {eval_run.system_version.get('commit', 'unknown')}")
    print("=" * 70)

    # Overall scores
    print("\nOVERALL SCORES")
    print("-" * 70)
    overall = eval_run.aggregate_scores.get("overall", {})

    if not overall:
        print("  No metrics available")
    else:
        for metric_name in sorted(overall.keys()):
            score = overall[metric_name]
            # Get threshold for this metric from effective_thresholds
            threshold = eval_run.effective_thresholds.get(metric_name, 0.8)
            status = "✅" if score >= threshold else "❌"
            print(f"  {metric_name:30s} {score:5.2f}  (threshold: {threshold:.2f}) {status}")

    # By-language breakdown
    if "by_language" in eval_run.aggregate_scores:
        for lang, scores in sorted(eval_run.aggregate_scores["by_language"].items()):
            print(f"\n{lang.upper()} ONLY")
            print("-" * 70)
            for metric_name in sorted(scores.keys()):
                score = scores[metric_name]
                threshold = eval_run.effective_thresholds.get(metric_name, 0.8)
                status = "✅" if score >= threshold else "❌"
                print(f"  {metric_name:30s} {score:5.2f}  (threshold: {threshold:.2f}) {status}")

    # Cross-language metrics (if available)
    if eval_run.aggregate_scores.get("cross_language"):
        print("\nCROSS-LANGUAGE METRICS")
        print("-" * 70)
        for metric_name in sorted(eval_run.aggregate_scores["cross_language"].keys()):
            score = eval_run.aggregate_scores["cross_language"][metric_name]
            threshold = eval_run.effective_thresholds.get(metric_name, 0.7)
            status = "✅" if score >= threshold else "❌"
            print(f"  {metric_name:30s} {score:5.2f}  (threshold: {threshold:.2f}) {status}")

    # Overall pass rate
    print("\n" + "=" * 70)
    passed = sum(1 for r in eval_run.example_results if r.passed)
    total = len(eval_run.example_results)
    print(f"OVERALL PASS RATE: {eval_run.overall_pass_rate:.1%} ({passed}/{total} examples)")
    print("=" * 70 + "\n")


def _print_next_steps(artifact_path: Path) -> None:
    """Print next steps after successful evaluation.

    Args:
        artifact_path: Path to the saved artifact file
    """
    print(f"\n✅ Evaluation complete!")
    print(f"\nNext steps:")
    print(f"  1. Review artifact: {artifact_path}")
    print(f"  2. Identify failing metrics (❌ in table above)")
    print(f"  3. Analyze failure modes in artifact JSON")
    print(f"  4. Make targeted improvements (prompts, config, dataset)")
    print(f"  5. Re-run eval to measure progress\n")


def _load_golden_dataset_from_args(
    golden_path: Path | None,
    logger: logging.Logger
) -> tuple[GoldenDataset, Path]:
    """Load golden dataset from CLI argument or auto-discovery.

    Args:
        golden_path: Explicit path to golden dataset, or None for auto-discovery
        logger: Logger for progress messages

    Returns:
        Tuple of (loaded GoldenDataset, resolved Path)

    Raises:
        FileNotFoundError: If dataset not found
    """
    if golden_path:
        logger.info(f"Loading golden dataset from: {golden_path}")
        resolved_path = golden_path
    else:
        logger.info(f"Auto-discovering latest golden dataset in: {DEFAULT_GOLDEN_DATASET_PATH}")
        # Auto-discovery uses default scope and author
        resolved_path = discover_latest_golden_dataset(directory=DEFAULT_GOLDEN_DATASET_PATH)
        logger.info(f"✓ Discovered: {resolved_path.name}")

    # Print out dataset attributes
    logger.info("Loading and validating golden dataset...")
    golden_dataset = load_golden_dataset(resolved_path)
    logger.info(f"✓ Loaded dataset: {golden_dataset.identifier}")
    logger.info(f"  - {len(golden_dataset.examples)} examples")
    logger.info(f"  - Scope: {golden_dataset.scope}")
    logger.info(f"  - Authors: {', '.join(golden_dataset.authors)}")

    return golden_dataset, resolved_path

def _check_metric_coverage(eval_run: EvalRun, logger: logging.Logger) -> None:
    """Check if all registered metrics were actually used in the evaluation.
    Warns if any metrics were registered but never applied to any examples.

    Args:
        eval_run: Completed EvalRun object
        logger: Logger for warnings
    """
    # Import metrics to get METRIC_REGISTRY
    from src.eval.metrics.base import METRIC_REGISTRY

    # Get all registered metric names
    registered_metrics = {spec.name for spec in METRIC_REGISTRY}

    # Get all metrics that were actually computed (appear in aggregate_scores)
    computed_metrics = set(eval_run.aggregate_scores.get("overall", {}).keys())

    # Find metrics that were registered but never computed
    unused_metrics = registered_metrics - computed_metrics

    if unused_metrics:
        logger.warning(
            f"\n{'='*70}\n"
            f"⚠️  WARNING: {len(unused_metrics)} registered metric(s) were NOT applied to any examples:\n"
        )

        for metric_name in sorted(unused_metrics):
            # Get the spec for diagnostics
            spec = next(s for s in METRIC_REGISTRY if s.name == metric_name)
            logger.warning(f"  - {metric_name}")
            logger.warning(f"      Required example fields: {spec.required_example_fields}")
            logger.warning(f"      Required response fields: {spec.required_response_fields}")
            logger.warning(f"      Languages: {spec.languages or 'all'}")

        print("\n" + "=" * 70)


def main() -> None:
    """Main CLI entry point for running evaluation harness.

    Workflow:
    - Parse command-line arguments
    - Setup logging (debug if --verbose)
    - Check Ollama availability
    - Load or discover golden dataset
    - Build chains for each author in the dataset
    - Run evaluation (by calling the eval runner)
    - Save artifact to disk (timestamped JSON)
    - Print summary table and next steps to stdout

    Exit codes:
    - 0: Success
    - 1: Error (Ollama unavailable, dataset not found, etc.)
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run evaluation harness against golden datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_parser_epilogue()
    )
    parser.add_argument(
        "--golden-path",
        type=Path,
        help="Path to golden dataset JSON file; If not provided the script calls an auto-discovery utility which falls back to default scope and author."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=f"Output directory for eval artifacts (default: {DEFAULT_EVAL_ARTIFACTS_PATH})"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        metavar="VALUE",
        help="Override default threshold for ALL metrics (e.g., --threshold 0.9). Must be between 0.0 and 1.0."
    )
    args = parser.parse_args()

    # Evaluate default arguments at runtime, avoiding variable setting at import time
    if args.output_path is None:
        args.output_path = DEFAULT_EVAL_ARTIFACTS_PATH

    # Validate threshold if provided
    if args.threshold is not None:
        if not (0.0 <= args.threshold <= 1.0):
            print(f"\n❌ ERROR: --threshold must be between 0.0 and 1.0, got {args.threshold}", file=sys.stderr)
            sys.exit(1)

    # Setup logging
    logger = setup_cli_logging(verbose=args.verbose)
    if args.verbose:
        logger.debug("Verbose logging enabled")

    # Check Ollama availability
    logger.info("Checking Ollama availability...")
    check_ollama_or_exit(logger)
    logger.info("✓ Ollama is running")

    # Load or discover golden dataset
    try:
        golden_dataset, golden_path = _load_golden_dataset_from_args(
            args.golden_path, logger
        )
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        print(f"\nTroubleshooting:", file=sys.stderr)
        print(f"  - Ensure golden dataset exists in: {DEFAULT_GOLDEN_DATASET_PATH}", file=sys.stderr)
        print(f"  - Check filename follows pattern: persona_{{author}}_v{{version}}_{{YYYY-MM-DD}}.json", file=sys.stderr)
        print(f"  - Or specify explicit path: --golden-path /path/to/dataset.json\n", file=sys.stderr)
        sys.exit(1)

    # Build chains for each author
    try:
        logger.info("Building chains for authors...")
        chains = {}
        for author in golden_dataset.authors:
            logger.debug(f"Building chain for author: {author}")
            chains[author] = build_chain(author=author)
        logger.info(f"✓ Built {len(chains)} chain(s)")
    except Exception as e:
        logger.exception("Error building chains")
        print(f"\n❌ ERROR building chains: {e}", file=sys.stderr)
        sys.exit(1)

    # Build threshold overrides if CLI argument provided
    override_thresholds = None
    if args.threshold is not None:
        # Import metrics to get METRIC_REGISTRY
        from src.eval.metrics.base import METRIC_REGISTRY
        # Apply the same threshold to all registered metrics
        override_thresholds = {spec.name: args.threshold for spec in METRIC_REGISTRY}
        logger.info(f"Overriding all metric thresholds to: {args.threshold}")

    # Run evaluation
    try:
        logger.info("Running evaluation harness...")
        logger.info("  (This may take several minutes depending on dataset size)")
        eval_run = run_eval(
            golden_dataset=golden_dataset,
            author_chains=chains,
            override_thresholds=override_thresholds,
        )
        logger.info(f"✓ Evaluation complete")
        logger.info(f"  - Overall pass rate: {eval_run.overall_pass_rate:.1%}")

        # Check for metrics that weren't applied
        _check_metric_coverage(eval_run, logger)
    except Exception as e:
        logger.exception("Error during evaluation")
        print(f"\n❌ ERROR during evaluation: {e}", file=sys.stderr)
        sys.exit(1)

    # Persist eval run artifact (machine-readable json)
    try:
        logger.info(f"Saving artifact to: {args.output_path}")
        artifact_path = save_eval_run(
            eval_run=eval_run,
            output_dir=args.output_path
        )
        logger.info(f"✓ Artifact saved: {artifact_path}")
    except (PermissionError, OSError) as e:
        logger.error(f"I/O error: {e}")
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        print(f"\nTroubleshooting:", file=sys.stderr)
        print(f"  - Check file permissions on output directory: {args.output_path}", file=sys.stderr)
        print(f"  - Ensure sufficient disk space available", file=sys.stderr)
        print(f"  - Try different output path: --output-path /tmp/eval_output\n", file=sys.stderr)
        sys.exit(1)

    # Print summary and next steps to terminal
    print_summary_table(eval_run)
    _print_next_steps(artifact_path)


if __name__ == "__main__":
    main()
