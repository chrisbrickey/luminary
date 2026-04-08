"""Evaluation runner for the RAG system.

Design principle: The runner is pure business logic. It does NOT:
- Load datasets from disk (caller's responsibility)
- Save results to disk (caller's responsibility)
This separation enables programmatic use (e.g., scripts, notebooks) and simplifies testing.
"""

import subprocess
from datetime import datetime, timezone
from typing import Any

from langchain_core.runnables import Runnable

from src.configs.common import DEFAULT_CHAT_MODEL
from src.eval.metrics.base import METRIC_REGISTRY, is_metric_applicable
from src.schemas.chat import ChatResponse
from src.schemas.eval import EvalRun, ExampleResult, GoldenDataset, MetricResult

# Auto-register all metrics; Skip unused import warning
import src.eval.metrics  # noqa: F401


def _get_system_version() -> dict[str, str]:
    """Capture system version for reproducibility.

    Returns dict with:
    - chat_model: Name of the chat model used
    - commit: Git commit hash (short form)
    - timestamp: ISO 8601 timestamp with timezone

    If git is unavailable or not in a repo, commit will be "unknown".
    """
    # Get git commit hash (short form)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"

    return {
        "chat_model": DEFAULT_CHAT_MODEL,
        "commit": commit,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _compute_aggregate_scores(
    example_results: list[ExampleResult],
) -> dict[str, Any]:
    """Aggregate metric scores across all examples.

    Computes:
    - overall: Average score for each metric across all examples
    - by_language: Average score for each metric, grouped by language

    Args:
        example_results: List of ExampleResult objects

    Returns:
        Dict with structure:
        {
            "overall": {"metric_name": avg_score, ...},
            "by_language": {
                "en": {"metric_name": avg_score, ...},
                "fr": {"metric_name": avg_score, ...}
            }
        }
    """
    if not example_results:
        return {"overall": {}, "by_language": {}}

    overall_scores: dict[str, list[float]] = {}
    by_language_scores: dict[str, dict[str, list[float]]] = {}

    for result in example_results:
        # Initialize language dict if needed
        if result.language not in by_language_scores:
            by_language_scores[result.language] = {}

        for metric in result.metrics:
            # Collect for overall
            if metric.name not in overall_scores:
                overall_scores[metric.name] = []
            overall_scores[metric.name].append(metric.score)

            # Collect for by-language
            if metric.name not in by_language_scores[result.language]:
                by_language_scores[result.language][metric.name] = []
            by_language_scores[result.language][metric.name].append(metric.score)

    # Compute averages
    overall: dict[str, float] = {}
    for metric_name, scores in overall_scores.items():
        overall[metric_name] = sum(scores) / len(scores)

    by_language: dict[str, dict[str, float]] = {}
    for lang, metrics in by_language_scores.items():
        by_language[lang] = {}
        for metric_name, scores in metrics.items():
            by_language[lang][metric_name] = sum(scores) / len(scores)

    return {
        "overall": overall,
        "by_language": by_language,
    }

def _build_effective_thresholds(overrides: dict[str, float] | None) -> dict[str, float]:
    """Create record of effective thresholds that will be used to determine success/failure.

    Automatically discovers all thresholds from each metric in the global registry.
    Applies any overrides from the primary entry point.

    Args:
        override_thresholds: Optional dict to override default thresholds per metric.

    Returns:
        Dict of thresholds (0.0 to 1.0) for each registered metric
    """
    # Collect default thresholds from each metric
    effective_thresholds: dict[str, float] = {}
    for spec in METRIC_REGISTRY:
        effective_thresholds[spec.name] = spec.default_threshold

    # Apply any override thresholds
    override_thresholds = {} if overrides is None else overrides
    effective_thresholds.update(override_thresholds)

    return effective_thresholds

def _apply_metrics(
    example: Any,  # GoldenExample
    response: ChatResponse,
) -> list[MetricResult]:
    """Apply all applicable metrics to a single example.

    Automatically discovers and applies metrics from the global registry
    based on their applicability rules (required fields, language constraints).

    Args:
        example: GoldenExample with expected values
        response: ChatResponse from the system

    Returns:
        List of MetricResult objects from all applicable metrics
    """
    metrics = []

    for spec in METRIC_REGISTRY:
        if is_metric_applicable(spec, example, response):
            metric_result = spec.compute(example, response)
            metrics.append(metric_result)

    return metrics


def _check_example_passed(
    metrics: list[MetricResult],
    thresholds: dict[str, float],
) -> bool:
    """Check if all metrics pass their thresholds.

    An example "passes" if every metric's score meets or exceeds its threshold.

    Args:
        metrics: List of MetricResult objects
        thresholds: Dict mapping metric names to threshold scores

    Returns:
        True if all metrics pass, False otherwise
    """
    for metric in metrics:
        if metric.score < thresholds[metric.name]:
            return False
    return True


def _validate_chains(
    dataset_authors: list[str],
    chains: dict[str, Runnable[str, ChatResponse]],
) -> None:
    """Validate that all required chains are provided.

    Args:
        dataset_authors: List of authors from GoldenDataset
        chains: Mapping from author key to chain

    Raises:
        ValueError: If any required chain is missing

    Warns:
        If extra chains are provided that aren't needed by the dataset
    """
    import warnings

    chain_authors = set(chains.keys())
    required_authors = set(dataset_authors)

    missing = required_authors - chain_authors
    if missing:
        raise ValueError(
            f"Missing chains for authors: {sorted(missing)}. "
            f"Dataset requires chains for: {sorted(required_authors)}"
        )

    # Warn on extra chains (helpful for debugging)
    extra = chain_authors - required_authors
    if extra:
        warnings.warn(
            f"Extra chains provided that are not needed by dataset: {sorted(extra)}. "
            f"Dataset only requires: {sorted(required_authors)}",
            UserWarning,
            stacklevel=3,
        )

def run_eval(
    golden_dataset: GoldenDataset,
    author_chains: dict[str, Runnable[str, ChatResponse]],
    override_thresholds: dict[str, float] | None = None,
) -> EvalRun:
    """Run evaluation harness on multiple author-specific chains using a golden dataset.

    This is the main entry point for evaluation.
    1. Accepts a golden dataset and a dict of author-specific LangChain runnables
    2. Routes each example to its corresponding author's chain
    3. Invokes the chain with the question and language
    4. Applies metrics to each response
    5. Aggregates scores across all examples
    6. Returns a complete EvalRun object (machine-readable artifact)

    Example usage:
        from src.chains.chat_chain import build_chain

        # Build but do not yet invoke chains for all authors in dataset
        chains = {
            "voltaire": build_chain(author="voltaire"),
            "gouges": build_chain(author="gouges"),
        }

        result = run_eval(golden_dataset, chains)

    Args:
        golden_dataset: GoldenDataset with test examples
        author_chains: Dict mapping author keys to LangChain runnables.
            Each chain takes a question string and returns ChatResponse.
            Must include chains for all authors in golden_dataset.authors.
        override_thresholds: Optional dict to override default thresholds per metric.
            Keys are metric names, values are threshold scores (0.0 to 1.0).
            If not provided, uses default_threshold from each MetricSpec.

    Returns:
        EvalRun object containing all results, scores, and metadata

    Raises:
        ValueError: If any required chain is missing from the chains dict

    Warns:
        If extra chains are provided that aren't needed by the dataset
    """
    # Validate chains before processing
    _validate_chains(golden_dataset.authors, author_chains)

    # Determine thresholds to use for each metric
    effective_thresholds: dict[str, float] = _build_effective_thresholds(override_thresholds)

    # Process each example
    example_results: list[ExampleResult] = []
    passed_count = 0

    for example in golden_dataset.examples:
        # Route to the appropriate author's chain
        chain = author_chains[example.author]

        # Invoke chain with the question and language
        response = chain.invoke(example.question, language=example.language)

        # Apply metrics
        metrics = _apply_metrics(example, response)

        # Check if example passed all metrics
        passed = _check_example_passed(metrics, effective_thresholds)
        if passed:
            passed_count += 1

        # Create ExampleResult
        example_result = ExampleResult(
            example_id=example.id,
            question=example.question,
            language=example.language,
            response=response,
            metrics=metrics,
            passed=passed,
        )
        example_results.append(example_result)

    # Aggregate scores
    aggregate_scores = _compute_aggregate_scores(example_results)

    # Calculate overall pass rate
    overall_pass_rate = passed_count / len(example_results) if example_results else 0.0

    # Capture system version
    system_version = _get_system_version()

    # Create and return EvalRun
    return EvalRun(
        dataset_scope=golden_dataset.scope,
        dataset_authors=sorted(golden_dataset.authors),
        dataset_identifier=golden_dataset.identifier,
        dataset_version=golden_dataset.version,
        dataset_date=golden_dataset.created_date,
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        system_version=system_version,
        effective_thresholds=effective_thresholds,
        example_results=example_results,
        aggregate_scores=aggregate_scores,
        overall_pass_rate=overall_pass_rate,
    )
