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

# Import all metric modules to ensure they register themselves
import src.eval.metrics.retrieval  # noqa: F401


# Each metric must score at or above thresholds in order for an example to "pass"
METRIC_THRESHOLDS = {
    "retrieval_relevance": 0.8,
}


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


def _check_example_passed(metrics: list[MetricResult]) -> bool:
    """Check if all metrics pass their thresholds.

    An example "passes" if every metric's score meets or exceeds its threshold.

    Args:
        metrics: List of MetricResult objects

    Returns:
        True if all metrics pass, False otherwise
    """
    for metric in metrics:
        threshold = METRIC_THRESHOLDS.get(metric.name, 0.8)  # Default threshold 0.8
        if metric.score < threshold:
            return False
    return True


def run_eval(
    chain: Runnable[str, ChatResponse],
    golden_dataset: GoldenDataset,
) -> EvalRun:
    """Run evaluation harness on a chain using a golden dataset.

    This is the main entry point for evaluation.
    1. Accepts a LangChain runnable and a golden dataset
    2. Invokes the chain for each example in the dataset
    3. Applies metrics to each response
    4. Aggregates scores across all examples
    5. Returns a complete EvalRun object (machine-readable artifact)

    Args:
        chain: LangChain runnable that takes a question string and returns ChatResponse
        golden_dataset: GoldenDataset with test examples

    Returns:
        EvalRun object containing all results, scores, and metadata
    """
    # Process each example
    example_results: list[ExampleResult] = []
    passed_count = 0

    for example in golden_dataset.examples:
        # Invoke chain with the question
        response = chain.invoke(example.question)

        # Apply metrics
        metrics = _apply_metrics(example, response)

        # Check if example passed all metrics
        passed = _check_example_passed(metrics)
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
        dataset_version=golden_dataset.version,
        dataset_name=golden_dataset.name,
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        system_version=system_version,
        example_results=example_results,
        aggregate_scores=aggregate_scores,
        overall_pass_rate=overall_pass_rate,
    )
