"""Base infrastructure for the metrics registry.

The registry pattern allows metrics to be discovered and applied automatically
without modifying eval runners. Each metric declares what fields it needs and
when it applies (e.g., specific languages, required example fields).

These structures are intentionally omitted from pydantic schemas because
they are never serialized or persisted and because it allows all registry
logic to be co-located on this file.
"""

from dataclasses import dataclass
from typing import Any, Callable

from src.schemas.eval import MetricResult


# Default threshold for metrics when not explicitly specified
FALLBACK_THRESHOLD = 0.8


@dataclass
class MetricSpec:
    """Specification for a metric that can be applied to evaluation examples.

    Attributes:
        name: Unique identifier for this metric
        compute: Function that takes (example, response) and returns MetricResult
        required_example_fields: Set of attribute names that must exist on example
        required_response_fields: Set of attribute names that must exist on response
        languages: Set of language codes this metric applies to (None = all languages)
        default_threshold: Minimum score required for this metric to pass (0.0 to 1.0)
    """

    name: str
    compute: Callable[[Any, Any], MetricResult]
    required_example_fields: set[str]
    required_response_fields: set[str]
    languages: set[str] | None = None
    default_threshold: float = FALLBACK_THRESHOLD


# Global registry of all available metrics
METRIC_REGISTRY: list[MetricSpec] = []


def register_metric(spec: MetricSpec) -> None:
    """Register a metric in the global registry.

    Args:
        spec: MetricSpec describing the metric and its applicability rules
    """
    METRIC_REGISTRY.append(spec)


def is_metric_applicable(
    spec: MetricSpec,
    example: Any,
    response: Any,
) -> bool:
    """Check if a metric can be applied to a given example/response pair.

    A metric is applicable if:
    1. All required example fields exist and are truthy
    2. All required response fields exist and are truthy
    3. The example's language matches (if the metric specifies languages)

    Args:
        spec: MetricSpec to check
        example: Example object (typically GoldenExample)
        response: Response object (typically ChatResponse)

    Returns:
        True if the metric can be applied, False otherwise
    """
    # Check required example fields
    for field in spec.required_example_fields:
        if not hasattr(example, field) or not getattr(example, field):
            return False

    # Check required response fields
    for field in spec.required_response_fields:
        if not hasattr(response, field) or not getattr(response, field):
            return False

    # Check language constraint
    if spec.languages is not None:
        if not hasattr(example, "language"):
            return False
        if example.language not in spec.languages:
            return False

    return True
