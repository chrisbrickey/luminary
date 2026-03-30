"""Tests for src/schemas/eval.py - evaluation harness schemas"""

from typing import Any

import pytest
from pydantic import ValidationError

from src.schemas.eval import MetricResult

# --- Shared test constants ---

METRIC_NAME = "test_metric"
VALID_SCORE = 0.75


def _metric_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return default MetricResult kwargs, with optional overrides."""
    defaults: dict[str, Any] = {
        "name": METRIC_NAME,
        "score": VALID_SCORE,
    }
    defaults.update(overrides)
    return defaults


class TestMetricResult:
    def test_construction_with_required_fields(self) -> None:
        metric = MetricResult(**_metric_kwargs())
        assert metric.name == METRIC_NAME
        assert metric.score == VALID_SCORE
        assert metric.details == {}

    def test_construction_with_details(self) -> None:
        test_details = {"reason": "sample reason", "count": 42}
        metric = MetricResult(**_metric_kwargs(details=test_details))
        assert metric.details == test_details

    def test_score_at_lower_bound(self) -> None:
        metric = MetricResult(**_metric_kwargs(score=0.0))
        assert metric.score == 0.0

    def test_score_at_upper_bound(self) -> None:
        metric = MetricResult(**_metric_kwargs(score=1.0))
        assert metric.score == 1.0

    def test_score_below_lower_bound_raises(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            MetricResult(**_metric_kwargs(score=-0.1))

    def test_score_above_upper_bound_raises(self) -> None:
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            MetricResult(**_metric_kwargs(score=1.1))

    def test_missing_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            MetricResult(score=VALID_SCORE)

    def test_missing_score_raises(self) -> None:
        with pytest.raises(ValidationError):
            MetricResult(name=METRIC_NAME)

    def test_score_type_enforced(self) -> None:
        with pytest.raises(ValidationError):
            MetricResult(name=METRIC_NAME, score="not-a-float")

    def test_details_defaults_to_empty_dict(self) -> None:
        metric = MetricResult(name="sample_metric", score=0.5)
        assert metric.details == {}
