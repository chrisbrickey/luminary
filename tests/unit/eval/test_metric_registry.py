"""Tests for src/eval/metrics/base.py — metric registry infrastructure."""

from unittest.mock import Mock

from src.eval.metrics.base import (
    METRIC_REGISTRY,
    MetricSpec,
    is_metric_applicable,
    register_metric,
)
from src.schemas.eval import MetricResult


# Test constants
METRIC_NAME = "test_metric"
SCORE_PERFECT = 1.0


class TestMetricSpec:
    """Tests for MetricSpec dataclass."""

    def test_create_metric_spec_with_all_fields(self) -> None:
        """MetricSpec can be created with all fields."""
        mock_compute = Mock(return_value=MetricResult(name=METRIC_NAME, score=SCORE_PERFECT))

        spec = MetricSpec(
            name=METRIC_NAME,
            compute=mock_compute,
            required_example_fields={"field1", "field2"},
            required_response_fields={"field3"},
            languages={"en", "fr"},
        )

        assert spec.name == METRIC_NAME
        assert spec.compute == mock_compute
        assert spec.required_example_fields == {"field1", "field2"}
        assert spec.required_response_fields == {"field3"}
        assert spec.languages == {"en", "fr"}

    def test_create_metric_spec_with_no_language_constraint(self) -> None:
        """MetricSpec with languages=None applies to all languages."""
        mock_compute = Mock()

        spec = MetricSpec(
            name=METRIC_NAME,
            compute=mock_compute,
            required_example_fields=set(),
            required_response_fields=set(),
            languages=None,
        )

        assert spec.languages is None


class TestRegisterMetric:
    """Tests for register_metric() function."""

    def test_register_metric_adds_to_registry(self) -> None:
        """Registering a metric adds it to METRIC_REGISTRY."""
        # Save initial registry state
        initial_count = len(METRIC_REGISTRY)

        # Register a new metric
        spec = MetricSpec(
            name="test_new_metric",
            compute=Mock(),
            required_example_fields=set(),
            required_response_fields=set(),
        )
        register_metric(spec)

        # Verify it was added
        assert len(METRIC_REGISTRY) == initial_count + 1
        assert spec in METRIC_REGISTRY

        # Cleanup - remove the test metric
        METRIC_REGISTRY.remove(spec)

    def test_register_multiple_metrics(self) -> None:
        """Multiple metrics can be registered."""
        initial_count = len(METRIC_REGISTRY)

        spec1 = MetricSpec(
            name="metric1",
            compute=Mock(),
            required_example_fields=set(),
            required_response_fields=set(),
        )
        spec2 = MetricSpec(
            name="metric2",
            compute=Mock(),
            required_example_fields=set(),
            required_response_fields=set(),
        )

        register_metric(spec1)
        register_metric(spec2)

        assert len(METRIC_REGISTRY) == initial_count + 2
        assert spec1 in METRIC_REGISTRY
        assert spec2 in METRIC_REGISTRY

        # Cleanup
        METRIC_REGISTRY.remove(spec1)
        METRIC_REGISTRY.remove(spec2)


class TestIsMetricApplicable:
    """Tests for is_metric_applicable() function."""

    # --- Required field tests ---

    def test_applicable_when_all_required_fields_present_and_truthy(self) -> None:
        """Metric is applicable when all required fields exist and are truthy."""
        spec = MetricSpec(
            name=METRIC_NAME,
            compute=Mock(),
            required_example_fields={"field1", "field2"},
            required_response_fields={"field3"},
        )

        example = Mock(field1="value1", field2="value2")
        response = Mock(field3="value3")

        assert is_metric_applicable(spec, example, response) is True

    def test_not_applicable_when_example_field_missing(self) -> None:
        """Metric is not applicable when required example field is missing."""
        spec = MetricSpec(
            name=METRIC_NAME,
            compute=Mock(),
            required_example_fields={"missing_field"},
            required_response_fields=set(),
        )

        example = Mock(spec=[])  # doesn't have 'missing_field'
        response = Mock()

        assert is_metric_applicable(spec, example, response) is False

    def test_not_applicable_when_example_field_is_falsy(self) -> None:
        """Metric is not applicable when required example field is falsy (empty list, None, etc.)."""
        spec = MetricSpec(
            name=METRIC_NAME,
            compute=Mock(),
            required_example_fields={"empty_list", "none_value"},
            required_response_fields=set(),
        )

        example = Mock(empty_list=[], none_value=None)
        response = Mock()

        assert is_metric_applicable(spec, example, response) is False

    def test_not_applicable_when_response_field_missing(self) -> None:
        """Metric is not applicable when required response field is missing."""
        spec = MetricSpec(
            name=METRIC_NAME,
            compute=Mock(),
            required_example_fields=set(),
            required_response_fields={"missing_field"},
        )

        example = Mock()
        response = Mock(spec=[])  # doesn't have 'missing_field'

        assert is_metric_applicable(spec, example, response) is False

    def test_not_applicable_when_response_field_is_falsy(self) -> None:
        """Metric is not applicable when required response field is falsy."""
        spec = MetricSpec(
            name=METRIC_NAME,
            compute=Mock(),
            required_example_fields=set(),
            required_response_fields={"empty_string"},
        )

        example = Mock()
        response = Mock(empty_string="")

        assert is_metric_applicable(spec, example, response) is False

    def test_applicable_when_no_required_fields(self) -> None:
        """Metric with no required fields is always applicable (ignoring language)."""
        spec = MetricSpec(
            name=METRIC_NAME,
            compute=Mock(),
            required_example_fields=set(),
            required_response_fields=set(),
            languages=None,
        )

        example = Mock()
        response = Mock()

        assert is_metric_applicable(spec, example, response) is True

    # --- Language constraint tests ---

    def test_applicable_when_no_language_constraint(self) -> None:
        """Metric with languages=None applies to all languages."""
        spec = MetricSpec(
            name=METRIC_NAME,
            compute=Mock(),
            required_example_fields=set(),
            required_response_fields=set(),
            languages=None,
        )

        example_en = Mock(language="en")
        example_fr = Mock(language="fr")
        example_es = Mock(language="es")
        response = Mock()

        assert is_metric_applicable(spec, example_en, response) is True
        assert is_metric_applicable(spec, example_fr, response) is True
        assert is_metric_applicable(spec, example_es, response) is True

    def test_applicable_when_language_matches(self) -> None:
        """Metric is applicable when example language matches constraint."""
        spec = MetricSpec(
            name=METRIC_NAME,
            compute=Mock(),
            required_example_fields=set(),
            required_response_fields=set(),
            languages={"en", "fr"},
        )

        example_en = Mock(language="en")
        example_fr = Mock(language="fr")
        response = Mock()

        assert is_metric_applicable(spec, example_en, response) is True
        assert is_metric_applicable(spec, example_fr, response) is True

    def test_not_applicable_when_language_does_not_match(self) -> None:
        """Metric is not applicable when example language doesn't match constraint."""
        spec = MetricSpec(
            name=METRIC_NAME,
            compute=Mock(),
            required_example_fields=set(),
            required_response_fields=set(),
            languages={"en"},
        )

        example_fr = Mock(language="fr")
        response = Mock()

        assert is_metric_applicable(spec, example_fr, response) is False

    def test_not_applicable_when_example_lacks_language_attribute(self) -> None:
        """Metric is not applicable when example has no language attribute but metric requires one."""
        spec = MetricSpec(
            name=METRIC_NAME,
            compute=Mock(),
            required_example_fields=set(),
            required_response_fields=set(),
            languages={"en"},
        )

        example = Mock(spec=[])  # No language attribute
        response = Mock()

        assert is_metric_applicable(spec, example, response) is False

    # --- Combined tests ---

    def test_not_applicable_when_language_matches_but_fields_missing(self) -> None:
        """Metric is not applicable even if language matches when required fields are missing."""
        spec = MetricSpec(
            name=METRIC_NAME,
            compute=Mock(),
            required_example_fields={"missing_field"},
            required_response_fields=set(),
            languages={"en"},
        )

        example = Mock(language="en", spec=[])  # Has language but not missing_field
        response = Mock()

        assert is_metric_applicable(spec, example, response) is False

    def test_applicable_when_all_constraints_satisfied(self) -> None:
        """Metric is applicable when all constraints (fields + language) are satisfied."""
        spec = MetricSpec(
            name=METRIC_NAME,
            compute=Mock(),
            required_example_fields={"expected_data"},
            required_response_fields={"actual_data"},
            languages={"fr"},
        )

        example = Mock(language="fr", expected_data=["item1"])
        response = Mock(actual_data=["item2"])

        assert is_metric_applicable(spec, example, response) is True
