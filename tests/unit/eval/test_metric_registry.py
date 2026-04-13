"""Tests for src/eval/metrics/base.py — metric registry infrastructure."""

from pathlib import Path
from unittest.mock import Mock

from src.configs.common import ENGLISH_ISO_CODE, FRENCH_ISO_CODE, SPANISH_ISO_CODE
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
            languages={ENGLISH_ISO_CODE, FRENCH_ISO_CODE},
        )

        assert spec.name == METRIC_NAME
        assert spec.compute == mock_compute
        assert spec.required_example_fields == {"field1", "field2"}
        assert spec.required_response_fields == {"field3"}
        assert spec.languages == {ENGLISH_ISO_CODE, FRENCH_ISO_CODE}

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

        example_en = Mock(language=ENGLISH_ISO_CODE)
        example_fr = Mock(language=FRENCH_ISO_CODE)
        example_es = Mock(language=SPANISH_ISO_CODE)
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
            languages={ENGLISH_ISO_CODE, FRENCH_ISO_CODE},
        )

        example_en = Mock(language=ENGLISH_ISO_CODE)
        example_fr = Mock(language=FRENCH_ISO_CODE)
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
            languages={ENGLISH_ISO_CODE},
        )

        example_fr = Mock(language=FRENCH_ISO_CODE)
        response = Mock()

        assert is_metric_applicable(spec, example_fr, response) is False

    def test_not_applicable_when_example_lacks_language_attribute(self) -> None:
        """Metric is not applicable when example has no language attribute but metric requires one."""
        spec = MetricSpec(
            name=METRIC_NAME,
            compute=Mock(),
            required_example_fields=set(),
            required_response_fields=set(),
            languages={ENGLISH_ISO_CODE},
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
            languages={ENGLISH_ISO_CODE},
        )

        example = Mock(language=ENGLISH_ISO_CODE, spec=[])  # Has language but not missing_field
        response = Mock()

        assert is_metric_applicable(spec, example, response) is False

    def test_applicable_when_all_constraints_satisfied(self) -> None:
        """Metric is applicable when all constraints (fields + language) are satisfied."""
        spec = MetricSpec(
            name=METRIC_NAME,
            compute=Mock(),
            required_example_fields={"expected_data"},
            required_response_fields={"actual_data"},
            languages={FRENCH_ISO_CODE},
        )

        example = Mock(language=FRENCH_ISO_CODE, expected_data=["item1"])
        response = Mock(actual_data=["item2"])

        assert is_metric_applicable(spec, example, response) is True


class TestMetricRegistryCompleteness:
    """Tests that validate all metrics are properly registered."""

    def test_every_metric_file_registers_at_least_one_metric(self) -> None:
        """Every .py file in src/eval/metrics/ (except base/__init__) registers at least one metric.

        This ensures new metric files don't forget to call register_metric().
        """
        # Import metrics to trigger auto-registration; Skip unused import warning
        import src.eval.metrics  # noqa: F401

        # Find all metric module files
        metrics_dir = Path("src/eval/metrics")
        metric_files = {
            f.stem
            for f in metrics_dir.glob("*.py")
            if f.stem not in ("__init__", "base")
        }

        # Build mapping of module → metrics
        module_to_metrics: dict[str, list[str]] = {}
        for spec in METRIC_REGISTRY:
            # Extract module name from compute function
            module_name = spec.compute.__module__
            # Extract just the file stem (e.g., "src.eval.metrics.retrieval" → "retrieval")
            if module_name.startswith("src.eval.metrics."):
                file_stem = module_name.split(".")[-1]
                if file_stem not in module_to_metrics:
                    module_to_metrics[file_stem] = []
                module_to_metrics[file_stem].append(spec.name)

        # Verify every metric file has at least one registered metric
        files_without_metrics = metric_files - set(module_to_metrics.keys())

        assert not files_without_metrics, (
            f"Metric files found with no registered metrics: {sorted(files_without_metrics)}. "
            f"Each .py file in src/eval/metrics/ must register at least one metric. "
            f"Files with metrics: {sorted(module_to_metrics.keys())}"
        )

    def test_all_registered_metrics_come_from_metrics_directory(self) -> None:
        """Every metric in METRIC_REGISTRY must be defined in src/eval/metrics/.

        This prevents metrics from being registered from outside the metrics directory.
        """
        # Import metrics to trigger auto-registration; Skip unused import warning
        import src.eval.metrics  # noqa: F401

        # Check all metrics come from the metrics directory
        invalid_metrics = []
        for spec in METRIC_REGISTRY:
            module_name = spec.compute.__module__
            if not module_name.startswith("src.eval.metrics."):
                invalid_metrics.append((spec.name, module_name))

        assert not invalid_metrics, (
            f"Metrics registered from outside src.eval.metrics/: {invalid_metrics}. "
            f"All metrics must be defined in modules under src/eval/metrics/"
        )

    def test_no_duplicate_metric_names(self) -> None:
        """Each metric name appears only once in the registry.

        This prevents bugs where two metrics use the same name and one
        silently overwrites the other.
        """
        # Import metrics to trigger auto-registration; Skip unused import warning
        import src.eval.metrics  # noqa: F401

        metric_names = [spec.name for spec in METRIC_REGISTRY]
        unique_names = set(metric_names)

        assert len(metric_names) == len(unique_names), (
            f"Duplicate metric names found. "
            f"All names: {sorted(metric_names)}. "
            f"Unique names: {sorted(unique_names)}"
        )

    def test_all_metric_registrations_succeed(self) -> None:
        """Verify every register_metric() call in source files actually registered a metric.

        This catches:
        - Metrics defined but not registered (forgot to call register_metric)
        - Registration calls that silently fail (import errors, exceptions)
        - Conditional registrations that didn't execute

        This test is fully dynamic and requires no manual updates when adding new metrics.
        """
        # Import metrics to trigger auto-registration; Skip unused import warning
        import src.eval.metrics  # noqa: F401

        # Count register_metric() calls in all metric files
        metrics_dir = Path("src/eval/metrics")
        total_registration_calls = 0
        registration_details = []

        for metric_file in metrics_dir.glob("*.py"):
            if metric_file.stem in ("__init__", "base"):
                continue

            content = metric_file.read_text()
            registration_calls = content.count("register_metric(")
            total_registration_calls += registration_calls
            if registration_calls > 0:
                registration_details.append(f"  {metric_file.name}: {registration_calls} call(s)")

        # Verify at least some metrics exist
        assert total_registration_calls > 0, (f"No register_metric() calls found in {metrics_dir}.")

        # Verify registry size matches total registration calls
        actual_registered = len(METRIC_REGISTRY)
        assert actual_registered == total_registration_calls, (
            f"Found {total_registration_calls} register_metric() calls in source files, but only {actual_registered} metrics in METRIC_REGISTRY.\n"
            f"Registration details:\n" + "\n".join(registration_details) + f"\n"
        )
