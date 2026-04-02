"""Pydantic schemas for evaluation harness."""

from typing import Annotated, Any

from pydantic import BaseModel, Field

from src.schemas.chat import ChatResponse


class MetricResult(BaseModel):
    """Result from a single metric on a single example.

    Uses standard scoring attributes:
        ge=0.0: greater than or equal to 0
        le=1.0: less than or equal to 1
    """

    name: str = Field(..., description="Name of the metric")
    score: float = Field(..., ge=0.0, le=1.0, description="Score 0.0 to 1.0 where 1.0 is perfect")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional details about the metric result")


class GoldenExample(BaseModel):
    """A single eval case in the golden dataset.

    Agile development: Add fields as new metrics are implemented. Only French and English are included for MVP.
    """

    id: str = Field(..., description="Unique identifier (e.g., 'tolerance_fr', 'pascal_en')")
    question: str = Field(..., description="The question to ask the philosopher")
    language: Annotated[str, Field(pattern=r"^(en|fr)$")] = Field(..., description="ISO 639-1 code: 'en' or 'fr'")

    # retrieval metrics
    expected_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Expected chunk IDs to be retrieved"
    )


class GoldenDataset(BaseModel):
    """Collection of eval cases with versioning.

    Tracks version and creation date to enable reproducible evaluation runs
    and comparison of results over time.

    - name field identifies the dataset (e.g., 'voltaire_golden', 'gouges_golden')
    - version field enables traceability
    - created_date provides temporal context for eval run comparisons
    - examples contain language-specific validation data (bilingual by design)
    """

    name: str = Field(..., description="Dataset name (e.g., 'voltaire_golden', 'gouges_golden')")
    version: Annotated[str, Field(pattern=r"^\d+\.\d+$")] = Field(
        ...,
        description="Semantic version (e.g., '1.0', '2.0')"
    )
    created_date: str = Field(..., description="ISO 8601 date (YYYY-MM-DD)")
    description: str = Field(..., description="What this dataset tests")
    examples: list[GoldenExample] = Field(..., description="List of test examples")


class ExampleResult(BaseModel):
    """The result for one example in the golden dataset across all metrics.

    Represents a complete evaluation of a single test case, including:
    - The question and expected language; Only French and English are included for MVP.
    - The actual system response
    - All metric results for this example
    - Overall pass/fail status (True if all required metrics above threshold)
    """

    example_id: str = Field(..., description="Unique identifier for this example")
    question: str = Field(..., description="The question that was asked")
    language: Annotated[str, Field(pattern=r"^(en|fr)$")] = Field(..., description="Language of the question")
    response: ChatResponse = Field(..., description="The chat response from the system")
    metrics: list[MetricResult] = Field(..., description="List of metric results for this example")
    passed: bool = Field(..., description="True if all required metrics above threshold")


class EvalRun(BaseModel):
    """Complete results from running evaluation harness (the JSON artifact).

    This is the machine-readable output saved to disk after running the evaluation
    harness. It contains all metadata needed for reproducibility and comparison:
    - Dataset and system version info
    - Individual results for every test case
    - Aggregated scores across all examples
    - Overall pass rate

    Design notes:
    - system_version captures reproducibility info (model, commit hash, config)
    - example_results is a list instead of structure with constant access (e.g. dict) because:
        - Order matters in eval runs. We want to see results in the order examples were processed.
        - JSON serialization is simpler with lists.
        - Evaluation harness is not on the hot path. Time complexity is not critical.
        - If we need constant lookup in production, we can build it at the call site.
    - aggregate_scores has nested structure:
      {"overall": {...}, "by_language": {"en": {...}, "fr": {...}}, "cross_language": {...}}
    - This artifact is saved to evals/runs/ and referenced in narrative reports
    """

    # Metadata
    dataset_version: str = Field(..., description="Version of the golden dataset used")
    dataset_name: str = Field(..., description="Name of the dataset (e.g., 'voltaire_golden')")
    run_timestamp: str = Field(..., description="ISO 8601 timestamp with timezone")

    # System configuration (for reproducibility)
    system_version: dict[str, str] = Field(..., description="System config: model, commit, etc.")

    # Results
    example_results: list[ExampleResult] = Field(..., description="Results for each example")
    aggregate_scores: dict[str, Any] = Field(..., description="Overall + Breakdown by language")
    overall_pass_rate: float = Field(..., ge=0.0, le=1.0, description="Fraction of examples passing all metrics")
