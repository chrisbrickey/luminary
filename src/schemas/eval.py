"""Pydantic schemas for evaluation harness."""

from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator, model_validator

from src.configs.common import ENGLISH_ISO_CODE, FRENCH_ISO_CODE
from src.schemas.chat import ChatResponse

# Languages evaluated
EVALUATED_LANGUAGES = [ENGLISH_ISO_CODE, FRENCH_ISO_CODE]

# Build language validation pattern dynamically from EVALUATED_LANGUAGES
_LANG_PATTERN = f"^({'|'.join(EVALUATED_LANGUAGES)})$"


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
    author: str = Field(..., description="Which author should answer (must be in AUTHOR_CONFIGS)")
    language: Annotated[str, Field(pattern=_LANG_PATTERN)] = Field(..., description="ISO 639-1 code from EVALUATED_LANGUAGES")

    # retrieval metrics
    expected_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Expected chunk IDs to be retrieved"
    )

    # citation metrics
    expected_source_titles: list[str] = Field(
        default_factory=list,
        description="Expected source titles in citations"
    )

    @field_validator('author')
    @classmethod
    def validate_author_in_configs(cls, v: str) -> str:
        """Ensure author exists in AUTHOR_CONFIGS registry."""
        from src.configs.authors import AUTHOR_CONFIGS
        if v not in AUTHOR_CONFIGS:
            valid = list(AUTHOR_CONFIGS.keys())
            raise ValueError(
                f"Author '{v}' not found in AUTHOR_CONFIGS. Valid authors: {valid}"
            )
        return v


class GoldenDataset(BaseModel):
    """Collection of eval cases with versioning.

    Tracks version and creation date to enable reproducible evaluation runs
    and comparison of results over time.

    Identifies datasets by explicit scope + authors with computed full identifier.

    - scope field identifies the dataset category (e.g., 'persona', 'retrieval')
    - authors lists all philosophers covered by this dataset (must match examples)
    - version field enables traceability
    - created_date provides temporal context for eval run comparisons
    - examples contain language-specific validation data (bilingual by design)
    - identifier property (computed): '{scope}_{sorted_authors}_v{version}_{date}'
    """

    scope: str = Field(..., description="Dataset scope (e.g., 'persona', 'retrieval', 'bilingual')")
    authors: list[str] = Field(
        ...,
        description="All authors covered by this dataset (must be in AUTHOR_CONFIGS)"
    )
    version: Annotated[str, Field(pattern=r"^\d+\.\d+$")] = Field(
        ...,
        description="Semantic version (e.g., '1.0', '2.0')"
    )
    created_date: str = Field(..., description="ISO 8601 date (YYYY-MM-DD)")
    description: str = Field(..., description="What this dataset tests")
    examples: list[GoldenExample] = Field(..., description="List of test examples")

    @field_validator('authors')
    @classmethod
    def validate_all_authors_in_configs_and_sorted(cls, v: list[str]) -> list[str]:
        """Ensure all authors exist in AUTHOR_CONFIGS registry and list is sorted."""
        from src.configs.authors import AUTHOR_CONFIGS

        # Check all authors are valid
        invalid = [author for author in v if author not in AUTHOR_CONFIGS]
        if invalid:
            valid = list(AUTHOR_CONFIGS.keys())
            raise ValueError(
                f"Authors {invalid} not found in AUTHOR_CONFIGS. Valid authors: {valid}"
            )

        # Check list is sorted alphabetically
        if v != sorted(v):
            raise ValueError(
                f"Authors list must be sorted alphabetically. Got {v}, expected {sorted(v)}"
            )

        return v

    @model_validator(mode='after')
    def validate_authors_match_examples(self) -> 'GoldenDataset':
        """Ensure dataset.authors contains exactly the authors from all examples."""
        example_authors = {ex.author for ex in self.examples}
        metadata_authors = set(self.authors)

        if example_authors != metadata_authors:
            missing_in_metadata = example_authors - metadata_authors
            extra_in_metadata = metadata_authors - example_authors
            msg = "Mismatch between dataset.authors and example authors."
            if missing_in_metadata:
                msg += f" Missing in metadata: {sorted(missing_in_metadata)}."
            if extra_in_metadata:
                msg += f" Extra in metadata: {sorted(extra_in_metadata)}."
            raise ValueError(msg)

        return self

    @property
    def identifier(self) -> str:
        """Unique human-readable identifier composed of dataset properties including version and date.
        Use as dataset filename and as the unique reference for traceability.

        Format: {scope}_{sorted_authors}_v{version}_{date}

        Returns:
            Complete identifier (e.g., 'persona_voltaire_v1.0_2026-04-01')
        """
        return f"{self.scope}_{'_'.join(sorted(self.authors))}_v{self.version}_{self.created_date}"


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
    language: Annotated[str, Field(pattern=_LANG_PATTERN)] = Field(..., description="Language of the question")
    response: ChatResponse = Field(..., description="The chat response from the system")
    metrics: list[MetricResult] = Field(..., description="List of metric results for this example")
    passed: bool = Field(..., description="True if all required metrics above threshold")


class SystemVersion(BaseModel):
    """System configuration captured at eval run time for reproducibility.

    All fields are optional for backwards compatability. This permits loading
    older eval artifact json files which may not include all of the current fields.
    """

    commit: str | None = Field(default=None, title="Commit", description="Git commit hash (short form) at run time")
    timestamp: str | None = Field(default=None, title="Timestamp", description="ISO 8601 timestamp with timezone")
    chat_model: str | None = Field(default=None, title="Chat Model", description="Name of the chat model used")
    embedding_model: str | None = Field(default=None, title="Embedding Model", description="Name of the embedding model used")
    retrieval_chunk_count: str | None = Field(default=None, title="Retrieval Chunk Count (k)", description="Number of chunks retrieved per query")
    retrieval_chunk_size: str | None = Field(default=None, title="Retrieval Chunk Size", description="Maximum chunk size in characters")


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
    - dataset_scope and dataset_authors are explicit fields for queryability
    - dataset_identifier stores the full versioned identifier for traceability
    """

    # Metadata - Dataset identification
    dataset_scope: str = Field(..., description="Scope from GoldenDataset (e.g., 'persona', 'retrieval')")
    dataset_authors: list[str] = Field(..., description="Authors from GoldenDataset (sorted)")
    dataset_identifier: str = Field(..., description="Full identifier: '{scope}_{sorted_authors}_v{version}_{date}'")
    dataset_version: str = Field(..., description="Version of the golden dataset used")
    dataset_date: str = Field(..., description="ISO 8601 date from GoldenDataset.created_date (YYYY-MM-DD)")
    run_timestamp: str = Field(..., description="ISO 8601 timestamp with timezone")

    # System configuration (for reproducibility)
    system_version: SystemVersion = Field(..., description="System config: chat model, commit, etc.")
    effective_thresholds: dict[str, float] = Field(..., description="Thresholds used for pass/fail (metric_name -> threshold)")

    # Results
    example_results: list[ExampleResult] = Field(..., description="Results for each example")
    aggregate_scores: dict[str, Any] = Field(..., description="Overall + Breakdown by language")
    overall_pass_rate: float = Field(..., ge=0.0, le=1.0, description="Fraction of examples passing all metrics")
