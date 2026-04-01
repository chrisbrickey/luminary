"""Pydantic schemas for evaluation harness."""

from typing import Annotated, Any

from pydantic import BaseModel, Field


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

    Agile development: Add fields as new metrics are implemented.
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

    - version field enables traceability (matches filename convention
      {author}_golden_v{version}_{YYYY-MM-DD}.json)
    - created_date provides temporal context for eval run comparisons
    - examples contain language-specific validation data (bilingual by design)
    """

    version: Annotated[str, Field(pattern=r"^\d+\.\d+$")] = Field(
        ...,
        description="Semantic version (e.g., '1.0', '2.0')"
    )
    created_date: str = Field(..., description="ISO 8601 date (YYYY-MM-DD)")
    description: str = Field(..., description="What this dataset tests")
    examples: list[GoldenExample] = Field(..., description="List of test examples")
