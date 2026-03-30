"""Pydantic schemas for evaluation harness."""

from typing import Any

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
