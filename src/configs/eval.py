"""Configuration constants for evaluation harness."""

from pathlib import Path

# used by eval harness for loading golden datasets
DEFAULT_GOLDEN_DATASET_PATH = Path("data/raw/golden")

# used by eval harness for saving eval run artifacts
DEFAULT_EVAL_ARTIFACTS_PATH = Path("evals/runs")
