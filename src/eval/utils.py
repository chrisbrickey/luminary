"""Utility functions for evaluation harness."""

import json
from pathlib import Path

from src.schemas.eval import GoldenDataset


def load_golden_dataset(path: Path) -> GoldenDataset:
    """Load and validate golden dataset from JSON file.

    Args:
        path: Path to golden dataset JSON file

    Returns:
        Validated GoldenDataset object

    Raises:
        FileNotFoundError: If path does not exist (include path in message)
        json.JSONDecodeError: If file is not valid JSON
        ValidationError: If JSON doesn't match GoldenDataset schema (Pydantic)
    """
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {path}\n")

    with path.open() as f:
        data = json.load(f)

    # Pydantic validation
    return GoldenDataset(**data)


def discover_latest_golden_dataset(
    directory: Path,
    author: str,
    scope: str = "golden",
) -> Path:
    """Find most recent golden dataset for an author.

    Filename format: {scope}_{author}_v{version}_{YYYY-MM-DD}.json
    Example: golden_voltaire_v1.0_2028-02-28.json

    Args:
        directory: Directory to search
        author: Author name (e.g., "voltaire", "gouges")
        scope: Dataset scope (default: "golden")

    Returns:
        Path to newest matching file (sorted by filename, newest first)

    Raises:
        FileNotFoundError: If no matching files found (include pattern and directory)
    """

    # lexicographic sorting
    #   1. ISO date format (YYYY-MM-DD) sorts correctly as strings: 2027-01-29 > 2027-01-28 (alphabetically)
    #   2. Version format (\d+\.\d+) sorts correctly for typical cases: v2.0 > v1.1 > v1.0 (alphabetically)
    #   3. reverse=True puts newest first: Higher versions come first; More recent dates come first
    #   edge case: v1.2 > v1.10 (lexicographically sorted but semantically wrong); Ok because the schema prevents multiple decimal places
    pattern = f"{scope}_{author}_v*.json"
    matches = sorted(directory.glob(pattern), reverse=True)

    if not matches:
        raise FileNotFoundError(
            f"No golden dataset found for author '{author}' in {directory}.\n"
            f"Expected pattern: {pattern}\n"
            f"Make sure you've created the dataset file first."
        )

    return matches[0]
