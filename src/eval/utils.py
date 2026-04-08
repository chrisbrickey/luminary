"""Utility functions for evaluation harness."""

import json
from pathlib import Path

from src.configs.authors import DEFAULT_AUTHOR
from src.schemas.eval import GoldenDataset


def load_golden_dataset(path: Path) -> GoldenDataset:
    """Load and validate golden dataset from specified JSON file.

    Intended to be used in conjunction with discover_latest_golden_dataset,
    which returns the path to the most recent of versioned datasets.

    Args:
        path: Path to one specific golden dataset JSON file

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
    scope: str = "persona",
    authors: list[str] | None = None, # avoids mutable default argument
) -> Path:
    """Find most recent golden dataset for author(s).

    Filename format: {scope}_{authors}_v{version}_{YYYY-MM-DD}.json
    Example (single author): persona_voltaire_v1.0_2028-02-23.json
    Example (multi author): persona_gouges_voltaire_v1.0_2028-02-23.json

    Filename components map directly to GoldenDataset schema fields:
    - {scope} → GoldenDataset.scope
    - {authors} → '_'.join(sorted(GoldenDataset.authors))
    - {version} → GoldenDataset.version
    - {YYYY-MM-DD} → GoldenDataset.created_date

    Args:
        directory: Directory to search
        scope: Dataset scope (default: "persona") - matches GoldenDataset.scope
        authors: List of author names (e.g., ["condorcet"]) - will be sorted and joined
                 with underscores to match the authors portion of the filename.
                 Defaults to [DEFAULT_AUTHOR] if not provided.

    Returns:
        Path to newest matching file (sorted by filename, newest first)

    Raises:
        FileNotFoundError: If no matching files found (include pattern and directory)
    """
    if authors is None:
        authors = [DEFAULT_AUTHOR]

    # Sort and join authors to match golden dataset filename convention
    authors_str = "_".join(sorted(authors))

    # lexicographic sorting
    #   1. ISO date format (YYYY-MM-DD) sorts correctly as strings: 2027-01-29 > 2027-01-28 (alphabetically)
    #   2. Version format (\d+\.\d+) sorts correctly for typical cases: v2.0 > v1.1 > v1.0 (alphabetically)
    #   3. reverse=True puts newest first: Higher versions come first; More recent dates come first
    #   edge case: v1.2 > v1.10 (lexicographically sorted but semantically wrong); Ok because the schema prevents multiple decimal places
    pattern = f"{scope}_{authors_str}_v*.json"
    matches = sorted(directory.glob(pattern), reverse=True)

    if not matches:
        raise FileNotFoundError(
            f"No golden dataset found for '{pattern}' in {directory}."
            f"Make sure you've created the dataset file first."
        )

    return matches[0]
