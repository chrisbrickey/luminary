"""Utility functions for evaluation harness."""

import json
from datetime import datetime
from pathlib import Path

from src.configs.authors import DEFAULT_AUTHOR
from src.schemas.eval import GoldenDataset, EvalRun


def load_golden_dataset(path: Path) -> GoldenDataset:
    """Load and validate golden dataset from specified JSON file.

    Intended to be used in conjunction with discover_latest_golden_dataset,
    which returns the path to the most recent of versioned datasets.

    Args:
        path: Path to one specific golden dataset JSON file

    Returns:
        Validated GoldenDataset object

    Raises:
        IsADirectoryError: If path points to a directory, not a file
        FileNotFoundError: If path does not exist
        PermissionError: If file cannot be read due to permissions
        json.JSONDecodeError: If file is not valid JSON
        ValidationError: If JSON doesn't match GoldenDataset schema (Pydantic)
    """
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {path}\n")

    try:
        # Load JSON data
        with path.open() as f:
            data = json.load(f)

    except PermissionError as e:
        raise PermissionError(
            f"Permission denied: Cannot read file '{path}'. Check file permissions and retry."
        ) from e
    except IsADirectoryError as e:
        raise IsADirectoryError(
            f"Expected file but found directory: '{path}'"
        ) from e
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in '{path}'; {e.msg}",
            e.doc,
            e.pos
        ) from e

    # Pydantic validation (let ValidationError bubble up with its detailed messages)
    return GoldenDataset(**data)


def discover_latest_golden_dataset(
    directory: Path,
    scope: str | None = "persona",
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
        scope: Dataset scope (default: "persona")
        authors: List of author names default [DEFAULT_AUTHOR] : e.g., ["condorcet", "gouges"]
                 The list will be sorted and joined with underscores to match the authors portion of the filename.

    Returns:
        Path to newest matching file (sorted by filename, newest first)

    Raises:
        FileNotFoundError: If no matching files found (include pattern and directory)
    """
    # Fallback to default attributes of golden dataset if not provided
    scope = "persona" if scope is None else scope
    authors = [DEFAULT_AUTHOR] if authors is None else authors

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

def save_eval_run(eval_run: EvalRun, output_dir: Path) -> Path:
    """Save EvalRun to timestamped JSON file.

    Filename format: {YYYY-MM-DD}T{HH-MM-SS}.json
    Example: 2026-03-28T14-30-45.json

    Args:
        eval_run: EvalRun object to save
        output_dir: Directory to save to (created if doesn't exist)

    Returns:
        Path to saved file

    Raises:
        PermissionError: If cannot create directory or write file
        OSError: If disk is full or other OS-level error
        TypeError: If eval_run contains non-JSON-serializable data
    """

    # Create output directory if it doesn't exist
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(
            f"Permission denied: Cannot create directory '{output_dir}'. "
            f"Check file permissions and try again."
        ) from e
    except OSError as e:
        raise OSError(
            f"Failed to create directory '{output_dir}': {e}"
        ) from e

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"{timestamp}.json"
    filepath = output_dir / filename

    # Save EvalRun as JSON
    try:
        with filepath.open("w") as f:
            json.dump(eval_run.model_dump(), f, indent=2)
    except PermissionError as e:
        raise PermissionError(
            f"Permission denied: Cannot write to file '{filepath}'. Check file permissions and retry."
        ) from e
    except OSError as e:
        raise OSError(
            f"Failed to write file '{filepath}': {e}. Check available disk space."
        ) from e
    except TypeError as e:
        raise TypeError(
            f"Failed to serialize EvalRun to JSON: {e}. EvalRun contains non-serializable data."
        ) from e

    return filepath
