"""Integration tests for golden dataset existence, discovery, and validation.
These tests run for every author listed in AUTHOR_CONFIGS.

These tests verify that:
1. Real golden dataset files exist in the expected location
2. Files can be discovered using the discovery utilities
3. Files conform to the GoldenDataset schema
4. All registered authors have valid golden datasets

These are integration tests because they:
- Read real files from disk (evals/golden/)
- Test the full pipeline: discover → load → validate
- Verify the project's actual golden dataset inventory (gitignored)
"""

import pytest

from src.configs.authors import AUTHOR_CONFIGS
from src.configs.eval import DEFAULT_GOLDEN_DATASET_PATH
from src.eval.utils import discover_latest_golden_dataset, load_golden_dataset
from src.schemas.eval import GoldenDataset


class TestGoldenDatasetIntegration:
    """Integration tests for golden dataset files."""

    @pytest.mark.parametrize("author", AUTHOR_CONFIGS.keys())
    def test_discover_golden_dataset_for_registered_author(self, author: str) -> None:
        """Should find a valid golden dataset file for each registered author.

        This test verifies:
        1. discover_latest_golden_dataset returns a path for the author
        2. The path exists on disk
        3. The path points to the correct directory (DEFAULT_GOLDEN_DATASET_PATH)
        4. The filename matches the expected pattern: persona_{author}_v*.json
        """
        # Act: Discover the latest golden dataset for this author
        dataset_path = discover_latest_golden_dataset(
            directory=DEFAULT_GOLDEN_DATASET_PATH,
            author=author,
            scope="persona",
        )

        # Assert: Path exists
        assert dataset_path.exists(), (
            f"Golden dataset file not found for author '{author}'. "
            f"Expected to find file matching pattern 'persona_{author}_v*.json' "
            f"in {DEFAULT_GOLDEN_DATASET_PATH}"
        )

        # Assert: Path is in the correct directory
        assert dataset_path.parent == DEFAULT_GOLDEN_DATASET_PATH, (
            f"Golden dataset for '{author}' found in wrong directory. "
            f"Expected: {DEFAULT_GOLDEN_DATASET_PATH}, Got: {dataset_path.parent}"
        )

        # Assert: Filename matches expected pattern
        assert dataset_path.name.startswith(f"persona_{author}_v"), (
            f"Golden dataset filename doesn't match pattern. "
            f"Expected: persona_{author}_v*.json, Got: {dataset_path.name}"
        )
        assert dataset_path.suffix == ".json", (
            f"Golden dataset file must be JSON. Got: {dataset_path.suffix}"
        )

    @pytest.mark.parametrize("author", AUTHOR_CONFIGS.keys())
    def test_load_and_validate_golden_dataset_for_registered_author(
        self, author: str
    ) -> None:
        """Should successfully load and validate golden dataset for each author.

        This test verifies:
        1. The file can be discovered
        2. The file can be loaded as valid JSON
        3. The JSON conforms to the GoldenDataset schema (Pydantic validation)
        4. The dataset has the expected structure and non-empty examples
        5. Dataset name matches the expected pattern
        """
        # Arrange: Discover the dataset
        dataset_path = discover_latest_golden_dataset(
            directory=DEFAULT_GOLDEN_DATASET_PATH,
            author=author,
            scope="persona",
        )

        # Act: Load and validate the dataset (Pydantic validation happens here)
        dataset = load_golden_dataset(dataset_path)

        # Assert: Dataset is a valid GoldenDataset instance
        assert isinstance(dataset, GoldenDataset), (
            f"Loaded dataset for '{author}' is not a GoldenDataset instance"
        )

        # Assert: Dataset name contains the author name
        assert author in dataset.name.lower(), (
            f"Dataset name '{dataset.name}' should contain author name '{author}'"
        )

        # Assert: Dataset has required metadata
        assert dataset.version, "Dataset must have a version"
        assert dataset.created_date, "Dataset must have a created_date"
        assert dataset.description, "Dataset must have a description"

        # Assert: Dataset has examples
        assert len(dataset.examples) > 0, (
            f"Golden dataset for '{author}' must have at least one example. "
            f"Got 0 examples."
        )

        # Assert: Each example has required fields
        for example in dataset.examples:
            assert example.id, "Example must have an id"
            assert example.question, "Example must have a question"
            assert example.language, "Example must have a language"
            # expected_chunk_ids can be empty for some metrics, but field must exist
            assert isinstance(example.expected_chunk_ids, list), (
                "Example must have expected_chunk_ids list"
            )

    @pytest.mark.parametrize("author", AUTHOR_CONFIGS.keys())
    def test_golden_dataset_authors_field_matches_examples(self, author: str) -> None:
        """Should verify that dataset.authors aligns with authors in examples.

        This test verifies:
        1. The authors field exists and is non-empty
        2. Every author in dataset.authors appears in at least one example
        3. Every unique author from examples is in dataset.authors
        4. No duplicates in dataset.authors

        The model_validator on GoldenDataset already enforces this,
        but this test explicitly verifies the real golden datasets comply.
        """
        # Arrange: Discover and load the dataset
        dataset_path = discover_latest_golden_dataset(
            directory=DEFAULT_GOLDEN_DATASET_PATH,
            author=author,
            scope="persona",
        )
        dataset = load_golden_dataset(dataset_path)

        # Act: Extract authors from metadata and examples
        metadata_authors = set(dataset.authors)
        example_authors = {ex.author for ex in dataset.examples}

        # Assert: No duplicates in authors field
        assert len(dataset.authors) == len(metadata_authors), (
            f"Duplicate authors in dataset.authors: {dataset.authors}"
        )

        # Assert: Authors field is non-empty
        assert len(metadata_authors) > 0, "dataset.authors must not be empty"

        # Assert: Perfect alignment (model_validator should enforce this)
        assert metadata_authors == example_authors, (
            f"Mismatch in {dataset_path.name}: "
            f"metadata.authors={sorted(metadata_authors)}, "
            f"example authors={sorted(example_authors)}"
        )

        # Assert: For single-author datasets, verify it matches the discovery author
        if len(metadata_authors) == 1:
            assert author in metadata_authors, (
                f"Single-author dataset for '{author}' should contain that author. "
                f"Got: {metadata_authors}"
            )

    def test_all_golden_datasets_discoverable(self) -> None:
        """Should be able to discover golden datasets for all registered authors.

        This test provides a summary view of all golden datasets in the project.
        It's useful for quickly checking the inventory without running all parametrized tests.
        """
        discovered_authors = []
        missing_authors = []

        for author in AUTHOR_CONFIGS.keys():
            try:
                dataset_path = discover_latest_golden_dataset(
                    directory=DEFAULT_GOLDEN_DATASET_PATH,
                    author=author,
                    scope="persona",
                )
                discovered_authors.append((author, dataset_path.name))
            except FileNotFoundError:
                missing_authors.append(author)

        # Assert: All authors have golden datasets
        assert len(missing_authors) == 0, (
            f"Missing golden datasets for authors: {missing_authors}. "
            f"Discovered datasets: {discovered_authors}"
        )

        # Report: Show what was discovered (visible in verbose test output)
        assert len(discovered_authors) == len(AUTHOR_CONFIGS), (
            f"Expected {len(AUTHOR_CONFIGS)} golden datasets, "
            f"found {len(discovered_authors)}: {discovered_authors}"
        )
