"""Integration tests for evaluation report template validation.

These tests verify that the real TEMPLATE.md file exists, contains the required
structure, and is valid markdown. They test the actual file on disk rather than
using test fixtures.
"""

import re
from pathlib import Path


# Template path (relative to project root)
TEMPLATE_PATH = Path("docs/eval_reports/TEMPLATE.md")

# Required section headers that must be present in the template
REQUIRED_SECTIONS = [
    "# Eval Report",
    "## Source Data",
    "## System Version",
    "## Eval Run Summary",
    "## Issue Analysis",
    "## Changes Made",
    "## Changes Deferred",
]


def test_template_exists_at_expected_path() -> None:
    """Test that TEMPLATE.md exists at the expected path."""
    assert TEMPLATE_PATH.exists(), (
        f"Template file not found at {TEMPLATE_PATH}. "
        f"Ensure the template has been created before running tests."
    )
    assert TEMPLATE_PATH.is_file(), (
        f"Expected {TEMPLATE_PATH} to be a file, but found a directory."
    )


def test_template_contains_required_sections() -> None:
    """Test that TEMPLATE.md contains all required section headers."""
    # Read the template file
    template_content = TEMPLATE_PATH.read_text()

    # Verify each required section is present
    missing_sections = []
    for section in REQUIRED_SECTIONS:
        if section not in template_content:
            missing_sections.append(section)

    assert not missing_sections, (
        f"Template is missing {missing_sections} of the required sections: {REQUIRED_SECTIONS}."
    )


def test_template_structure_is_valid_markdown() -> None:
    """Test that TEMPLATE.md has valid markdown structure."""
    template_content = TEMPLATE_PATH.read_text()
    lines = template_content.split("\n")

    # Verify no malformed headers (e.g., "##Section" without space)
    malformed_headers = [
        line for line in lines
        if re.match(r"^#{1,6}[^#\s]", line)  # Hash followed by non-space, non-hash
    ]

    assert not malformed_headers, (
        f"Template contains malformed headers (missing space after #): {malformed_headers}"
    )

    # Verify template is not empty
    assert len(template_content.strip()) > 100, "Template file should not be empty"
