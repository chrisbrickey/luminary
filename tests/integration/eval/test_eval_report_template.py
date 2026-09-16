"""Integration tests for evaluation report template validation.

These tests verify that the real TEMPLATE.md file exists, contains the required
structure, and is valid markdown. They test the actual file on disk rather than
using test fixtures.
"""

import re
from pathlib import Path

from src.eval.utils import NARRATIVE_SECTIONS, REPORT_SECTIONS
from src.schemas.eval import SystemSnapshot


# Template path (relative to project root)
TEMPLATE_PATH = Path("docs/eval_reports/TEMPLATE.md")

H2_PATTERN = re.compile(r"^## (.+)$", re.MULTILINE)


def _section_body(content: str, section: str) -> str:
    """Return the text between the given H2 header and the next H2 header."""
    sections = content.split("\n## ")
    matches = [block for block in sections if block.startswith(f"{section}\n")]
    assert matches, f"Template is missing the '## {section}' section"
    return matches[0]


def test_template_exists_at_expected_path() -> None:
    """Test that TEMPLATE.md exists at the expected path."""
    assert TEMPLATE_PATH.exists(), (
        f"Template file not found at {TEMPLATE_PATH}. "
        f"Ensure the template has been created before running tests."
    )
    assert TEMPLATE_PATH.is_file(), (
        f"Expected {TEMPLATE_PATH} to be a file, but found a directory."
    )


def test_template_sections_match_shared_constant_in_order() -> None:
    """Test that the template's H2 sections match REPORT_SECTIONS by name and order.

    The script and the template both read section names from this constant, so this
    assertion is what keeps the generated stub aligned with the documented structure.
    """
    template_content = TEMPLATE_PATH.read_text()

    assert tuple(H2_PATTERN.findall(template_content)) == REPORT_SECTIONS


def test_narrative_sections_are_a_subset_of_report_sections() -> None:
    """Test that every section the author completes by hand exists in the template."""
    assert set(NARRATIVE_SECTIONS).issubset(set(REPORT_SECTIONS))


def test_narrative_sections_scaffold_a_subsection_per_item() -> None:
    """Test that each narrative section offers an H3 subsection to repeat per issue or change.

    Real reports carry one subsection per issue and per change, so the template scaffolds
    that shape rather than a flat bullet list.
    """
    template_content = TEMPLATE_PATH.read_text()

    sections_without_subsection = [
        section
        for section in NARRATIVE_SECTIONS
        if "\n### " not in _section_body(template_content, section)
    ]

    assert not sections_without_subsection, (
        f"Narrative sections missing an H3 subsection: {sections_without_subsection}"
    )


def test_template_contains_system_snapshot_bullets() -> None:
    """Test that TEMPLATE.md contains a bullet for every SystemSnapshot field."""
    template_content = TEMPLATE_PATH.read_text()

    missing_bullets = [
        f"- **{info.title}:**"
        for info in SystemSnapshot.model_fields.values()
        if f"- **{info.title}:**" not in template_content
    ]

    assert not missing_bullets, (
        f"Template is missing System Snapshot bullets: {missing_bullets}"
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
