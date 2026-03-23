"""Integration test: verify no unused imports in source files."""

import subprocess


def test_no_unused_imports() -> None:
    """Test that autoflake finds no unused imports in src/, scripts/, or tests/.

    This test runs autoflake in check mode across all Python code to ensure:
    - No unused imports
    - No unused variables (optional, can be disabled)

    The test fails if autoflake would make any changes, indicating that
    unused imports exist and should be removed.
    """
    result = subprocess.run(
        [
            "uv",
            "run",
            "autoflake",
            "--check",  # Check mode: exit 1 if changes would be made
            "--recursive",  # Check all files recursively
            "--remove-all-unused-imports",  # Check for unused imports
            "--remove-unused-variables",  # Also check for unused variables
            "--ignore-init-module-imports",  # Allow imports in __init__.py
            "src",
            "scripts",
            "tests",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"autoflake found unused imports or variables:\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}\n\n"
        f"To fix automatically, run:\n"
        f"  uv run autoflake --in-place --recursive "
        f"--remove-all-unused-imports --remove-unused-variables "
        f"--ignore-init-module-imports src scripts tests"
    )
