"""Integration test: verify mypy passes on all source files."""

import subprocess


def test_mypy_passes_on_src_and_scripts() -> None:
    result = subprocess.run(
        ["uv", "run", "mypy", "src"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"mypy failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )