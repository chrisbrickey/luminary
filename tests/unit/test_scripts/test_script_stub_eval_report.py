"""Unit tests for scripts/stub_eval_report.py."""

import re
from pathlib import Path
from unittest.mock import patch

import pytest

FAKE_ARTIFACT_PATH = "evals/runs/2025-05-15T12-25-34.json" # created before triggering eval report
FROZEN_ISO_TIMESTAMP = "2025-05-16T10:32:55"  # ISO 8601 — returned by datetime.now().isoformat()
FROZEN_TIMESTAMP = "2025-05-16T10-32-55"      # filename-safe — returned by format_timestamp(FROZEN_ISO_TIMESTAMP)
FAKE_MARKDOWN_CONTENT = f"# Eval Report {FROZEN_TIMESTAMP}\nsome content"
EXPECTED_FILENAME = f"eval_report_{FROZEN_TIMESTAMP}.md"
EXPECTED_OUTPUT_DIR = Path("docs/eval_reports")


class TestCreateEvalReportStubMain:
    """Tests for main() in scripts/stub_eval_report.py."""

    def test_calls_format_stub_with_artifact_path(self, tmp_path: Path) -> None:
        """Test that main() calls format_eval_report_stub with the artifact path."""
        with (
            patch("scripts.stub_eval_report.format_eval_report_stub", return_value=FAKE_MARKDOWN_CONTENT) as mock_format,
            patch("scripts.stub_eval_report.datetime") as mock_dt,
            patch("sys.argv", ["stub_eval_report.py", FAKE_ARTIFACT_PATH, "--output-path", str(tmp_path)]),
        ):
            mock_dt.now.return_value.isoformat.return_value = FROZEN_ISO_TIMESTAMP
            from scripts.stub_eval_report import main
            main()

        mock_format.assert_called_once_with(Path(FAKE_ARTIFACT_PATH), FROZEN_ISO_TIMESTAMP)

    def test_saves_markdown_to_expected_directory(self, tmp_path: Path) -> None:
        """Test that main() writes the markdown file to the output directory."""
        with (
            patch("scripts.stub_eval_report.format_eval_report_stub", return_value=FAKE_MARKDOWN_CONTENT),
            patch("scripts.stub_eval_report.datetime") as mock_dt,
            patch("sys.argv", ["stub_eval_report.py", FAKE_ARTIFACT_PATH, "--output-path", str(tmp_path)]),
        ):
            mock_dt.now.return_value.isoformat.return_value = FROZEN_ISO_TIMESTAMP
            from scripts.stub_eval_report import main
            main()

        output_file = tmp_path / EXPECTED_FILENAME
        assert output_file.exists(), f"Expected output file at {output_file}"
        assert output_file.read_text() == FAKE_MARKDOWN_CONTENT

    def test_filename_matches_timestamp_pattern(self, tmp_path: Path) -> None:
        """Test that the saved filename follows eval_report_{YYYY-MM-DDTHH-MM-SS}.md pattern."""
        with (
            patch("scripts.stub_eval_report.format_eval_report_stub", return_value=FAKE_MARKDOWN_CONTENT),
            patch("sys.argv", ["stub_eval_report.py", FAKE_ARTIFACT_PATH, "--output-path", str(tmp_path)]),
        ):
            from scripts.stub_eval_report import main
            main()

        files = list(tmp_path.iterdir())
        assert len(files) == 1
        pattern = re.compile(r"^eval_report_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.md$")
        assert pattern.match(files[0].name), f"Unexpected filename: {files[0].name}"

    def test_prints_success_message_with_output_path(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test that main() prints the output path on success."""
        with (
            patch("scripts.stub_eval_report.format_eval_report_stub", return_value=FAKE_MARKDOWN_CONTENT),
            patch("scripts.stub_eval_report.datetime") as mock_dt,
            patch("sys.argv", ["stub_eval_report.py", FAKE_ARTIFACT_PATH, "--output-path", str(tmp_path)]),
        ):
            mock_dt.now.return_value.isoformat.return_value = FROZEN_ISO_TIMESTAMP
            from scripts.stub_eval_report import main
            main()

        captured = capsys.readouterr()
        assert EXPECTED_FILENAME in captured.out

    def test_handles_artifact_not_found(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that main() exits with code 1 when the artifact file does not exist."""
        with (
            patch(
                "scripts.stub_eval_report.format_eval_report_stub",
                side_effect=FileNotFoundError("Eval run artifact not found"),
            ),
            patch("sys.argv", ["stub_eval_report.py", FAKE_ARTIFACT_PATH, "--output-path", str(tmp_path)]),
            pytest.raises(SystemExit) as exc_info,
        ):
            from scripts.stub_eval_report import main
            main()

        captured = capsys.readouterr()
        assert FAKE_ARTIFACT_PATH in captured.err
        assert exc_info.value.code == 1

    def _make_validation_error(self) -> "ValidationError":
        """Create a real ValidationError instance for testing."""
        from pydantic import ValidationError
        from src.schemas.eval import EvalRun

        caught: ValidationError | None = None
        try:
            EvalRun()  # type: ignore[call-arg]
        except ValidationError as e:
            caught = e
        assert caught is not None
        return caught

    def test_handles_validation_error(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that main() exits with code 1 when artifact fails schema validation."""
        with (
            patch(
                "scripts.stub_eval_report.format_eval_report_stub",
                side_effect=self._make_validation_error(),
            ),
            patch("sys.argv", ["stub_eval_report.py", FAKE_ARTIFACT_PATH, "--output-path", str(tmp_path)]),
            pytest.raises(SystemExit) as exc_info,
        ):
            from scripts.stub_eval_report import main
            main()

        captured = capsys.readouterr()
        assert FAKE_ARTIFACT_PATH in captured.err
        assert exc_info.value.code == 1
