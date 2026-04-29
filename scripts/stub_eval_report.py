"""CLI script for creating a pre-populated narrative eval report stub.

Reads an eval run artifact and generates a markdown report stub with
auto-populated fields, ready for manual analysis and narrative completion.

Usage:
    uv run python scripts/stub_eval_report.py path/to/eval/artifact

    # Save to a custom directory
    uv run python scripts/stub_eval_report.py path/to/eval/artifact --output-path custom/reports
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from src.eval.utils import format_eval_report_stub

DEFAULT_OUTPUT_PATH = Path("docs/eval_reports")


def main() -> None:
    """Main CLI entry point for creating a narrative eval report stub.

    Workflow:
    - Parse the artifact path from CLI args
    - Call format_eval_report_stub to generate pre-populated markdown
    - Save markdown to docs/eval_reports/eval_report_{timestamp}.md
    - Print the output path on success

    Exit codes:
    - 0: Success
    - 1: Error (artifact not found, invalid schema, I/O failure)
    """
    parser = argparse.ArgumentParser(
        description="Generate a pre-populated narrative eval report stub from an eval run artifact."
    )
    parser.add_argument(
        "artifact_path",
        type=Path,
        help="Path to the eval run JSON artifact (e.g., evals/runs/2026-04-17T12-25-34.json)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Directory to save the report (default: {DEFAULT_OUTPUT_PATH})",
    )
    args = parser.parse_args()

    # Generate markdown stub from artifact
    try:
        markdown = format_eval_report_stub(args.artifact_path)
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Eval artifact not found: {args.artifact_path}", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        sys.exit(1)
    except ValidationError as e:
        print(f"\n❌ ERROR: Artifact at {args.artifact_path} does not match expected schema.", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ ERROR loading artifact {args.artifact_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Persist markdown to output directory
    try:
        args.output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        output_path = args.output_path / f"eval_report_{timestamp}.md"
        output_path.write_text(markdown)
    except (PermissionError, OSError) as e:
        print(f"\n❌ ERROR writing report: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n✅ Eval report stub saved: {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Open {output_path}")
    print(f"  2. Fill out sections: Issue Analysis, Changes Made, Changes Deferred\n")


if __name__ == "__main__":
    main()
