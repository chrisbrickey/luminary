"""CLI script for combined document ingestion (scrape + embed).

This script combines scraping from Wikisource and embedding into ChromaDB into a
single command by calling the existing scripts. Individual steps can be skipped
with --skip-scrape or --skip-embed.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from src.configs.common import DEFAULT_DB_PATH, DEFAULT_RAW_DIR
from src.configs.loader_configs import INGEST_CONFIGS
from src.utils.ollama_health import check_ollama_available

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Constants
_SECTION_WIDTH = 70
_SECTION_CHAR = '='
_PHASE_CHAR = '─'


def _log_section_header(title: str, subtitle: str = "") -> None:
    """Log a section header with equal signs separator.

    Args:
        title: Main title to display
        subtitle: Optional subtitle to display below title
    """
    logger.info(f"\n{_SECTION_CHAR * _SECTION_WIDTH}")
    logger.info(title)
    if subtitle:
        logger.info(subtitle)
    logger.info(_SECTION_CHAR * _SECTION_WIDTH)


def _log_phase_header(phase_name: str) -> None:
    """Log a phase header with dash separator.

    Args:
        phase_name: Name of the phase to display
    """
    logger.info(f"\n{_PHASE_CHAR * _SECTION_WIDTH}")
    logger.info(phase_name)
    logger.info(_PHASE_CHAR * _SECTION_WIDTH)


def _run_scrape_phase(author: str, raw_dir: str) -> None:
    """Run the scraping phase for a single author.

    Args:
        author: Author key to process
        raw_dir: Base directory for saving scraped documents

    Raises:
        subprocess.CalledProcessError: If scraping script fails
    """
    _log_phase_header("PHASE 1: SCRAPING")

    scrape_cmd = [
        "uv", "run", "python", "scripts/scrape_wikisource.py",
        "--author", author,
        "--output-dir", raw_dir
    ]
    subprocess.run(scrape_cmd, check=True)


def _run_embed_phase(author: str, raw_dir: str, db_dir: str) -> None:
    """Run the embedding phase for a single author.

    Args:
        author: Author key to process
        raw_dir: Base directory containing scraped documents
        db_dir: ChromaDB persist directory

    Raises:
        subprocess.CalledProcessError: If embedding script fails
    """
    _log_phase_header("PHASE 2: EMBEDDING")

    embed_cmd = [
        "uv", "run", "python", "scripts/embed_and_store.py",
        "--author", author,
        "--input-dir", raw_dir,
        "--db", db_dir
    ]
    subprocess.run(embed_cmd, check=True)


def ingest_author(
    author: str,
    raw_dir: str,
    db_dir: str,
    skip_scrape: bool,
    skip_embed: bool
) -> None:
    """Ingest documents for a single author (scrape + embed).

    Args:
        author: Author key to process
        raw_dir: Base directory for saving scraped documents
        db_dir: ChromaDB persist directory
        skip_scrape: If True, skip scraping step
        skip_embed: If True, skip embedding step

    Raises:
        ValueError: If author is not found in INGEST_CONFIGS
        subprocess.CalledProcessError: If either script fails
    """
    if author not in INGEST_CONFIGS:
        raise ValueError(
            f"Unknown author: {author}. "
            f"Available authors: {', '.join(INGEST_CONFIGS.keys())}"
        )

    # Scraping phase
    if not skip_scrape:
        _run_scrape_phase(author, raw_dir)
    else:
        logger.info(f"\n⏭  Skipping scrape phase for {author}")

    # Embedding phase
    if not skip_embed:
        _run_embed_phase(author, raw_dir, db_dir)
    else:
        logger.info(f"\n⏭  Skipping embed phase for {author}")


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build and configure the argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Combined ingestion: scrape from Wikisource and embed into ChromaDB"
    )
    parser.add_argument(
        "--author",
        type=str,
        default=None,
        help=f"Author key to process (optional, defaults to all). Available: {', '.join(INGEST_CONFIGS.keys())}"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(DEFAULT_RAW_DIR),
        help=f"Base directory for scraped documents (default: {DEFAULT_RAW_DIR})"
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help=f"ChromaDB persist directory (default: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip scraping phase (use existing scraped documents)"
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Skip embedding phase (only scrape documents)"
    )
    return parser


def main() -> None:
    """Main entry point for combined ingestion script."""
    parser = _build_argument_parser()
    args = parser.parse_args()

    # Validate flags
    if args.skip_scrape and args.skip_embed:
        logger.error("Error: Cannot skip both scrape and embed phases")
        sys.exit(1)
        return  # For testing when sys.exit is mocked

    # Determine which authors to process
    if args.author is None:
        authors_to_process = list(INGEST_CONFIGS.keys())
        _log_section_header(
            "COMBINED INGESTION - All Authors",
            f"Authors: {', '.join(authors_to_process)}"
        )
    else:
        authors_to_process = [args.author]
        _log_section_header("COMBINED INGESTION", f"Author: {args.author}")

    # Check Ollama availability (only needed for embedding)
    if not args.skip_embed:
        try:
            check_ollama_available()
        except RuntimeError as e:
            logger.error(str(e))
            sys.exit(1)
            return  # For testing when sys.exit is mocked

    try:
        for author in authors_to_process:
            ingest_author(
                author=author,
                raw_dir=args.raw_dir,
                db_dir=args.db,
                skip_scrape=args.skip_scrape,
                skip_embed=args.skip_embed
            )

        # Final summary
        _log_section_header("✓ INGESTION COMPLETE", f"  Database location: {args.db}")
        logger.info("")  # Add trailing newline

    except ValueError as e:
        # Invalid author name
        logger.error(str(e))
        sys.exit(1)
        return  # For testing when sys.exit is mocked
    except subprocess.CalledProcessError as e:
        # Script execution failed
        logger.error(f"Error: Script execution failed with exit code {e.returncode}")
        sys.exit(1)
        return  # For testing when sys.exit is mocked
    except Exception as e:
        logger.error(f"Error during ingestion: {e}", exc_info=True)
        sys.exit(1)
        return  # For testing when sys.exit is mocked


if __name__ == "__main__":
    main()
