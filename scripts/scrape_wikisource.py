"""CLI script for scraping documents from Wikisource.

This script loads a document collection from Wikisource using the configured
loader and saves the documents to disk as JSON files.
"""

import argparse
import logging
from pathlib import Path

from src.configs.common import RAW_DATA_PATH
from src.configs.loader_configs import INGEST_CONFIGS
from src.document_loaders.wikisource_loader import WikisourceLoader
from src.utils.cli_helpers import exit_on_error, resolve_authors, validate_author
from src.utils.io import save_documents_to_disk
from src.utils.logging import setup_cli_logging


def scrape_author(author: str, output_base_path: str, logger: logging.Logger) -> int:
    """Scrape documents for a single author.

    Args:
        author: Author key to scrape
        output_base_path: Base directory for saving documents
        logger: Logger instance for output

    Returns:
        Number of documents saved

    Raises:
        ValueError: If author is not found in INGEST_CONFIGS
        Exception: For any scraping or saving errors
    """
    validate_author(author)
    config = INGEST_CONFIGS[author]
    logger.info(f"\nConfiguration:")
    logger.info(f"  Author: {author}")
    logger.info(f"  Document: {config.document_title}")
    logger.info(f"  Output: {Path(output_base_path) / config.document_id}")

    # Create output path
    output_path = Path(output_base_path) / config.document_id

    # Initialize loader
    loader = WikisourceLoader(config)
    logger.info(f"\nStarting to fetch documents from Wikisource...")

    # Load documents
    documents = loader.load()

    if not documents:
        logger.warning(f"\n⚠ No documents were loaded for {author}")
        return 0

    logger.info(f"\nSaving {len(documents)} documents to disk...")
    saved_paths = save_documents_to_disk(documents, output_path)

    logger.info(f"✓ Saved {len(saved_paths)} files to {output_path}\n")
    return len(saved_paths)


def main() -> None:
    """Main entry point for the scraper script."""
    parser = argparse.ArgumentParser(
        description="Scrape documents from Wikisource for authors"
    )
    parser.add_argument(
        "--author",
        type=str,
        default=None,
        help=f"Author key to scrape (optional, defaults to all). Available: {', '.join(INGEST_CONFIGS.keys())}"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(RAW_DATA_PATH),
        help=f"Base directory for saving scraped documents (default: {RAW_DATA_PATH})"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_cli_logging(verbose=args.verbose)

    if args.verbose:
        logger.debug("Verbose logging enabled")

    # Determine which authors to scrape
    logger.info(f"\n{'='*70}")
    authors_to_scrape = resolve_authors(args.author, logger)
    logger.info(f"{'='*70}")

    try:
        total_saved = 0
        for author in authors_to_scrape:
            num_saved = scrape_author(author, args.output_path, logger)
            total_saved += num_saved

        logger.info(f"\n{'='*70}")
        logger.info(f"✓ SCRAPING COMPLETE")
        logger.info(f"  Total documents saved: {total_saved}")
        logger.info(f"{'='*70}\n")

    except ValueError as e:
        # Invalid author name
        exit_on_error(logger, e)
    except Exception as e:
        exit_on_error(logger, e, context="during scraping")


if __name__ == "__main__":
    main()
