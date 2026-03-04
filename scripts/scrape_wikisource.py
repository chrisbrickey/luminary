"""CLI script for scraping documents from Wikisource.

This script loads a document collection from Wikisource using the configured
loader and saves the documents to disk as JSON files.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.configs.loader_configs import INGEST_CONFIGS
from src.document_loaders.wikisource_loader import WikisourceLoader
from src.utils.io import save_documents_to_disk

# Configure logging with simpler format for better readability
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simple format for user-friendly progress messages
)
logger = logging.getLogger(__name__)


def scrape_author(author: str, output_base_dir: str) -> int:
    """Scrape documents for a single author.

    Args:
        author: Author key to scrape
        output_base_dir: Base directory for saving documents

    Returns:
        Number of documents saved

    Raises:
        ValueError: If author is not found in INGEST_CONFIGS
        Exception: For any scraping or saving errors
    """
    if author not in INGEST_CONFIGS:
        raise ValueError(
            f"Unknown author: {author}. "
            f"Available authors: {', '.join(INGEST_CONFIGS.keys())}"
        )

    config = INGEST_CONFIGS[author]
    logger.info(f"\nConfiguration:")
    logger.info(f"  Author: {author}")
    logger.info(f"  Document: {config.document_title}")
    logger.info(f"  Output: {Path(output_base_dir) / config.document_id}")

    # Create output directory path
    output_dir = Path(output_base_dir) / config.document_id

    # Initialize loader
    loader = WikisourceLoader(config)
    logger.info(f"\nStarting to fetch documents from Wikisource...")

    # Load documents
    documents = loader.load()

    if not documents:
        logger.warning(f"\n⚠ No documents were loaded for {author}")
        return 0

    logger.info(f"\nSaving {len(documents)} documents to disk...")
    saved_paths = save_documents_to_disk(documents, output_dir)

    logger.info(f"✓ Saved {len(saved_paths)} files to {output_dir}\n")
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
        "--output-dir",
        type=str,
        default="data/raw",
        help="Base directory for saving scraped documents (default: data/raw)"
    )

    args = parser.parse_args()

    # Determine which authors to scrape
    if args.author is None:
        # Scrape all configured authors
        authors_to_scrape = list(INGEST_CONFIGS.keys())
        logger.info(f"\n{'='*70}")
        logger.info(f"No author specified - scraping all configured authors: {', '.join(authors_to_scrape)}")
        logger.info(f"{'='*70}")
    else:
        # Scrape only the specified author
        authors_to_scrape = [args.author]
        logger.info(f"\n{'='*70}")
        logger.info(f"Scraping author: {args.author}")
        logger.info(f"{'='*70}")

    try:
        total_saved = 0
        for author in authors_to_scrape:
            num_saved = scrape_author(author, args.output_dir)
            total_saved += num_saved

        logger.info(f"\n{'='*70}")
        logger.info(f"✓ SCRAPING COMPLETE")
        logger.info(f"  Total documents saved: {total_saved}")
        logger.info(f"{'='*70}\n")

    except ValueError as e:
        # Invalid author name
        logger.error(str(e))
        sys.exit(1)
        return  # For testing when sys.exit is mocked
    except Exception as e:
        logger.error(f"Error during scraping: {e}", exc_info=True)
        sys.exit(1)
        return  # For testing when sys.exit is mocked


if __name__ == "__main__":
    main()
