"""CLI script for embedding documents and storing them in ChromaDB.

This script loads documents from disk, chunks them, embeds the chunks,
and stores them in ChromaDB for retrieval.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.configs.common import RAW_DATA_PATH
from src.configs.loader_configs import INGEST_CONFIGS
from src.configs.vectorstore_config import COLLECTION_NAME
from src.utils.chunker import chunk_documents
from src.utils.cli_helpers import check_ollama_or_exit, exit_on_error, resolve_authors, validate_author
from src.utils.io import load_documents_from_disk
from src.utils.logging import setup_cli_logging
from src.vectorstores.chroma import embed_and_store


def embed_author(author: str, input_base_path: str, logger: logging.Logger) -> int:
    """Embed and store documents for a single author.

    Args:
        author: Author key to process
        input_base_path: Base directory containing scraped documents
        logger: Logger instance for output

    Returns:
        Number of chunks stored

    Raises:
        ValueError: If author is not found in INGEST_CONFIGS
        FileNotFoundError: If author's document directory doesn't exist
        Exception: For any embedding or storage errors
    """
    validate_author(author)
    config = INGEST_CONFIGS[author]
    document_path = Path(input_base_path) / config.document_id

    logger.info(f"\nConfiguration:")
    logger.info(f"  Author: {author}")
    logger.info(f"  Input: {document_path}")

    # Load documents from disk
    logger.info(f"\nLoading documents from {document_path}...")
    documents = load_documents_from_disk(document_path)
    logger.info(f"✓ Loaded {len(documents)} documents")

    # Chunk documents
    logger.info(f"\nChunking documents...")
    chunks = chunk_documents(documents)
    logger.info(f"✓ Created {len(chunks)} chunks")

    # Embed and store
    logger.info(f"\nEmbedding chunks and storing in ChromaDB...")
    logger.info(f"  (This may take a few minutes)")
    embed_and_store(
        chunks=chunks,
        collection_name=COLLECTION_NAME
    )
    logger.info(f"✓ Stored {len(chunks)} chunks in ChromaDB")

    return len(chunks)


def main() -> None:
    """Main entry point for embedding and storage script."""
    parser = argparse.ArgumentParser(
        description="Load documents, chunk, embed, and store in ChromaDB"
    )
    parser.add_argument(
        "--author",
        type=str,
        default=None,
        help=f"Author key to process (optional, defaults to all). Available: {', '.join(INGEST_CONFIGS.keys())}"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help=f"Base directory containing scraped documents (default: {RAW_DATA_PATH})"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Evaluate default arguments at runtime, avoiding variable setting at import time
    if args.input_path is None:
        args.input_path = str(RAW_DATA_PATH)

    # Setup logging
    logger = setup_cli_logging(verbose=args.verbose)
    if args.verbose:
        logger.debug("Verbose logging enabled")

    # Determine which authors to process
    logger.info(f"\n{'='*70}")
    authors_to_process = resolve_authors(args.author, logger)
    logger.info(f"{'='*70}")

    # Check Ollama availability
    check_ollama_or_exit(logger)

    try:
        total_chunks = 0
        for author in authors_to_process:
            num_chunks = embed_author(author, args.input_path, logger)
            total_chunks += num_chunks

        logger.info(f"\n{'='*70}")
        logger.info(f"✓ EMBEDDING COMPLETE")
        logger.info(f"  Total chunks stored: {total_chunks}")
        logger.info(f"{'='*70}\n")

    except ValueError as e:
        # Invalid author name
        exit_on_error(logger, e)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        if args.author:
            logger.error(
                f"Run 'python scripts/scrape_wikisource.py --author {args.author}' first"
            )
        else:
            logger.error(
                f"Run 'python scripts/scrape_wikisource.py' first to scrape all authors"
            )
        sys.exit(1)
        return  # For testing when sys.exit is mocked
    except Exception as e:
        exit_on_error(logger, e, context="during embedding")


if __name__ == "__main__":
    main()
