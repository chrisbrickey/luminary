"""CLI script for embedding documents and storing them in ChromaDB.

This script loads documents from disk, chunks them, embeds the chunks,
and stores them in ChromaDB for retrieval.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.configs.common import DEFAULT_RAW_DIR
from src.configs.loader_configs import INGEST_CONFIGS
from src.configs.vectorstore_config import COLLECTION_NAME
from src.utils.chunker import chunk_documents
from src.utils.io import load_documents_from_disk
from src.utils.ollama_health import check_ollama_available
from src.vectorstores.chroma import embed_and_store

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def embed_author(author: str, input_base_dir: str) -> int:
    """Embed and store documents for a single author.

    Args:
        author: Author key to process
        input_base_dir: Base directory containing scraped documents

    Returns:
        Number of chunks stored

    Raises:
        ValueError: If author is not found in INGEST_CONFIGS
        FileNotFoundError: If author's document directory doesn't exist
        Exception: For any embedding or storage errors
    """
    if author not in INGEST_CONFIGS:
        raise ValueError(
            f"Unknown author: {author}. "
            f"Available authors: {', '.join(INGEST_CONFIGS.keys())}"
        )

    config = INGEST_CONFIGS[author]
    document_dir = Path(input_base_dir) / config.document_id

    logger.info(f"\nConfiguration:")
    logger.info(f"  Author: {author}")
    logger.info(f"  Input: {document_dir}")

    # Load documents from disk
    logger.info(f"\nLoading documents from {document_dir}...")
    documents = load_documents_from_disk(document_dir)
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
        "--input-dir",
        type=str,
        default=str(DEFAULT_RAW_DIR),
        help=f"Base directory containing scraped documents (default: {DEFAULT_RAW_DIR})"
    )

    args = parser.parse_args()

    # Determine which authors to process
    if args.author is None:
        # Process all configured authors
        authors_to_process = list(INGEST_CONFIGS.keys())
        logger.info(f"\n{'='*70}")
        logger.info(f"No author specified - processing all configured authors: {', '.join(authors_to_process)}")
        logger.info(f"{'='*70}")
    else:
        # Process only the specified author
        authors_to_process = [args.author]
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing author: {args.author}")
        logger.info(f"{'='*70}")

    # Check Ollama availability
    try:
        check_ollama_available()
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    try:
        total_chunks = 0
        for author in authors_to_process:
            num_chunks = embed_author(author, args.input_dir)
            total_chunks += num_chunks

        logger.info(f"\n{'='*70}")
        logger.info(f"✓ EMBEDDING COMPLETE")
        logger.info(f"  Total chunks stored: {total_chunks}")
        logger.info(f"{'='*70}\n")

    except ValueError as e:
        # Invalid author name
        logger.error(str(e))
        sys.exit(1)
        return  # For testing when sys.exit is mocked
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
        logger.error(f"Error during embedding: {e}", exc_info=True)
        sys.exit(1)
        return  # For testing when sys.exit is mocked


if __name__ == "__main__":
    main()
