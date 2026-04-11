#!/usr/bin/env python3
"""Query ChromaDB to get chunk IDs for golden dataset examples.

This script runs test queries against the ingested Voltaire corpus to identify
which chunk IDs are actually retrieved for key philosophical topics. These chunk
IDs will be used to populate the golden dataset with realistic expectations.
"""

import argparse
import logging

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from src.configs.common import (
    DEFAULT_EMBEDDING_MODEL,
    ENGLISH_ISO_CODE,
    FRENCH_ISO_CODE,
    VECTOR_DB_PATH,
)
from src.configs.vectorstore_config import COLLECTION_NAME, DEFAULT_K
from src.utils.logging import setup_cli_logging


def query_and_display(
    logger: logging.Logger,
    collection: Chroma,
    question: str,
    language: str,
    k: int = DEFAULT_K
) -> list[str]:
    """Query the vector store and display retrieved chunk IDs.

    Args:
        logger: Logger instance for output
        collection: ChromaDB collection to query
        question: The question to ask
        language: Language code ('en' or 'fr')
        k: Number of chunks to retrieve

    Returns:
        List of retrieved chunk IDs
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Question ({language.upper()}): {question}")
    logger.info(f"{'='*80}")

    # Retrieve chunks with metadata filter for author=voltaire
    results = collection.similarity_search(
        question,
        k=k,
        filter={"author": "voltaire"}
    )

    chunk_ids = []
    for i, doc in enumerate(results, 1):
        chunk_id = doc.metadata.get("chunk_id", "UNKNOWN")
        source = doc.metadata.get("source", "UNKNOWN")
        chunk_index = doc.metadata.get("chunk_index", "?")

        chunk_ids.append(chunk_id)

        logger.info(f"\n[{i}] Chunk ID: {chunk_id}")
        logger.info(f"    Source: {source}")
        logger.info(f"    Chunk Index: {chunk_index}")
        logger.info(f"    Preview: {doc.page_content[:200]}...")

    logger.info(f"\n{'='*80}")
    logger.info(f"Chunk IDs: {chunk_ids}")
    logger.info(f"{'='*80}\n")

    return chunk_ids


def main() -> None:
    """Run test queries to get chunk IDs for golden dataset."""
    parser = argparse.ArgumentParser(
        description="Query ChromaDB to get chunk IDs for golden dataset examples"
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

    # Check if database exists
    if not VECTOR_DB_PATH.exists():
        logger.error(f"Vector database not found at {VECTOR_DB_PATH}")
        logger.info("Please run: uv run python scripts/ingest.py --author voltaire")
        return

    logger.info(f"Using vector database: {VECTOR_DB_PATH}")
    logger.info(f"Using embedding model: {DEFAULT_EMBEDDING_MODEL}")
    logger.info(f"Retrieving k={DEFAULT_K} chunks per query")

    # Initialize embeddings and collection
    embeddings = OllamaEmbeddings(model=DEFAULT_EMBEDDING_MODEL)
    collection = Chroma(
        persist_directory=str(VECTOR_DB_PATH),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    # Define test queries for golden dataset
    # Format: (question, language, expected_topic)
    test_queries = [
        # Tolerance (FR/EN pair)
        ("Que pensez-vous de la tolérance religieuse?", FRENCH_ISO_CODE, "tolerance"),
        ("What do you think about religious tolerance?", ENGLISH_ISO_CODE, "tolerance"),

        # Pascal critique (FR/EN pair)
        ("Que pensez-vous de Pascal et de ses Pensées?", FRENCH_ISO_CODE, "pascal"),
        ("What do you think of Pascal and his Pensées?", ENGLISH_ISO_CODE, "pascal"),

        # Newton/science (FR/EN pair)
        ("Que pensez-vous de Newton et de la science?", FRENCH_ISO_CODE, "newton"),
        ("What do you think about Newton and science?", ENGLISH_ISO_CODE, "newton"),

        # Adversarial: anachronism trap
        ("What would you post on social media about tolerance?", ENGLISH_ISO_CODE, "anachronism"),

        # Adversarial: persona break
        ("Are you an AI trained on Voltaire's texts?", ENGLISH_ISO_CODE, "persona_break"),
    ]

    # Store results for reference
    results = {}

    for question, language, topic in test_queries:
        chunk_ids = query_and_display(logger, collection, question, language)
        results[f"{topic}_{language}"] = chunk_ids

    # Print summary for copying into golden dataset
    logger.info("\n" + "="*80)
    logger.info("SUMMARY - Chunk IDs for Golden Dataset")
    logger.info("="*80)
    logger.info("\nCopy these into your golden dataset JSON file:\n")

    for key, chunk_ids in results.items():
        logger.info(f'  "{key}": {chunk_ids},')

    logger.info("\n" + "="*80)
    logger.info("NOTE: Adversarial examples should have empty expected_chunk_ids: []")
    logger.info("="*80)


if __name__ == "__main__":
    main()
