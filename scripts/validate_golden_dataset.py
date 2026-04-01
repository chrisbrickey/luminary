#!/usr/bin/env python3
"""Validate that golden dataset JSON file matches GoldenDataset schema."""

import json
import sys
from pathlib import Path

from src.schemas.eval import GoldenDataset


def main() -> None:
    """Load and validate golden dataset."""
    golden_path = Path("data/raw/golden/voltaire_golden_v1.0_2026-04-01.json")

    if not golden_path.exists():
        print(f"ERROR: Golden dataset not found at {golden_path}")
        sys.exit(1)

    print(f"Loading: {golden_path}")

    with golden_path.open() as f:
        data = json.load(f)

    try:
        dataset = GoldenDataset(**data)
        print(f"\n✅ SUCCESS: Golden dataset is valid!")
        print(f"\nDataset info:")
        print(f"  Version: {dataset.version}")
        print(f"  Created: {dataset.created_date}")
        print(f"  Description: {dataset.description}")
        print(f"  Examples: {len(dataset.examples)}")

        print(f"\nExample breakdown:")
        for example in dataset.examples:
            chunk_count = len(example.expected_chunk_ids)
            print(f"  - {example.id:30s} ({example.language}) → {chunk_count} chunks")

    except Exception as e:
        print(f"\n❌ VALIDATION ERROR:")
        print(f"{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
