"""Generate test datasets for URL corpus data under ../data/url_corpus/test.

Uses same logic as utils/generate_test_datasets.py, but for
../data/url_corpus/dataset_full.csv and ../data/url_corpus/dataset_optimized.csv.

Run from backend/:
  ../.venv/bin/python scripts/generate_url_corpus_test_datasets.py
"""

import os
import sys
from pathlib import Path

# Make backend root importable when script executed from backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from utils.generate_test_datasets import (  # noqa: E402
    compare_datasets,
    generate_test_dataset,
)


def main():
    data_dir = BACKEND_DIR / "data" / "url_corpus"
    test_dir = data_dir / "test"

    full_csv = data_dir / "dataset_full.csv"
    optimized_csv = data_dir / "dataset_optimized.csv"

    os.makedirs(test_dir, exist_ok=True)

    test_full_csv = test_dir / "test_dataset_full.csv"
    test_optimized_csv = test_dir / "test_dataset_optimized.csv"

    if not full_csv.exists():
        print(f"Błąd: Nie znaleziono pliku {full_csv}")
        return

    if not optimized_csv.exists():
        print(f"Błąd: Nie znaleziono pliku {optimized_csv}")
        return

    test_full_df, original_full_df = generate_test_dataset(
        str(full_csv),
        str(test_full_csv),
        "URL corpus - Pełny zbiór danych (111 cech)",
    )

    test_optimized_df, original_optimized_df = generate_test_dataset(
        str(optimized_csv),
        str(test_optimized_csv),
        "URL corpus - Zoptymalizowany zbiór danych (74 cechy)",
    )

    compare_datasets(test_full_df, original_full_df, "URL corpus - Pełny zbiór danych")
    compare_datasets(test_optimized_df, original_optimized_df, "URL corpus - Zoptymalizowany zbiór danych")

    print("\n✅ Generowanie test datasetów URL corpus zakończone.")
    print(f"   Full test: {test_full_csv}")
    print(f"   Optimized test: {test_optimized_csv}")


if __name__ == "__main__":
    main()
