"""Export predefined URL corpus test datasets.

This script intentionally does not sample from a single CSV and does not create
synthetic records. URL corpus evaluation must use the already separated
train/val/test files generated under data/url_corpus/quality_v4_repaired_features.

Run from repository root:
  ./.venv/bin/python backend/scripts/generate_url_corpus_test_datasets.py
"""

from __future__ import annotations

from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from utils.split_dataset_loader import load_split_frames


def main() -> None:
    """Export the predefined URL-corpus test split."""
    split_dir = BACKEND_DIR / "data" / "url_corpus" / "quality_v4_repaired_features"
    output_dir = BACKEND_DIR / "data" / "url_corpus" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_split_frames(split_dir, "dataset_full")
    test_df = frames["test"]
    output_path = output_dir / "test_dataset_full.csv"
    test_df.to_csv(output_path, index=False)

    counts = test_df["phishing"].value_counts().to_dict()
    print("✅ Exported predefined URL corpus test split.")
    print(f"   Source split dir: {split_dir}")
    print(f"   Output: {output_path}")
    print(f"   Rows: {len(test_df):,}")
    print(f"   Safe: {counts.get(0, 0):,}")
    print(f"   Phishing: {counts.get(1, 0):,}")


if __name__ == "__main__":
    main()
