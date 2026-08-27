"""Merge corrected feature shards, derive optimized columns, and audit coverage."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from backend.scripts.data_pipeline.merge_dataset_shards import merge
except ImportError:  
    from merge_dataset_shards import merge


SPLIT_NAMES = ("train", "val", "test")


def read_header(path: Path) -> list[str]:
    """Read the first CSV header row."""
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        return next(reader)


def derive_optimized(full_path: Path, output_path: Path, columns: list[str]) -> int:
    """Write selected optimized columns from a full feature CSV."""
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    count = 0
    with full_path.open(newline="", encoding="utf-8") as source, tmp_path.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source)
        missing = [column for column in columns if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing optimized columns in {full_path}: {missing}")
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        for row in reader:
            writer.writerow({column: row[column] for column in columns})
            count += 1
    tmp_path.replace(output_path)
    return count


def key_set(path: Path, label_column: str) -> set[tuple[str, str]]:
    """Return normalized URL/label keys from a feature or raw CSV."""
    with path.open(newline="", encoding="utf-8") as file:
        return {
            ((row.get("url") or "").strip().lower(), (row.get(label_column) or "").strip())
            for row in csv.DictReader(file)
        }


def finalize(
    raw_split_dir: Path,
    feature_dir: Path,
    optimized_reference: Path,
) -> dict[str, dict[str, int]]:
    """Merge shards, derive optimized files and verify split coverage."""
    optimized_columns = read_header(optimized_reference)
    report = {split: _finalize_split(raw_split_dir, feature_dir, optimized_columns, split)
              for split in SPLIT_NAMES}

    with (feature_dir / "finalize_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
    return report


def _finalize_split(raw_split_dir, feature_dir, optimized_columns, split):
    """Merge, project and validate one corrected split."""
    reused = feature_dir / f"dataset_full_{split}_reused.csv"
    extracted = sorted(feature_dir.glob(f"{split}_missing_shards/dataset_full_{split}_missing_shard_*.csv"))
    if not extracted:
        raise ValueError(f"No extracted feature shards for {split}")
    full_output = feature_dir / f"dataset_full_{split}.csv"
    full_rows = merge([reused, *extracted], full_output)
    optimized_path = feature_dir / f"dataset_optimized_{split}.csv"
    optimized_rows = derive_optimized(full_output, optimized_path, optimized_columns)
    raw_keys = key_set(raw_split_dir / f"raw_{split}.csv", "label")
    feature_keys = key_set(full_output, "phishing")
    missing, extra = len(raw_keys - feature_keys), len(feature_keys - raw_keys)
    if missing or extra or full_rows != len(raw_keys):
        raise ValueError(
            f"Coverage mismatch for {split}: rows={full_rows}, raw={len(raw_keys)}, "
            f"missing={missing}, extra={extra}"
        )
    return {"raw_rows": len(raw_keys), "full_rows": full_rows, "optimized_rows": optimized_rows,
            "missing": missing, "extra": extra}


def parse_args() -> argparse.Namespace:
    """Parse corrected feature finalization options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-split-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--optimized-reference", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Finalize corrected features and print the coverage report."""
    args = parse_args()
    report = finalize(args.raw_split_dir, args.feature_dir, args.optimized_reference)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
