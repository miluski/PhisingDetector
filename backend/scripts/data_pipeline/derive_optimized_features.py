"""Derive optimized feature CSVs from already extracted full CSVs.

Optimized schema is a strict column subset of full schema. Reusing full rows
prevents a second live DNS/WHOIS/HTTP/SSL pass and guarantees identical
network observations and missing indicators in both feature variants.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config.feature_names_config import FeatureConfig
from config.feature_policy import AUDIT_INDICATORS


METADATA_COLUMNS = ("source_index", "url", *AUDIT_INDICATORS)
LABEL_COLUMN = "phishing"
SPLITS = ("train", "val", "test")


def derive(input_path: Path, output_path: Path, mode: str = "optimized") -> int:
    """Project a full feature CSV onto an optimized schema."""
    columns = [*METADATA_COLUMNS, *FeatureConfig.get_feature_names(mode), LABEL_COLUMN]
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _project_rows(input_path, temporary, columns)
    temporary.replace(output_path)
    return rows


def _project_rows(input_path, temporary, columns):
    """Project rows to selected columns and return the count."""
    rows = 0
    with input_path.open(newline="", encoding="utf-8") as source, temporary.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source)
        available = set(reader.fieldnames or [])
        missing = sorted(set(columns) - available)
        if missing:
            raise ValueError(f"{input_path} lacks required full-schema columns: {missing}")
        if "asn_ip" in available:
            raise ValueError(f"Legacy asn_ip found in full schema: {input_path}")
        writer = csv.DictWriter(target, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in reader:
            writer.writerow({column: row[column] for column in columns})
            rows += 1
    return rows


def parse_args() -> argparse.Namespace:
    """Parse optimized feature derivation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Derive optimized files for all predefined splits."""
    args = parse_args()
    for split in SPLITS:
        full_path = args.full_dir / f"dataset_full_{split}.csv"
        full_count = derive(full_path, full_path, mode="full")
        optimized_count = derive(
            full_path,
            args.output_dir / f"dataset_optimized_{split}.csv",
            mode="optimized",
        )
        print(f"{split}: full={full_count:,}, optimized={optimized_count:,} rows")


if __name__ == "__main__":
    main()
