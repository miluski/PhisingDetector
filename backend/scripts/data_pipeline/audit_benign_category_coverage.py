"""Validate benign URL category coverage in corpus split CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from utils.benign_url_categories import BENIGN_CATEGORY_PATHS, count_benign_categories


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read split rows from a CSV file."""
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def audit_file(path: Path, minimum: int) -> dict[str, object]:
    """Return category counts and missing-category failures for one split."""
    rows = read_rows(path)
    counts = count_benign_categories(rows)
    missing = [category for category, count in counts.items() if count < minimum]
    return {"path": str(path), "rows": len(rows), "counts": counts, "missing": missing}


def validate(paths: list[Path], minimum: int) -> dict[str, object]:
    """Validate every split and raise when a category is underrepresented."""
    if minimum < 0:
        raise ValueError("minimum must be non-negative")
    report = {"minimum_per_category": minimum, "categories": sorted(BENIGN_CATEGORY_PATHS),
              "splits": [audit_file(path, minimum) for path in paths]}
    failures = {item["path"]: item["missing"] for item in report["splits"] if item["missing"]}
    report["passed"] = not failures
    if failures:
        raise ValueError(json.dumps({"category_coverage_failures": failures}, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    """Parse category coverage audit options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="+", type=Path, required=True)
    parser.add_argument("--minimum-per-category", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the audit, save JSON and report the result."""
    args = parse_args()
    report = validate(args.files, args.minimum_per_category)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
