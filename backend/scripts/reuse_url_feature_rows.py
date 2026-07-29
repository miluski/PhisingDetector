"""Reuse previously extracted URL feature rows for new domain-safe raw splits."""

from __future__ import annotations

import argparse
import csv
import json
from contextlib import ExitStack
from pathlib import Path


SPLIT_NAMES = ("train", "val", "test")


def url_key(value: str) -> str:
    """Return a normalized URL key for feature-row matching."""
    return (value or "").strip().lower()


def reuse(
    raw_split_dir: Path,
    existing_features: list[Path],
    output_dir: Path,
) -> dict[str, dict[str, int]]:
    """Reuse existing feature rows for corrected raw split assignments."""
    targets, headers, raw_rows = _load_targets(raw_split_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    found = _copy_feature_rows(existing_features, targets, output_dir)
    report = _write_missing_reports(output_dir, headers, raw_rows, found)
    with (output_dir / "reuse_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
    return report


def _load_targets(raw_split_dir):
    """Load raw split assignments keyed by normalized URL."""
    targets, headers, rows_by_split = {}, {}, {}
    for split in SPLIT_NAMES:
        path = raw_split_dir / f"raw_{split}.csv"
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise ValueError(f"{path} has no header")
            rows = list(reader)
        headers[split], rows_by_split[split] = reader.fieldnames, rows
        for row in rows:
            key = url_key(row.get("url", ""))
            if key in targets:
                raise ValueError(f"Duplicate URL across corrected splits: {row.get('url')}")
            targets[key] = (split, row)
    return targets, headers, rows_by_split


def _copy_feature_rows(paths, targets, output_dir):
    """Copy matching feature rows into split-specific outputs."""
    found, header = set(), None
    with ExitStack() as stack:
        files = {
            split: stack.enter_context(
                (output_dir / f"dataset_full_{split}_reused.csv").open(
                    "w", newline="", encoding="utf-8"
                )
            ) for split in SPLIT_NAMES
        }
        writers = {}
        for path in paths:
            header = _prepare_writers(path, files, writers, header)
            _copy_one_feature_file(path, targets, writers, found)
    return found


def _prepare_writers(path, files, writers, header):
    """Validate one feature header and initialize output writers."""
    with path.open(newline="", encoding="utf-8") as file:
        fields = csv.DictReader(file).fieldnames
    if fields is None:
        raise ValueError(f"{path} has no header")
    if header is not None and fields != header:
        raise ValueError(f"Feature header mismatch: {path}")
    if header is None:
        for split, output in files.items():
            writers[split] = csv.DictWriter(output, fieldnames=fields)
            writers[split].writeheader()
    return fields


def _copy_one_feature_file(path, targets, writers, found):
    """Copy matching rows from one extracted feature CSV."""
    with path.open(newline="", encoding="utf-8") as file:
        for feature_row in csv.DictReader(file):
            key = url_key(feature_row.get("url", ""))
            target = targets.get(key)
            if target is None or key in found:
                continue
            split, raw_row = target
            if str(feature_row.get("phishing", "")) != str(raw_row["label"]):
                raise ValueError(f"Label mismatch for URL: {raw_row['url']}")
            feature_row.update({"source_index": raw_row["source_index"], "url": raw_row["url"]})
            writers[split].writerow(feature_row)
            found.add(key)


def _write_missing_reports(output_dir, headers, rows_by_split, found):
    """Write missing-row CSVs and return split coverage counts."""
    report = {}
    for split in SPLIT_NAMES:
        missing = [row for row in rows_by_split[split] if url_key(row["url"]) not in found]
        with (output_dir / f"raw_{split}_missing_features.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=headers[split])
            writer.writeheader()
            writer.writerows(missing)
        report[split] = {
            "raw": len(rows_by_split[split]),
            "reused": len(rows_by_split[split]) - len(missing),
            "missing": len(missing),
        }
    return report


def parse_args() -> argparse.Namespace:
    """Parse feature-row reuse options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-split-dir", type=Path, required=True)
    parser.add_argument("--existing-features", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Reuse rows and print the resulting coverage report."""
    args = parse_args()
    report = reuse(args.raw_split_dir, args.existing_features, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
