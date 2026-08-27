"""Audit quality URL corpus balance, duplicates and shard coverage."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read all CSV rows from a corpus file."""
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_key(row: dict[str, str]) -> tuple[str, str]:
    """Return a normalized URL/label key."""
    return ((row.get("url") or "").strip().lower(), (row.get("label") or "").strip())


def url_only_key(row: dict[str, str]) -> str:
    """Return a normalized URL key without its label."""
    return (row.get("url") or "").strip().lower()


def label_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    """Count benign and phishing rows."""
    counts = Counter((row.get("label") or "").strip() for row in rows)
    return {"0": counts.get("0", 0), "1": counts.get("1", 0)}


def audit_file(path: Path) -> dict[str, object]:
    """Audit duplicates and balance for one CSV file."""
    rows = read_rows(path)
    keys = [row_key(row) for row in rows]
    url_keys = [url_only_key(row) for row in rows]
    counts = label_counts(rows)
    return {
        "path": str(path),
        "rows": len(rows),
        "labels": counts,
        "balanced_50_50": counts["0"] == counts["1"],
        "duplicate_url_rows": len(url_keys) - len(set(url_keys)),
        "duplicate_url_label_rows": len(keys) - len(set(keys)),
    }


def audit_shards(source: Path, shards: list[Path]) -> dict[str, object]:
    """Compare shard coverage and duplicates against a source file."""
    source_keys = {row_key(row) for row in read_rows(source)}
    source_url_keys = {url_only_key(row) for row in read_rows(source)}
    shard_key_sets, shard_url_key_sets, shard_reports, duplicate_between_shards = _audit_shard_rows(shards)

    combined = set().union(*shard_key_sets) if shard_key_sets else set()
    combined_urls = set().union(*shard_url_key_sets) if shard_url_key_sets else set()
    return {
        "source": str(source),
        "source_rows": len(source_keys),
        "combined_shard_rows": sum(len(keys) for keys in shard_key_sets),
        "missing_from_shards": len(source_keys - combined),
        "extra_in_shards": len(combined - source_keys),
        "missing_urls_from_shards": len(source_url_keys - combined_urls),
        "extra_urls_in_shards": len(combined_urls - source_url_keys),
        "duplicate_between_shards": duplicate_between_shards,
        "shards": shard_reports,
    }


def _audit_shard_rows(shards):
    """Collect sets, reports and cross-shard duplicate count."""
    key_sets, url_sets, reports, duplicate_count, seen = [], [], [], 0, set()
    for shard in shards:
        rows = read_rows(shard)
        keys, urls = {row_key(row) for row in rows}, {url_only_key(row) for row in rows}
        duplicate_count += len(seen & keys)
        seen.update(keys)
        key_sets.append(keys)
        url_sets.append(urls)
        reports.append({"path": str(shard), "rows": len(rows), "labels": label_counts(rows),
                        "duplicate_url_rows": len(rows) - len(urls),
                        "duplicate_url_label_rows": len(rows) - len(keys)})
    return key_sets, url_sets, reports, duplicate_count


def parse_args():
    """Parse corpus audit command-line options."""
    parser = argparse.ArgumentParser(description="Audit quality URL corpus files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--files", nargs="+", type=Path, required=True)
    parser.add_argument("--shard-group", nargs="+", action="append", default=[],
                        metavar=("SOURCE", "SHARD"),
                        help="Source file followed by one or more shard files")
    return parser.parse_args()


def main() -> None:
    """Audit requested files and shard groups, then write JSON output."""
    args = parse_args()
    report = {
        "files": [audit_file(path) for path in args.files],
        "shard_groups": [
            audit_shards(Path(group[0]), [Path(path) for path in group[1:]])
            for group in args.shard_group
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
