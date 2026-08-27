"""Filter raw url,label CSV by excluding registered domains from split CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from build_url_corpus_splits import row_fingerprint


def load_excluded_domains(paths: list[Path]) -> set[str]:
    """Load registered domains that must be excluded from a raw CSV."""
    domains = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                domain = (row.get("registered_domain") or "").strip()
                if not domain and row.get("url"):
                    domain = row_fingerprint(row["url"])["registered_domain"]
                if domain:
                    domains.add(domain)
    return domains


def filter_file(input_path: Path, output_path: Path, excluded_domains: set[str]) -> dict[str, int]:
    """Filter valid URL/label rows by registered-domain membership."""
    kept = 0
    removed = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(newline="", encoding="utf-8") as in_f, \
            output_path.open("w", newline="", encoding="utf-8") as out_f:
        reader = csv.DictReader(in_f)
        _validate_input(reader)
        writer = csv.DictWriter(out_f, fieldnames=["url", "label"])
        writer.writeheader()
        kept, removed = _write_filtered_rows(reader, writer, excluded_domains)
    return {"kept": kept, "removed": removed}


def _validate_input(reader):
    """Validate raw URL/label input columns."""
    if reader.fieldnames is None or not {"url", "label"}.issubset(reader.fieldnames):
        raise ValueError("input CSV must contain url,label columns")


def _write_filtered_rows(reader, writer, excluded_domains):
    """Write rows outside excluded domains and return counts."""
    kept = removed = 0
    for row in reader:
        url, label = (row.get("url") or "").strip(), (row.get("label") or "").strip()
        if not url or label not in {"0", "1"}:
            continue
        if row_fingerprint(url)["registered_domain"] in excluded_domains:
            removed += 1
            continue
        writer.writerow({"url": url, "label": label})
        kept += 1
    return kept, removed


def parse_args():
    """Parse raw-domain filtering options."""
    parser = argparse.ArgumentParser(description="Exclude raw URLs whose domains occur in split CSVs")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-domains-from", nargs="+", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Filter a raw file and print kept/removed counts."""
    args = parse_args()
    excluded = load_excluded_domains(args.exclude_domains_from)
    stats = filter_file(args.input, args.output, excluded)
    print(
        f"Saved {args.output}: kept {stats['kept']} rows, "
        f"removed {stats['removed']} rows, excluded domains {len(excluded)}"
    )


if __name__ == "__main__":
    main()
