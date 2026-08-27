"""Extract trusted-looking benign root URLs while excluding phishing domains."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from backend.scripts.data_pipeline.build_quality_url_corpus import is_benign_root, registered_domain
except ImportError: 
    from build_quality_url_corpus import is_benign_root, registered_domain


def load_phishing_domains(paths: list[Path]) -> set[str]:
    """Collect registered domains labelled phishing in source files."""
    domains = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if (row.get("label") or "").strip() != "1":
                    continue
                domain = registered_domain(row.get("url") or "")
                if domain:
                    domains.add(domain)
    return domains


def build(
    inputs: list[Path],
    phishing_inputs: list[Path],
    output: Path,
    limit: int | None,
) -> dict[str, int]:
    """Write unique benign roots absent from phishing domains."""
    phishing_domains = load_phishing_domains(phishing_inputs)
    urls, rejected = _collect_roots(inputs, phishing_domains, limit)
    _write_roots(output, urls)
    return {"written": len(urls), "phishing_domains": len(phishing_domains),
            "rejected_phishing_domain": rejected}


def _collect_roots(inputs, phishing_domains, limit):
    """Collect unique benign roots until an optional limit is reached."""
    seen_domains = set()
    urls = []
    rejected = 0
    for path in inputs:
        accepted, rejected_now = _collect_file_roots(path, phishing_domains, seen_domains, limit, len(urls))
        urls.extend(accepted)
        rejected += rejected_now
        if limit and len(urls) >= limit:
            break
    return urls, rejected


def _collect_file_roots(path, phishing_domains, seen_domains, limit, current):
    """Collect roots from one file until the global limit is reached."""
    urls, rejected = [], 0
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            url = (row.get("url") or "").strip()
            if (row.get("label") or "").strip() != "0" or not is_benign_root(url):
                continue
            domain = registered_domain(url)
            if not domain or domain in seen_domains:
                continue
            if domain in phishing_domains:
                rejected += 1
                continue
            seen_domains.add(domain)
            urls.append(url)
            if limit and current + len(urls) >= limit:
                break
    return urls, rejected


def _write_roots(output, urls):
    """Write collected roots as benign URL/label rows."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["url", "label"])
        writer.writerows((url, 0) for url in urls)


def parse_args() -> argparse.Namespace:
    """Parse benign-root extraction options."""
    parser = argparse.ArgumentParser(
        description="Build benign root hard negatives from ranked-domain CSV files"
    )
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--phishing-input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Build benign roots and print extraction statistics."""
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    stats = build(args.input, args.phishing_input, args.output, args.limit)
    print(
        f"Saved {args.output}: {stats['written']} benign roots; "
        f"rejected {stats['rejected_phishing_domain']} roots seen on "
        f"{stats['phishing_domains']} phishing domains"
    )


if __name__ == "__main__":
    main()
