"""Replace benign path samples with benign roots without changing split sizes."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

try:
    from backend.scripts.data_pipeline.build_quality_url_corpus import registered_domain
    from backend.scripts.data_pipeline.build_url_corpus_splits import row_fingerprint
except ImportError: 
    from build_quality_url_corpus import registered_domain
    from build_url_corpus_splits import row_fingerprint


SPLIT_NAMES = ("train", "val", "test")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV and return its header and rows."""
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        return reader.fieldnames, list(reader)


def proportional_targets(total: int, capacities: dict[str, int]) -> dict[str, int]:
    """Distribute a replacement total proportionally across splits."""
    capacity_total = sum(capacities.values())
    exact = {
        split: total * capacity / capacity_total
        for split, capacity in capacities.items()
    }
    targets = {split: int(value) for split, value in exact.items()}
    remaining = total - sum(targets.values())
    order = sorted(
        capacities,
        key=lambda split: exact[split] - targets[split],
        reverse=True,
    )
    for split in order[:remaining]:
        targets[split] += 1
    return targets


def inject(source_dir, benign_roots_path, output_dir, root_limit, seed):
    """Replace benign rows with non-overlapping benign root URLs."""
    plan = _injection_plan(source_dir, benign_roots_path, root_limit, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_splits, corrected_rows = _write_replaced_splits(
        output_dir, plan["headers"], plan["source_rows"], plan["selected"],
        plan["targets"], plan["next_source_index"], seed
    )
    _write_combined_rows(output_dir, corrected_rows)
    report = _build_report(
        source_dir, benign_roots_path, output_dir,
        plan["candidates"], plan["selected"], report_splits,
    )
    _write_report(output_dir, report)
    return report


def _injection_plan(source_dir, benign_roots_path, root_limit, seed):
    """Load inputs and calculate deterministic replacement allocation."""
    headers, source_rows, capacities, used_domains, max_index = _load_sources(source_dir)
    candidates = _root_candidates(benign_roots_path, used_domains)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:min(root_limit, len(candidates))]
    return {"headers": headers, "source_rows": source_rows, "candidates": candidates,
            "selected": selected, "targets": proportional_targets(len(selected), capacities),
            "next_source_index": max_index + 1}


def _load_sources(source_dir):
    """Load source split rows and calculate replacement capacities."""
    headers = {}
    source_rows = {}
    used_domains = set()
    max_source_index = -1
    capacities = {}

    for split in SPLIT_NAMES:
        header, rows = read_csv(source_dir / f"raw_{split}.csv")
        headers[split] = header
        source_rows[split] = rows
        capacities[split] = sum(row["label"] == "0" for row in rows)
        used_domains.update(row["registered_domain"] for row in rows)
        max_source_index = max(
            max_source_index,
            *(int(row["source_index"]) for row in rows),
        )

    return headers, source_rows, capacities, used_domains, max_source_index


def _root_candidates(benign_roots_path, used_domains):
    """Select unique root URLs whose domains are absent from source splits."""
    _, root_rows = read_csv(benign_roots_path)
    candidates = []
    candidate_domains = set()
    for row in root_rows:
        url = row["url"].strip()
        domain = registered_domain(url)
        if not domain or domain in used_domains or domain in candidate_domains:
            continue
        candidate_domains.add(domain)
        candidates.append(url)

    return candidates


def _write_replaced_splits(output_dir, headers, source_rows, selected, targets, next_source_index, seed):
    """Replace benign positions and write each corrected split."""
    cursor = 0
    report_splits = {}
    corrected_rows = []

    for offset, split in enumerate(SPLIT_NAMES):
        result, added, removed, next_source_index = _replace_one_split(
            source_rows[split], selected[cursor:cursor + targets[split]],
            targets[split], headers[split], output_dir / f"raw_{split}.csv", seed + offset + 1, next_source_index
        )
        cursor += targets[split]
        corrected_rows.extend(result)
        report_splits[split] = _split_report(result, added, removed)

    return report_splits, corrected_rows


def _replace_one_split(rows, urls, replace_count, headers, path, seed, next_index):
    """Replace selected benign positions in one split and write it."""
    positions = [i for i, row in enumerate(rows) if row["label"] == "0"]
    random.Random(seed).shuffle(positions)
    removed = set(positions[:replace_count])
    kept = [row for i, row in enumerate(rows) if i not in removed]
    added = [_root_row(url, next_index + offset) for offset, url in enumerate(urls)]
    result = kept + added
    _write_split(path, headers, result)
    return result, added, removed, next_index + len(added)


def _root_row(url, source_index):
    """Build a split row from a benign root URL."""
    fingerprint = row_fingerprint(url)
    return {"source_index": str(source_index), "url": url, "label": "0",
            "registered_domain": fingerprint["registered_domain"], "host": fingerprint["host"],
            "template_key": fingerprint["template_key"], "normalized_path": fingerprint["normalized_path"],
            "query_keys": fingerprint["query_keys"]}


def _split_report(rows, added, removed):
    """Build counts for one corrected split."""
    return {"rows": len(rows), "benign": sum(row["label"] == "0" for row in rows),
            "phishing": sum(row["label"] == "1" for row in rows),
            "benign_roots_added": len(added), "benign_paths_removed": len(removed)}


def _write_split(path, headers, rows):
    """Write one corrected split preserving its source columns."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _write_combined_rows(output_dir, rows):
    """Write a combined URL/label audit CSV."""
    with (output_dir / "raw_urls_quality_corrected.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["url", "label"])
        writer.writerows((row["url"], row["label"]) for row in rows)


def _build_report(source_dir, roots_path, output_dir, candidates, selected, report_splits):
    """Build domain-overlap and replacement summary."""

    split_domains = _split_domains(output_dir)
    return {
        "source_dir": str(source_dir),
        "benign_roots": str(roots_path),
        "available_non_overlapping_roots": len(candidates),
        "selected_roots": len(selected),
        "splits": report_splits,
        "domain_overlap": {
            "train_val": len(split_domains["train"] & split_domains["val"]),
            "train_test": len(split_domains["train"] & split_domains["test"]),
            "val_test": len(split_domains["val"] & split_domains["test"]),
        },
    }


def _split_domains(output_dir):
    """Load registered-domain sets for corrected split files."""
    return {split: {row["registered_domain"] for row in read_csv(output_dir / f"raw_{split}.csv")[1]}
            for split in SPLIT_NAMES}
def _write_report(output_dir, report):
    """Persist the injection report as JSON."""
    with (output_dir / "injection_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    """Parse benign-root injection options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--benign-roots", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root-limit", type=int, default=39000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Validate options, inject roots and print the report."""
    args = parse_args()
    if args.root_limit <= 0:
        raise SystemExit("--root-limit must be positive")
    report = inject(
        args.source_dir,
        args.benign_roots,
        args.output_dir,
        args.root_limit,
        args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
