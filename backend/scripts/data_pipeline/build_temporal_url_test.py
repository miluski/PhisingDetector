"""Build a future, domain-disjoint URL test set from dated source files.

Input files must contain ``url``, ``label`` and ``collected_at``. The script
does not infer time from filesystem timestamps or URL text. That prevents a
fake temporal split. Use a source file collected after training data and keep
its source/date in the output manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import pandas as pd
import tldextract


TLD_EXTRACTOR = tldextract.TLDExtract(
    cache_dir=None, fallback_to_snapshot=True, suffix_list_urls=()
)


def registered_domain(url: str) -> str:
    """Extract a registrable domain from a URL."""
    value = str(url or "")
    candidate = value if "://" in value else f"https://{value}"
    extracted = TLD_EXTRACTOR(candidate)
    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}".lower()
    return extracted.domain.lower()


def load_domains(paths: list[Path]) -> set[str]:
    """Load registered domains from training feature files."""
    domains = set()
    for path in paths:
        frame = pd.read_csv(path, usecols=lambda column: column in {"url", "registered_domain"})
        if "registered_domain" in frame:
            domains.update(frame["registered_domain"].dropna().astype(str).str.lower())
        elif "url" in frame:
            domains.update(registered_domain(url) for url in frame["url"].dropna())
    return {domain for domain in domains if domain}


def build(
    inputs: list[Path],
    train_inputs: list[Path],
    cutoff: str,
    output: Path,
    per_class: int,
    seed: int,
) -> dict[str, object]:
    """Build a future, domain-disjoint balanced test set and manifest."""
    cutoff_value = pd.Timestamp(cutoff, tz="UTC")
    excluded_domains = load_domains(train_inputs)
    candidates = _load_candidates(inputs, cutoff_value, excluded_domains)
    selected = _select_balanced(candidates, per_class, seed)
    report = _write_output(output, selected, cutoff_value, excluded_domains)
    return report


def _load_candidates(inputs, cutoff_value, excluded_domains):
    """Load valid post-cutoff rows that do not overlap training domains."""
    candidates, seen = [], set()
    for path in inputs:
        frame = pd.read_csv(path)
        _validate_columns(path, frame)
        candidates.extend(_rows_after_cutoff(path, frame, cutoff_value, excluded_domains, seen))
    return candidates


def _validate_columns(path, frame):
    """Validate temporal source columns."""
    required = {"url", "label", "collected_at"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks required columns: {sorted(missing)}")


def _rows_after_cutoff(path, frame, cutoff_value, excluded_domains, seen):
    """Convert one source frame to unique candidate dictionaries."""
    dates = pd.to_datetime(frame["collected_at"], utc=True, errors="coerce")
    rows = []
    for index, row in frame.iterrows():
        candidate = _candidate_row(path, row, dates.iloc[index], cutoff_value)
        if candidate and candidate["registered_domain"] not in excluded_domains:
            key = (candidate["label"], candidate["url"].lower())
            if key not in seen:
                seen.add(key)
                rows.append(candidate)
    return rows


def _candidate_row(path, row, date, cutoff_value):
    """Convert a temporal source row or return none when invalid."""
    url = str(row.get("url") or "").strip()
    label = str(row.get("label") if row.get("label") is not None else "").strip()
    if not url or label not in {"0", "1"} or pd.isna(date) or date < cutoff_value:
        return None
    return {"url": url, "label": int(label), "collected_at": date.isoformat(),
            "source": str(row.get("source") or path.stem), "registered_domain": registered_domain(url)}


def _select_balanced(candidates, per_class, seed):
    """Select an equal deterministic number of rows for both labels."""
    by_label = {label: [row for row in candidates if row["label"] == label] for label in (0, 1)}
    available = min(len(by_label[0]), len(by_label[1]))
    if available < per_class:
        raise ValueError(
            f"Not enough future domain-disjoint rows: available_per_class={available}, requested={per_class}"
        )
    rng = random.Random(seed)
    selected = []
    for label in (0, 1):
        rows = by_label[label][:]
        rng.shuffle(rows)
        selected.extend(rows[:per_class])
    rng.shuffle(selected)
    return selected


def _write_output(output, selected, cutoff_value, excluded_domains):
    """Write temporal CSV and JSON manifest."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["url", "label", "collected_at", "source", "registered_domain"]
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    report = _temporal_report(selected, cutoff_value, excluded_domains)
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def _temporal_report(selected, cutoff_value, excluded_domains):
    """Build the temporal-test manifest without reading the output file."""
    counts = {label: sum(row["label"] == label for row in selected) for label in (0, 1)}
    return {
        "cutoff": cutoff_value.isoformat(), "excluded_train_domains": len(excluded_domains),
        "rows": len(selected), "labels": {"benign": counts[0], "phishing": counts[1]},
        "sources": sorted({str(row["source"]) for row in selected}),
        "registered_domains": len({str(row["registered_domain"]) for row in selected}),
        "independent_temporal_test": True,
    }


def parse_args() -> argparse.Namespace:
    """Parse temporal test-set generation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--train-input", nargs="+", type=Path, required=True)
    parser.add_argument("--cutoff", required=True, help="UTC date, e.g. 2026-01-01")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Validate options, build the test set and print its manifest."""
    args = parse_args()
    if args.per_class <= 0:
        raise SystemExit("--per-class must be positive")
    print(json.dumps(build(
        args.input, args.train_input, args.cutoff, args.output, args.per_class, args.seed
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
