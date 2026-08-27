"""Create leakage-resistant train/validation/test splits for raw URL corpus."""

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlsplit

import tldextract


TLD_EXTRACTOR = tldextract.TLDExtract(
    cache_dir=None,
    fallback_to_snapshot=True,
    suffix_list_urls=(),
)
TOKEN_RE = re.compile(r"[0-9a-f]{8,}|[0-9]{3,}|[a-z0-9]{16,}", re.IGNORECASE)
MULTISLASH_RE = re.compile(r"/+")
PHISHING_HINTS = {
    "account", "admin", "auth", "bank", "billing", "confirm", "dashboard",
    "login", "logon", "oauth", "password", "payment", "secure", "signin",
    "support", "update", "verify", "wallet", "webscr",
}


def normalize_host(hostname: str) -> str:
    """Normalize a hostname for grouping and fingerprinting."""
    host = hostname.lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def registered_domain(hostname: str) -> str:
    """Return the registrable domain or normalized hostname."""
    ext = TLD_EXTRACTOR(hostname)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower()
    return hostname.lower()


def normalize_path(path: str) -> str:
    """Normalize path separators and replace high-entropy tokens."""
    decoded = unquote(path or "/").lower()
    decoded = MULTISLASH_RE.sub("/", decoded)
    parts = []
    for part in decoded.split("/"):
        if not part:
            continue
        parts.append(TOKEN_RE.sub("{token}", part))
    return "/" + "/".join(parts)


def normalize_query(query: str) -> str:
    """Return sorted lower-case query keys without parameter values."""
    if not query:
        return ""
    keys = sorted({key.lower() for key, _ in parse_qsl(query, keep_blank_values=True)})
    return "&".join(keys)


def row_fingerprint(url: str) -> dict:
    """Build domain, template and exact keys for one URL."""
    parsed = _parse_url(url)
    host, path, query, exact = _fingerprint_parts(url, parsed)
    domain = registered_domain(host)
    if not domain:
        domain = hashlib.sha1(url.encode("utf-8")).hexdigest()
    template = f"{domain}|{path}|{query}"
    return _fingerprint_payload(host, domain, path, query, template, exact)


def _parse_url(url):
    """Parse a URL while tolerating malformed source rows."""
    candidate = url if "://" in url else f"http://{url}"
    try:
        return urlsplit(candidate)
    except ValueError:
        return None


def _fingerprint_parts(url, parsed):
    """Extract normalized fingerprint components from a parsed URL."""
    if parsed is None:
        return "", normalize_path(url), "", url.lower()
    host = normalize_host(parsed.hostname or "")
    path = normalize_path(parsed.path)
    query = normalize_query(parsed.query)
    exact = f"{parsed.scheme.lower()}://{host}{parsed.path.lower()}?{parsed.query.lower()}"
    return host, path, query, exact


def _fingerprint_payload(host, domain, path, query, template, exact):
    """Hash normalized components and return the row metadata dictionary."""
    return {
        "host": host,
        "registered_domain": domain,
        "normalized_path": path,
        "query_keys": query,
        "template_key": hashlib.sha1(template.encode("utf-8")).hexdigest(),
        "exact_key": hashlib.sha1(exact.encode("utf-8")).hexdigest(),
    }


def load_rows(input_path: Path):
    """Load valid URL/label rows and calculate their fingerprints."""
    rows = []
    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "url" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("Input CSV must contain url,label columns")
        for source_index, row in enumerate(reader):
            url = (row.get("url") or "").strip()
            label = (row.get("label") or "").strip()
            if not url or label not in {"0", "1"}:
                continue
            fp = row_fingerprint(url)
            rows.append({
                "source_index": source_index,
                "url": url,
                "label": int(label),
                **fp,
            })
    return rows


def deduplicate(rows: list[dict], dedupe: str) -> tuple[list[dict], dict]:
    """Remove duplicate exact or template fingerprints within labels."""
    seen = set()
    kept = []
    removed = 0
    for row in rows:
        if dedupe == "none":
            key = None
        elif dedupe == "exact":
            key = (row["label"], row["exact_key"])
        else:
            key = (row["label"], row["template_key"])
        if key is not None and key in seen:
            removed += 1
            continue
        if key is not None:
            seen.add(key)
        kept.append(row)
    return kept, {"dedupe": dedupe, "removed": removed}


def group_rows(rows: list[dict], group_by: str) -> list[list[dict]]:
    """Group rows by the selected leakage boundary."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[group_by]].append(row)
    return list(grouped.values())


def split_groups(groups: list[list[dict]], val_size: float, test_size: float, seed: int):
    """Assign whole groups to deterministic train, validation and test splits."""
    rng = random.Random(seed)
    groups = list(groups)
    rng.shuffle(groups)
    splits = {"train": [], "val": [], "test": []}
    total = sum(len(group) for group in groups)
    val_target = total * val_size
    test_target = total * test_size
    counts = {"val": 0, "test": 0}

    for group in groups:
        target = _split_target(counts, test_target, val_target)
        if target != "train":
            counts[target] += len(group)
        splits[target].extend(group)
    return splits


def _split_target(counts, test_target, val_target):
    """Choose the next split according to target row counts."""
    if counts["test"] < test_target:
        return "test"
    if counts["val"] < val_target:
        return "val"
    return "train"


def balance_split(rows: list[dict], seed: int) -> list[dict]:
    """Downsample both labels to the smaller class count."""
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    if not by_label.get(0) or not by_label.get(1):
        return rows
    target = min(len(by_label[0]), len(by_label[1]))
    balanced = []
    for label_rows in by_label.values():
        picked = list(label_rows)
        rng.shuffle(picked)
        balanced.extend(picked[:target])
    return balanced


def is_hard_case(row: dict) -> bool:
    """Identify benign URLs with realistic path or query indicators."""
    path = row["normalized_path"]
    query = row["query_keys"]
    tokens = set(re.split(r"[^a-z0-9]+", path + " " + query))
    return bool(row["query_keys"] or len(path) > 1 or tokens & PHISHING_HINTS)


def write_split(path: Path, rows: list[dict]):
    """Write one split in stable source-index order."""
    fieldnames = [
        "source_index", "url", "label", "registered_domain", "host",
        "template_key", "normalized_path", "query_keys",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["source_index"]):
            writer.writerow({field: row[field] for field in fieldnames})


def summarize(rows: list[dict]) -> dict:
    """Return row, class, domain and hard-negative summary counts."""
    labels = Counter(row["label"] for row in rows)
    domains = {row["registered_domain"] for row in rows}
    templates = {row["template_key"] for row in rows}
    hard_legit = sum(1 for row in rows if row["label"] == 0 and is_hard_case(row))
    return {
        "rows": len(rows),
        "labels": {"legit": labels.get(0, 0), "phishing": labels.get(1, 0)},
        "registered_domains": len(domains),
        "templates": len(templates),
        "hard_legit": hard_legit,
    }


def parse_args():
    """Parse raw URL split generation options."""
    parser = argparse.ArgumentParser(description="Build robust raw URL train/val/test splits")
    parser.add_argument("--input", type=Path, required=True, help="Raw url,label CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for split CSV files")
    parser.add_argument(
        "--group-by", choices=["registered_domain", "template_key", "host"],
        default="registered_domain",
    )
    parser.add_argument("--dedupe", choices=["none", "exact", "template"], default="template")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance-per-split", action="store_true",
                        help="Downsample each split to equal legit/phishing counts")
    parser.add_argument(
        "--min-hard-benign-per-split", type=int, default=0,
        help="Fail unless every split contains at least this many benign hard negatives",
    )
    return parser.parse_args()


def _validate_args(args):
    """Validate split proportions and hard-negative requirements."""
    if args.val_size < 0 or args.test_size < 0 or args.val_size + args.test_size >= 1:
        raise SystemExit("val-size + test-size must be between 0 and 1")
    if args.min_hard_benign_per_split < 0:
        raise SystemExit("min-hard-benign-per-split must be non-negative")


def _prepare_splits(args, rows):
    """Deduplicate, group, split and optionally balance URL rows."""
    usable, dedupe_stats = deduplicate(rows, args.dedupe)
    splits = split_groups(group_rows(usable, args.group_by), args.val_size, args.test_size, args.seed)
    if args.balance_per_split:
        splits = {name: balance_split(values, args.seed + offset)
                  for offset, (name, values) in enumerate(splits.items())}
    return usable, dedupe_stats, splits


def _validate_hard_negatives(splits, minimum):
    """Fail when a configured split lacks enough benign hard negatives."""
    if not minimum:
        return
    counts = {name: sum(1 for row in values if row["label"] == 0 and is_hard_case(row))
              for name, values in splits.items()}
    failed = {name: count for name, count in counts.items() if count < minimum}
    if failed:
        raise SystemExit(f"Insufficient benign hard negatives in split(s): {failed}; requested={minimum}")


def _write_report(args, loaded, usable, dedupe_stats, splits):
    """Write split CSVs and the JSON summary report."""
    for name, values in splits.items():
        write_split(args.output_dir / f"raw_{name}.csv", values)
    report = {
        "input": str(args.input), "group_by": args.group_by,
        "balance_per_split": args.balance_per_split,
        "min_hard_benign_per_split": args.min_hard_benign_per_split,
        "dedupe": dedupe_stats, "loaded_rows": len(loaded), "usable_rows": len(usable),
        "splits": {name: summarize(values) for name, values in splits.items()},
    }
    with (args.output_dir / "split_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


def main():
    """Coordinate leakage-resistant split generation and reporting."""
    args = parse_args()
    _validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_rows(args.input)
    usable, dedupe_stats, splits = _prepare_splits(args, loaded)
    _validate_hard_negatives(splits, args.min_hard_benign_per_split)
    _write_report(args, loaded, usable, dedupe_stats, splits)


if __name__ == "__main__":
    main()
