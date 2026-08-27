"""Build a higher-quality raw URL corpus from licensed source CSV files."""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlsplit

import tldextract


PHISHING_HINTS = {
    "account", "admin", "auth", "bank", "billing", "confirm", "dashboard",
    "login", "logon", "oauth", "password", "payment", "secure", "signin",
    "support", "update", "verify", "wallet", "webscr", "reset", "invoice",
}
HARD_NEGATIVE_TEMPLATES = (
    "/login",
    "/account",
    "/payment",
    "/verify-email",
    "/secure",
    "/support/contact",
    "/about",
    "/contact",
    "/docs",
    "/help",
    "/playlist/example",
    "/pricing",
    "/status",
    "/watch",
    "/products",
)
TOKEN_RE = re.compile(r"[0-9a-f]{8,}|[0-9]{3,}|[a-z0-9]{16,}", re.IGNORECASE)
TLD_EXTRACTOR = tldextract.TLDExtract(
    cache_dir=None,
    fallback_to_snapshot=True,
    suffix_list_urls=(),
)


def normalize_url(url: str) -> str:
    """Trim a source URL while preserving its original path and query."""
    return str(url or "").strip()


def url_key(url: str) -> str:
    """Return a case-insensitive deduplication key."""
    return normalize_url(url).lower()


def is_hard_benign(url: str) -> bool:
    """Return whether a benign URL contains a realistic path or query."""
    value = normalize_url(url)
    candidate = value if "://" in value else f"http://{value}"
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return False

    path = unquote(parsed.path or "").lower()
    query = parsed.query or ""
    query_keys = {key.lower() for key, _ in parse_qsl(query, keep_blank_values=True)}
    path_tokens = set(re.split(r"[^a-z0-9]+", TOKEN_RE.sub("{token}", path)))

    has_real_path = path not in {"", "/"}
    has_query = bool(query_keys)
    has_sensitive_word = bool((path_tokens | query_keys) & PHISHING_HINTS)
    return has_real_path or has_query or has_sensitive_word


def is_benign_root(url: str) -> bool:
    """Return whether a URL is a host root without query or fragment."""
    value = normalize_url(url)
    candidate = value if "://" in value else f"http://{value}"
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return False
    return bool(parsed.hostname) and parsed.path in {"", "/"} and not parsed.query and not parsed.fragment


def registered_domain(url: str) -> str:
    """Extract a registrable domain from a URL or host."""
    value = normalize_url(url)
    candidate = value if "://" in value else f"http://{value}"
    try:
        host = (urlsplit(candidate).hostname or "").lower().strip(".")
    except ValueError:
        return ""
    if host.startswith("www."):
        host = host[4:]
    extracted = TLD_EXTRACTOR(host)
    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}".lower()
    return host


def load_excluded(paths: list[Path]) -> set[str]:
    """Load URL keys that must not be reintroduced into the output."""
    excluded = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                url = normalize_url(row.get("url", ""))
                if url:
                    excluded.add(url_key(url))
    return excluded


def load_rows(paths: list[Path], excluded: set[str], require_hard_benign: bool) -> list[tuple[str, int]]:
    """Load valid, unique URL/label rows from all source files."""
    seen = set(excluded)
    rows = []
    for path in paths:
        rows.extend(_load_rows_from_file(path, seen, require_hard_benign))
    return rows


def _load_rows_from_file(path, seen, require_hard_benign):
    """Load and deduplicate one source CSV."""
    rows = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"url", "label"}.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain url,label columns")
        for row in reader:
            item = _parse_source_row(row, require_hard_benign)
            if item and url_key(item[0]) not in seen:
                seen.add(url_key(item[0]))
                rows.append(item)
    return rows


def _parse_source_row(row, require_hard_benign):
    """Parse one source row and apply the optional benign filter."""
    url = normalize_url(row.get("url", ""))
    label_raw = (row.get("label") or "").strip()
    if not url or label_raw not in {"0", "1"}:
        return None
    label = int(label_raw)
    if require_hard_benign and label == 0 and not is_hard_benign(url):
        return None
    return url, label


def load_trusted_domains(path: Path) -> list[str]:
    """Load domains from Tranco-style CSV or the project's url,label CSV."""
    """Load unique registrable domains from a Tranco-style CSV."""
    domains = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            candidate = _trusted_candidate(row)
            if candidate in {"domain", "url"} or candidate.startswith("#"):
                continue
            if "://" in candidate:
                domain = registered_domain(candidate)
            else:
                domain = registered_domain(f"https://{candidate}/")
            if domain and domain not in domains:
                domains.append(domain)
    return domains


def _trusted_candidate(row):
    """Select a domain-like value from one CSV row."""
    return next((value.strip().lower() for value in (row[0], row[-1])
                 if "://" in value or ("." in value and value not in {"domain", "url"})), "")


def build_hard_negative_rows(
    trusted_domains: list[str],
    phishing_domains: set[str],
    excluded: set[str],
    per_domain: int,
) -> list[tuple[str, int]]:
    """Create synthetic benign paths on trusted domains."""
    if per_domain <= 0:
        return []
    rows = []
    for domain in trusted_domains:
        if domain in phishing_domains:
            continue
        rows.extend(_domain_hard_rows(domain, excluded, per_domain))
    return rows


def _domain_hard_rows(domain, excluded, per_domain):
    """Create non-duplicate hard paths for one trusted domain."""
    rows = []
    for path in HARD_NEGATIVE_TEMPLATES[:per_domain]:
        url = f"https://{domain}{path}"
        if url_key(url) not in excluded:
            rows.append((url, 0))
    return rows


def exclude_benign_roots_on_phishing_domains(
    rows: list[tuple[str, int]],
) -> tuple[list[tuple[str, int]], int]:
    """Remove benign roots whose domain is present among phishing URLs."""
    phishing_domains = {
        registered_domain(url)
        for url, label in rows
        if label == 1 and registered_domain(url)
    }
    filtered = [row for row in rows if not _is_root_collision(row, phishing_domains)]
    removed = len(rows) - len(filtered)
    return filtered, removed


def _is_root_collision(row, phishing_domains):
    """Return whether one benign root collides with a phishing domain."""
    url, label = row
    return label == 0 and is_benign_root(url) and registered_domain(url) in phishing_domains


def sample_balanced(
    rows: list[tuple[str, int]],
    per_class_limit: int | None,
    seed: int,
    minimum_benign_roots: int = 0,
) -> list[tuple[str, int]]:
    """Return a deterministic balanced sample honoring benign-root minimums."""
    rng = random.Random(seed)
    by_label = _group_labels(rows)
    target = _sample_target(by_label, per_class_limit)
    selected_benign = _select_benign(by_label[0], target, minimum_benign_roots, rng)
    phishing_rows = list(by_label[1])
    rng.shuffle(phishing_rows)
    sampled = selected_benign + phishing_rows[:target]
    rng.shuffle(sampled)
    return sampled


def _group_labels(rows):
    """Group rows by binary label and validate both classes."""
    by_label = defaultdict(list)
    for row in rows:
        by_label[row[1]].append(row)
    if not by_label.get(0) or not by_label.get(1):
        raise ValueError("Both classes must be present")
    return by_label


def _sample_target(by_label, per_class_limit):
    """Return the balanced per-class sample target."""
    target = min(len(by_label[0]), len(by_label[1]))
    return min(target, per_class_limit) if per_class_limit else target


def _select_benign(rows, target, minimum, rng):
    """Select benign rows while guaranteeing a requested root count."""
    roots = [row for row in rows if is_benign_root(row[0])]
    if minimum > min(target, len(roots)):
        raise ValueError(
            "Requested minimum benign roots cannot be satisfied: "
            f"requested={minimum}, available={len(roots)}, target={target}"
        )
    rng.shuffle(roots)
    selected = roots[:minimum]
    keys = {url_key(url) for url, _ in selected}
    remainder = [row for row in rows if url_key(row[0]) not in keys]
    rng.shuffle(remainder)
    return selected + remainder[:target - len(selected)]


def write_rows(rows: list[tuple[str, int]], output: Path) -> None:
    """Write sampled URL/label rows to a CSV file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "label"])
        writer.writerows(rows)


def parse_args():
    """Parse quality corpus builder command-line arguments."""
    return _quality_parser().parse_args()


def _quality_parser():
    """Build the quality corpus argument parser."""
    parser = argparse.ArgumentParser(description="Build a quality raw URL corpus")
    _add_source_args(parser)
    _add_sampling_args(parser)
    _add_negative_args(parser)
    return parser


def _add_source_args(parser):
    """Add input, exclusion and output options."""
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--exclude", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)


def _add_sampling_args(parser):
    """Add balancing and seed options."""
    parser.add_argument("--per-class-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-benign-roots", type=int, default=0)


def _add_negative_args(parser):
    """Add hard-negative generation options."""
    parser.add_argument("--trusted-domains", type=Path, default=None)
    parser.add_argument("--hard-negative-per-domain", type=int, default=0)
    parser.add_argument("--require-hard-benign", action="store_true")
    parser.add_argument("--exclude-benign-root-domains-seen-in-phishing", action="store_true")


def _validate_args(args):
    """Validate hard-negative and benign-root option combinations."""
    if args.minimum_benign_roots < 0 or args.hard_negative_per_domain < 0:
        raise SystemExit("Counts must be non-negative")
    if args.hard_negative_per_domain and not args.trusted_domains:
        raise SystemExit("--trusted-domains is required for hard-negative generation")
    if args.require_hard_benign and args.minimum_benign_roots:
        raise SystemExit("--require-hard-benign cannot be combined with --minimum-benign-roots")


def _augment_rows(args, rows, excluded):
    """Append trusted-domain hard negatives and remove root collisions."""
    generated = 0
    if args.trusted_domains and args.hard_negative_per_domain:
        phishing_domains = {registered_domain(url) for url, label in rows if label == 1 and registered_domain(url)}
        existing = excluded | {url_key(url) for url, _ in rows}
        hard_rows = build_hard_negative_rows(load_trusted_domains(args.trusted_domains), phishing_domains,
                                             existing, args.hard_negative_per_domain)
        rows.extend(hard_rows)
        generated = len(hard_rows)
    removed = 0
    if args.exclude_benign_root_domains_seen_in_phishing:
        rows, removed = exclude_benign_roots_on_phishing_domains(rows)
    return rows, generated, removed


def _print_summary(args, sampled, removed, generated):
    """Print class and synthetic-hard-negative counts."""
    counts = Counter(label for _, label in sampled)
    roots = sum(1 for url, label in sampled if label == 0 and is_benign_root(url))
    print(f"Saved {args.output}: {len(sampled)} rows ({counts[1]} phishing, {counts[0]} legit, "
          f"{roots} benign roots, {removed} root-domain collisions removed, {generated} generated hard negatives)")


def main() -> None:
    """Coordinate raw corpus filtering, balancing and report output."""
    args = parse_args()
    _validate_args(args)
    excluded = load_excluded(args.exclude)
    rows = load_rows(args.input, excluded, args.require_hard_benign)
    rows, generated, removed = _augment_rows(args, rows, excluded)
    sampled = sample_balanced(rows, args.per_class_limit, args.seed, args.minimum_benign_roots)
    write_rows(sampled, args.output)
    _print_summary(args, sampled, removed, generated)


if __name__ == "__main__":
    main()
