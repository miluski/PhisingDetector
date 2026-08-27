"""Import PhreshPhish URL + label pairs into raw url,label CSV."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("import_phreshphish")


def _require_datasets():
    """Load the optional Hugging Face datasets dependency."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install with: "
            "../.venv/bin/python -m pip install datasets"
        ) from exc
    return load_dataset


def _label_to_int(label: str) -> int | None:
    """Map PhreshPhish labels to binary integers."""
    normalized = (label or "").strip().lower()
    if normalized == "phish":
        return 1
    if normalized == "benign":
        return 0
    return None


def import_urls(output: Path, per_class_limit: int, split: str,
                dedupe_existing: list[Path]) -> None:
    """Stream PhreshPhish and write a deduplicated balanced CSV."""
    load_dataset = _require_datasets()
    seen = _load_seen_urls(dedupe_existing)
    if seen:
        logger.info("Loaded %d existing URLs for dedupe", len(seen))

    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {0: 0, 1: 0}
    total_limit = per_class_limit * 2

    logger.info("Streaming PhreshPhish split=%s, per_class_limit=%d", split, per_class_limit)
    dataset = load_dataset("phreshphish/phreshphish", split=split, streaming=True)

    with output.open("w", newline="", encoding="utf-8") as file:
        _write_stream(file, dataset, seen, counts, per_class_limit, total_limit)

    logger.info("Saved %s (%d benign, %d phish)", output, counts[0], counts[1])


def _load_seen_urls(paths):
    """Load URL keys from existing files used for deduplication."""
    seen = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as file:
            seen.update((row.get("url") or "").strip() for row in csv.DictReader(file))
        logger.info("Loaded existing URLs for dedupe: %s", path)
    return {url for url in seen if url}


def _write_stream(file, dataset, seen, counts, limit, total_limit):
    """Write valid streamed records until both class limits are met."""
    writer = csv.writer(file)
    writer.writerow(["url", "label"])
    for item in dataset:
        url, label = (item.get("url") or "").strip(), _label_to_int(item.get("label") or "")
        if not url or label is None or url in seen or counts[label] >= limit:
            continue
        writer.writerow([url, label])
        seen.add(url)
        counts[label] += 1
        _log_progress(counts, total_limit)
        if min(counts.values()) >= limit:
            break


def _log_progress(counts, total_limit):
    """Log every thousand imported rows."""
    written = counts[0] + counts[1]
    if written % 1000 == 0:
        logger.info("Progress: %d/%d rows (%d benign, %d phish)", written, total_limit, counts[0], counts[1])


def parse_args():
    """Parse PhreshPhish import options."""
    parser = argparse.ArgumentParser(description="Import PhreshPhish URLs to raw url,label CSV")
    parser.add_argument("--output", type=Path,
                        default=BACKEND_DIR / "data" / "url_corpus" / "raw_urls_phreshphish.csv")
    parser.add_argument("--per-class-limit", type=int, default=50000,
                        help="Max URLs per class")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--dedupe-existing", nargs="*", type=Path,
                        default=[BACKEND_DIR / "data" / "url_corpus" / "raw_urls.csv"],
                        help="Existing raw CSV used only for dedupe")
    return parser.parse_args()


def main() -> None:
    """Import PhreshPhish with validated class limits."""
    args = parse_args()
    if args.per_class_limit <= 0:
        raise SystemExit("--per-class-limit must be positive")
    import_urls(args.output, args.per_class_limit, args.split, args.dedupe_existing)


if __name__ == "__main__":
    sys.exit(main())
