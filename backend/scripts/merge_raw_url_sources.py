"""Merge multiple raw URL CSV sources into one deduplicated url,label file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent


def merge(inputs: list[Path], output: Path) -> dict[str, int]:
    seen = set()
    counts = {0: 0, 1: 0}

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["url", "label"])

        for input_path in inputs:
            with input_path.open(newline="", encoding="utf-8") as in_f:
                reader = csv.DictReader(in_f)
                for row in reader:
                    url = (row.get("url") or "").strip()
                    label_raw = (row.get("label") or "").strip()
                    if not url or label_raw not in ("0", "1") or url in seen:
                        continue

                    label = int(label_raw)
                    writer.writerow([url, label])
                    seen.add(url)
                    counts[label] += 1

    return {
        "total": counts[0] + counts[1],
        "legit": counts[0],
        "phishing": counts[1],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Merge raw url,label CSV sources")
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=BACKEND_DIR / "data" / "url_corpus" / "raw_urls.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = merge(args.input, args.output)
    print(
        f"Saved {args.output}: {stats['total']} rows "
        f"({stats['phishing']} phishing, {stats['legit']} legit)"
    )


if __name__ == "__main__":
    main()
