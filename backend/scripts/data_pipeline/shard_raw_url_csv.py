"""Split raw URL CSV into deterministic shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from urllib.parse import urlsplit

import tldextract

TLD_EXTRACTOR = tldextract.TLDExtract(
    cache_dir=None,
    suffix_list_urls=(),
    fallback_to_snapshot=True,
)


def registered_domain(url: str) -> str:
    """Extract a registrable domain for deterministic sharding."""
    value = str(url or "").strip()
    if not value:
        return ""
    normalized = value if "://" in value else f"http://{value}"
    try:
        host = (urlsplit(normalized).hostname or "").lower()
    except ValueError:
        host = ""
    extracted = TLD_EXTRACTOR(host)
    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}"
    return host or value.lower()


def choose_shard(row_index: int, url: str, shards: int, strategy: str) -> int:
    """Choose a shard by row order or stable domain hash."""
    if strategy == "round-robin":
        return row_index % shards
    key = registered_domain(url)
    digest = hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:12], 16) % shards


def shard(input_path: Path, output_dir: Path, prefix: str, shards: int,
          strategy: str) -> list[Path]:
    """Write a raw CSV into deterministic header-preserving shards."""
    if shards <= 0:
        raise ValueError("shards must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [output_dir / f"{prefix}_shard_{index:02d}.csv" for index in range(shards)]
    _write_shard_files(input_path, output_paths, shards, strategy)
    return output_paths


def _write_shard_files(input_path, output_paths, shards, strategy):
    """Open shard files, copy headers and route input rows."""
    handles = [path.open("w", newline="", encoding="utf-8") for path in output_paths]
    try:
        writers = [csv.writer(handle) for handle in handles]
        with input_path.open(newline="", encoding="utf-8") as input_file:
            reader = csv.DictReader(input_file)
            if reader.fieldnames is None:
                raise ValueError("input CSV must have a header")
            for writer in writers:
                writer.writerow(reader.fieldnames)

            _write_shard_rows(reader, writers, shards, strategy)
    finally:
        for handle in handles:
            handle.close()

def _write_shard_rows(reader, writers, shards, strategy):
    """Route every input row to its deterministic shard writer."""
    for row_index, row in enumerate(reader):
        index = choose_shard(row_index, row.get("url", ""), shards, strategy)
        writers[index].writerow([row.get(field, "") for field in reader.fieldnames])


def parse_args():
    """Parse raw CSV sharding options."""
    parser = argparse.ArgumentParser(description="Split raw url,label CSV into shards")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="raw")
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--strategy", choices=["domain-hash", "round-robin"],
                        default="domain-hash")
    return parser.parse_args()


def main() -> None:
    """Shard the input CSV and print output paths."""
    args = parse_args()
    paths = shard(args.input, args.output_dir, args.prefix, args.shards, args.strategy)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
