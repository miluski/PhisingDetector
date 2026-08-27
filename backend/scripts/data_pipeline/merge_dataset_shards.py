"""Merge extracted dataset shards into one CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def merge(inputs: list[Path], output: Path) -> int:
    """Merge CSV shards with identical headers into one atomic output."""
    if not inputs:
        raise ValueError("at least one input shard is required")

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    total_rows, header = 0, None
    with tmp_output.open("w", newline="", encoding="utf-8") as output_file:
        for input_path in inputs:
            header, added = _append_shard(input_path, output_file, header)
            total_rows += added

    tmp_output.replace(output)
    return total_rows


def _append_shard(path, output_file, header):
    """Append one shard and return its header plus row count."""
    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        if header is None:
            header = reader.fieldnames
            csv.DictWriter(output_file, fieldnames=header).writeheader()
        elif reader.fieldnames != header:
            raise ValueError(f"{path} header differs from previous shards")
        writer = csv.DictWriter(output_file, fieldnames=header)
        return header, sum(_write_row(writer, row) for row in reader)


def _write_row(writer, row):
    """Write one row and return its count contribution."""
    writer.writerow(row)
    return 1


def parse_args():
    """Parse dataset shard merge options."""
    parser = argparse.ArgumentParser(description="Merge dataset shard CSV files")
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Merge requested shards and print the resulting row count."""
    args = parse_args()
    rows = merge(args.input, args.output)
    print(f"Saved {args.output}: {rows} rows")


if __name__ == "__main__":
    main()
