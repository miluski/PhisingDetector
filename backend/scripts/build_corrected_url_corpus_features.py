"""Extract missing corrected-corpus feature shards with resumable live lookups."""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent.parent
EXTRACT_SCRIPT = BACKEND_DIR / "scripts" / "build_url_corpus_dataset.py"


def build_shard(input_path: Path, workers: int, timeout: int) -> Path:
    """Run resumable live feature extraction for one raw shard."""
    output_name = input_path.name.replace("raw_", "dataset_full_", 1)
    output_path = input_path.with_name(output_name)
    command = _extract_command(input_path, output_path, workers, timeout)
    subprocess.run(command, cwd=BACKEND_DIR.parent, check=True)
    return output_path


def _extract_command(input_path, output_path, workers, timeout):
    """Build the resumable feature-extraction subprocess command."""
    return [
        sys.executable,
        str(EXTRACT_SCRIPT),
        "--input", str(input_path),
        "--mode", "full",
        "--output", str(output_path),
        "--workers", str(workers),
        "--network-mode", "live",
        "--missing-policy", "keep",
        "--timeout", str(timeout),
        "--progress-interval", "1000",
        "--resume-temp",
        "--include-metadata",
        "--failure-policy", "keep-empty",
    ]


def parse_args() -> argparse.Namespace:
    """Parse corrected feature extraction options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--parallel-shards", type=int, default=8)
    parser.add_argument("--workers-per-shard", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    """Extract all missing shards concurrently."""
    args = parse_args()
    shard_paths = sorted(args.feature_dir.glob("*_missing_shards/raw_*_shard_*.csv"))
    if not shard_paths:
        raise SystemExit(f"No missing raw shards found under {args.feature_dir}")

    print(f"Extracting {len(shard_paths)} shards with live network features")
    with ThreadPoolExecutor(max_workers=args.parallel_shards) as pool:
        futures = {
            pool.submit(
                build_shard, path, args.workers_per_shard, args.timeout
            ): path
            for path in shard_paths
        }
        for future in as_completed(futures):
            input_path = futures[future]
            output_path = future.result()
            print(f"Completed {input_path.name} -> {output_path.name}")


if __name__ == "__main__":
    main()
