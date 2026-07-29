"""Validate the canonical URL-corpus feature datasets and split boundaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config.project_paths import RESULTS_ROOT
from config.feature_names_config import FeatureConfig
from utils.split_dataset_loader import feature_columns

from build_url_corpus_splits import registered_domain


SPLITS = ("train", "val", "test")
MODES = ("full", "optimized", "selected")
DEFAULT_DATA_DIR = (
    BACKEND_ROOT / "data" / "url_corpus" / "quality_v4_repaired_features"
)
DEFAULT_OUTPUT = RESULTS_ROOT / "url_corpus" / "consistency_report.json"


def parse_args() -> argparse.Namespace:
    """Parse dataset and report paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_frames(data_dir: Path) -> dict[tuple[str, str], pd.DataFrame]:
    """Load every canonical feature mode and predefined data split."""
    return {
        (mode, split): pd.read_csv(
            data_dir / f"dataset_{mode}_{split}.csv"
        )
        for mode in MODES
        for split in SPLITS
    }


def frame_report(frame: pd.DataFrame) -> dict[str, object]:
    """Summarize one feature frame and its validation checks."""
    features = feature_columns(frame)
    labels = frame["phishing"].astype(int).value_counts().sort_index()
    values = frame[features].to_numpy(dtype=float)
    return {
        "rows": len(frame),
        "features": len(features),
        "labels": {str(key): int(value) for key, value in labels.items()},
        "duplicate_urls": int(frame["url"].duplicated().sum()),
        "duplicate_source_indices": int(frame["source_index"].duplicated().sum()),
        "missing_values": int(frame[features].isna().sum().sum()),
        "finite_features": bool(np.isfinite(values).all()),
    }


def split_overlaps(frames, mode: str) -> dict[str, dict[str, int]]:
    """Count URL and registrable-domain overlaps between split pairs."""
    urls = {
        split: set(frames[mode, split]["url"].astype(str).str.lower())
        for split in SPLITS
    }
    domains = {
        split: {registered_domain(url) for url in urls[split]}
        for split in SPLITS
    }
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    return {
        f"{left}_{right}": {
            "urls": len(urls[left] & urls[right]),
            "registered_domains": len(domains[left] & domains[right]),
        }
        for left, right in pairs
    }


def aligned_modes(frames, mode: str, split: str) -> bool:
    """Check that one reduced mode describes the same samples as full."""
    full, reduced = frames["full", split], frames[mode, split]
    columns = ["source_index", "url", "phishing"]
    return full[columns].equals(reduced[columns])


def expected_feature_counts() -> dict[str, int]:
    """Return prediction-feature counts from canonical schemas."""
    return {
        mode: len(FeatureConfig.get_feature_names(mode))
        for mode in MODES
    }


def build_report(data_dir: Path) -> dict[str, object]:
    """Build the complete consistency report."""
    frames = load_frames(data_dir)
    expected = expected_feature_counts()
    report = {
        "data_directory": str(data_dir),
        "expected_feature_counts": expected,
        "frames": _frame_reports(frames),
        "overlaps": {mode: split_overlaps(frames, mode) for mode in MODES},
        "mode_alignment": {
            mode: {
                split: aligned_modes(frames, mode, split)
                for split in SPLITS
            }
            for mode in MODES[1:]
        },
        "reduced_features_are_full_subsets": _reduced_subsets(frames),
    }
    report["valid"] = report_is_valid(report)
    return report


def _frame_reports(frames) -> dict[str, dict[str, object]]:
    """Return nested reports for every mode and split."""
    return {
        mode: {
            split: frame_report(frames[mode, split])
            for split in SPLITS
        }
        for mode in MODES
    }


def _reduced_subsets(frames) -> dict[str, bool]:
    """Check reduced feature membership against the full schema."""
    full = set(feature_columns(frames["full", "train"]))
    return {
        mode: set(feature_columns(frames[mode, "train"])) < full
        for mode in MODES[1:]
    }


def report_is_valid(report: dict[str, object]) -> bool:
    """Return whether all corpus consistency conditions pass."""
    frames = report["frames"]
    expected = report["expected_feature_counts"]
    frame_checks = [
        _frame_is_valid(frames[mode][split], expected[mode])
        for mode in MODES for split in SPLITS
    ]
    overlap_checks = [
        not any(values.values())
        for mode in MODES for values in report["overlaps"][mode].values()
    ]
    alignment = all(
        all(values.values()) for values in report["mode_alignment"].values()
    )
    subsets = all(report["reduced_features_are_full_subsets"].values())
    return all(frame_checks + overlap_checks) and alignment and subsets


def _frame_is_valid(frame: dict[str, object], expected_features: int) -> bool:
    """Validate balance, uniqueness, completeness and feature count."""
    labels = frame["labels"]
    return all((
        frame["features"] == expected_features,
        labels.get("0") == labels.get("1"),
        set(labels) == {"0", "1"},
        frame["duplicate_urls"] == 0,
        frame["duplicate_source_indices"] == 0,
        frame["missing_values"] == 0,
        frame["finite_features"],
    ))


def write_report(report: dict[str, object], output: Path) -> None:
    """Write one deterministic JSON report."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Validate the corpus, save the report and fail on inconsistency."""
    args = parse_args()
    report = build_report(args.data_dir)
    write_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
