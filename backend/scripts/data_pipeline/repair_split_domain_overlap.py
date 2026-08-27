"""Repair a small number of cross-split domain collisions without re-extraction."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from retrain_url_corpus_all_models import registered_domain
from config.feature_modes import CANONICAL_MODES


SPLITS = ("train", "val", "test")
MODES = CANONICAL_MODES


def read_frames(directory: Path, prefix: str) -> dict[str, pd.DataFrame]:
    """Read the three split files sharing a filename prefix."""
    return {
        split: pd.read_csv(directory / f"{prefix}_{split}.csv")
        for split in SPLITS
    }


def domain_sets(frames: dict[str, pd.DataFrame]) -> dict[str, set[str]]:
    """Build registrable-domain sets for each split."""
    return {
        split: {registered_domain(url) for url in frame["url"]}
        for split, frame in frames.items()
    }


def overlap_pairs(sets: dict[str, set[str]]) -> list[tuple[str, str, str]]:
    """Return every domain shared by two split sets."""
    pairs = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        pairs.extend((left, right, domain) for domain in sorted(sets[left] & sets[right]))
    return pairs


def choose_donor(
    frames: dict[str, pd.DataFrame],
    sets: dict[str, set[str]],
    source: str,
    target: str,
    label: int,
) -> int:
    """Choose a same-label source row whose domain is unique globally."""
    all_counts = _domain_counts(sets)
    candidates = frames[source][frames[source]["phishing"].astype(int) == label]
    for _, row in candidates.sort_values("source_index").iterrows():
        domain = registered_domain(row["url"])
        if domain not in sets[target] and all_counts[domain] == 1:
            return int(row["source_index"])
    raise ValueError(f"No safe donor found for {source}->{target}, label={label}")


def _domain_counts(sets: dict[str, set[str]]) -> dict[str, int]:
    """Count in how many split sets each domain occurs."""
    counts: dict[str, int] = {}
    for values in sets.values():
        for domain in values:
            counts[domain] = counts.get(domain, 0) + 1
    return counts


def swap_rows(
    frames: dict[str, pd.DataFrame],
    left: str,
    right: str,
    left_index: int,
    right_index: int,
) -> None:
    """Swap complete records between two split dataframes."""
    left_row = frames[left].index[frames[left]["source_index"] == left_index].tolist()
    right_row = frames[right].index[frames[right]["source_index"] == right_index].tolist()
    if len(left_row) != 1 or len(right_row) != 1:
        raise ValueError("source_index must identify exactly one row per split")
    left_position, right_position = left_row[0], right_row[0]
    left_values = frames[left].loc[left_position].copy()
    frames[left].loc[left_position] = frames[right].loc[right_position].to_numpy()
    frames[right].loc[right_position] = left_values.to_numpy()


def copy_inputs(raw_dir: Path, feature_dir: Path, output_raw: Path, output_features: Path) -> None:
    """Copy source files before applying the reversible repair."""
    output_raw.mkdir(parents=True, exist_ok=True)
    output_features.mkdir(parents=True, exist_ok=True)
    for source in raw_dir.iterdir():
        if source.is_file():
            shutil.copy2(source, output_raw / source.name)
    for source in feature_dir.iterdir():
        if source.is_file():
            shutil.copy2(source, output_features / source.name)


def apply_repair(
    raw_frames: dict[str, pd.DataFrame],
    feature_frames: dict[str, pd.DataFrame],
) -> list[dict[str, object]]:
    """Swap one same-label donor for each cross-split domain collision."""
    repairs = []
    sets = domain_sets(feature_frames)
    for left, right, domain in overlap_pairs(sets):
        repair = _repair_collision(feature_frames, raw_frames, sets, left, right, domain)
        sets = domain_sets(feature_frames)
        repairs.append(repair)
    return repairs


def _repair_collision(feature_frames, raw_frames, sets, left, right, domain):
    """Swap one collision row with a unique same-label donor."""
    match = feature_frames[left]
    match = match[match["url"].map(registered_domain) == domain]
    if len(match) != 1:
        raise ValueError(f"Expected one collision row in {left}: {domain}")
    left_index = int(match.iloc[0]["source_index"])
    label = int(match.iloc[0]["phishing"])
    donor_index = choose_donor(feature_frames, sets, right, left, label)
    donor = feature_frames[right]
    donor_url = donor.loc[donor["source_index"] == donor_index, "url"].iloc[0]
    swap_rows(feature_frames, left, right, left_index, donor_index)
    swap_rows(raw_frames, left, right, left_index, donor_index)
    return _repair_report(left, right, domain, left_index, donor_index, donor_url)


def _repair_report(left, right, domain, left_index, donor_index, donor_url):
    """Build an auditable description of one row swap."""
    return {
        "left_split": left,
        "right_split": right,
        "overlap_domain": domain,
        "left_source_index": left_index,
        "donor_source_index": donor_index,
        "donor_domain": registered_domain(donor_url),
    }


def write_frames(frames: dict[str, pd.DataFrame], directory: Path, prefix: str) -> None:
    """Persist repaired split dataframes with stable CSV formatting."""
    for split, frame in frames.items():
        frame.to_csv(directory / f"{prefix}_{split}.csv", index=False)


def parse_args() -> argparse.Namespace:
    """Parse repair input and output directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-raw-dir", type=Path, required=True)
    parser.add_argument("--output-feature-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Copy splits, repair collisions and write an audit report."""
    args = parse_args()
    copy_inputs(args.raw_dir, args.feature_dir, args.output_raw_dir, args.output_feature_dir)
    raw_frames = read_frames(args.output_raw_dir, "raw")
    feature_frames = {
        mode: read_frames(args.output_feature_dir, f"dataset_{mode}") for mode in MODES
    }
    repairs = apply_repair(raw_frames, feature_frames["full"])
    for mode, frames in feature_frames.items():
        if mode != "full":
            _swap_repaired_rows(frames, repairs)
        write_frames(frames, args.output_feature_dir, f"dataset_{mode}")
    write_frames(raw_frames, args.output_raw_dir, "raw")
    report = {"repairs": repairs, "source": str(args.feature_dir)}
    (args.output_raw_dir / "overlap_repair_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Repaired {len(repairs)} domain collision(s)")


def _swap_repaired_rows(frames: dict[str, pd.DataFrame], repairs: list[dict[str, object]]) -> None:
    """Apply the same source-index swaps to projected feature modes."""
    for repair in repairs:
        swap_rows(
            frames,
            str(repair["left_split"]),
            str(repair["right_split"]),
            int(repair["left_source_index"]),
            int(repair["donor_source_index"]),
        )


if __name__ == "__main__":
    main()
