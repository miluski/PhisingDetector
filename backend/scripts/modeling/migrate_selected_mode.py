"""Rename the active URL-corpus selection mode to the canonical ``selected``."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = BACKEND_ROOT / "data" / "url_corpus" / "quality_v4_repaired_features"
MODEL_DIR = BACKEND_ROOT / "models" / "staged_url_corpus_v4"
MANIFEST_SOURCE = BACKEND_ROOT / "config" / "feature_manifests" / "url_corpus_optimized_v4.json"
MANIFEST_TARGET = BACKEND_ROOT / "config" / "feature_manifests" / "url_corpus" / "quality_v4" / "selected.json"


def parse_args() -> argparse.Namespace:
    """Parse migration options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Only list planned changes")
    return parser.parse_args()


def rename_pair(source: Path, target: Path, dry_run: bool) -> None:
    """Rename one file when the source exists and target is absent."""
    if not source.exists():
        return
    if target.exists():
        raise FileExistsError(f"Migration target already exists: {target}")
    print(f"{'PLAN' if dry_run else 'MOVE'}: {source} -> {target}")
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))


def rename_feature_files(dry_run: bool) -> None:
    """Rename selected feature CSVs and their audit reports."""
    for split in ("train", "val", "test"):
        rename_pair(
            FEATURE_DIR / f"dataset_optimized_v2_{split}.csv",
            FEATURE_DIR / f"dataset_selected_{split}.csv",
            dry_run,
        )
    rename_pair(
        FEATURE_DIR / "optimized_v2_feature_selection.csv",
        FEATURE_DIR / "selected_feature_selection.csv",
        dry_run,
    )
    rename_pair(
        FEATURE_DIR / "optimized_v2_selection_report.json",
        FEATURE_DIR / "selected_selection_report.json",
        dry_run,
    )


def rename_model_files(dry_run: bool) -> None:
    """Rename selected staged model artifacts."""
    for path in MODEL_DIR.glob("**/*optimized_v2*.pkl"):
        rename_pair(path, path.with_name(path.name.replace("optimized_v2", "selected")), dry_run)


def rewrite_json(path: Path, dry_run: bool) -> None:
    """Replace legacy mode labels in a JSON report or manifest."""
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    updated = _replace_mode_labels(payload)
    if updated == payload:
        return
    print(f"{'PLAN' if dry_run else 'UPDATE'}: {path}")
    if not dry_run:
        path.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _replace_mode_labels(value):
    """Recursively replace legacy mode labels in JSON-compatible values."""
    if isinstance(value, dict):
        return {
            "selected" if key == "optimized_v2" else key: _replace_mode_labels(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_mode_labels(item) for item in value]
    return "selected" if value == "optimized_v2" else value


def rewrite_metrics(dry_run: bool) -> None:
    """Replace the legacy mode column in the staged metrics CSV."""
    path = MODEL_DIR / "retraining_metrics.csv"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    updated = text.replace(",optimized_v2,", ",selected,")
    if updated == text:
        return
    print(f"{'PLAN' if dry_run else 'UPDATE'}: {path}")
    if not dry_run:
        path.write_text(updated, encoding="utf-8")


def migrate_manifest(dry_run: bool) -> None:
    """Move the active v4 manifest into the versioned manifest directory."""
    rename_pair(MANIFEST_SOURCE, MANIFEST_TARGET, dry_run)
    rewrite_json(MANIFEST_TARGET, dry_run)


def main() -> None:
    """Run the active-mode migration without touching historical versions."""
    args = parse_args()
    migrate_manifest(args.dry_run)
    rename_feature_files(args.dry_run)
    rename_model_files(args.dry_run)
    rewrite_json(FEATURE_DIR / "selected_selection_report.json", args.dry_run)
    rewrite_json(MODEL_DIR / "retraining_report.json", args.dry_run)
    rewrite_metrics(args.dry_run)


if __name__ == "__main__":
    main()
