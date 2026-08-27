"""Select an URL-corpus feature subset using the Grabarica procedure.

The selector is deliberately fit on the training split only.  The resulting
manifest is then applied unchanged to validation, test, production extraction
and model training.  This prevents target/test leakage and makes the selected
schema reproducible.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config.feature_names_config import FeatureConfig
from config.feature_policy import AUDIT_INDICATORS, FEATURE_POLICY_VERSION
from config.feature_modes import CANONICAL_MODES


SPLITS = ("train", "val", "test")
METADATA_COLUMNS = ("source_index", "url", *AUDIT_INDICATORS)
LABEL_COLUMN = "phishing"


def select_features(train: pd.DataFrame, seed: int) -> tuple[list[str], dict[str, object]]:
    """Return selected features and an auditable report.

    The thresholds mirror the historical Grabarica notebook.  A truly
    constant feature is always removed; other candidates require at least two
    independent elimination reasons.
    """
    feature_names = _full_feature_names(train)
    x, y = train[feature_names].apply(pd.to_numeric, errors="raise"), train[LABEL_COLUMN].astype(int)
    importance = _fit_importance(x, y, feature_names, seed)
    reasons, zero_variance, near_zero_variance, weak_target, low_importance, target_correlation = _base_reasons(
        x, y, importance
    )
    high_correlation_pairs = _correlation_reasons(x, feature_names, importance, reasons)
    selected = _selected_features(feature_names, reasons, zero_variance)
    if not selected:
        raise ValueError("Feature selection removed every feature")
    return selected, _selection_report(feature_names, selected, importance, target_correlation, reasons,
                                       zero_variance, near_zero_variance, low_importance,
                                       weak_target, high_correlation_pairs)


def _full_feature_names(train):
    """Validate and return the canonical full feature schema."""
    names = FeatureConfig.get_feature_names("full")
    missing = sorted(set(names) - set(train.columns))
    if missing:
        raise ValueError(f"Training CSV lacks full-schema features: {missing}")
    return names


def _fit_importance(x, y, feature_names, seed):
    """Fit the selector forest and return feature importances."""
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                                   class_weight="balanced", random_state=seed, n_jobs=-1)
    model.fit(x, y)
    return pd.Series(model.feature_importances_, index=feature_names, dtype=float)


def _base_reasons(x, y, importance):
    """Calculate variance, target-correlation and importance reasons."""
    variance = x.var()
    names = list(x.columns)
    reasons = {feature: set() for feature in names}
    zero, near_zero = _variance_reasons(variance, reasons)
    target, weak = _target_reasons(x, y, variance, reasons)
    low = importance[importance < 0.001].index.tolist()
    for feature in low:
        reasons[feature].add("low_random_forest_importance")
    return reasons, zero, near_zero, weak, low, target


def _variance_reasons(variance, reasons):
    """Mark constant and near-constant features."""
    zero = variance[variance == 0].index.tolist()
    near_zero = variance[(variance > 0) & (variance < 0.01)].index.tolist()
    for feature in zero:
        reasons[feature].add("zero_variance")
    for feature in near_zero:
        reasons[feature].add("near_zero_variance")
    return zero, near_zero


def _target_reasons(x, y, variance, reasons):
    """Calculate target correlations without constant-column warnings."""
    variable = variance[variance > 0].index
    target = pd.Series(0.0, index=x.columns)
    target.loc[variable] = x[variable].corrwith(y).fillna(0.0)
    weak = target.abs()[target.abs() < 0.05].index.tolist()
    for feature in weak:
        reasons[feature].add("weak_target_correlation")
    return target, weak


def _correlation_reasons(x, feature_names, importance, reasons):
    """Mark the less important member of highly correlated feature pairs."""
    variable = [feature for feature in feature_names if x[feature].nunique(dropna=False) > 1]
    matrix = x[variable].corr()
    pairs = []
    for index, left in enumerate(feature_names):
        if left not in matrix.index:
            continue
        for right in feature_names[index + 1:]:
            if right not in matrix.columns:
                continue
            correlation = matrix.at[left, right]
            if pd.notna(correlation) and abs(correlation) > 0.95:
                remove = left if importance[left] < importance[right] else right
                reasons[remove].add("high_pairwise_correlation")
                pairs.append({
                    "left": left, "right": right,
                    "correlation": float(correlation), "removed": remove,
                })
    return pairs


def _selected_features(feature_names, reasons, zero_variance):
    """Keep features that are not constant and lack multiple removal reasons."""
    return [feature for feature in feature_names if feature not in zero_variance and len(reasons[feature]) < 2]


def _selection_report(feature_names, selected, importance, target, reasons, zero, near_zero, low, weak, pairs):
    """Build an auditable feature-selection report."""
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_split": "train_only",
        "input_features": len(feature_names),
        "selected_features": len(selected),
        "removed_features": len(feature_names) - len(selected),
        "thresholds": _selection_thresholds(),
        "zero_variance": zero,
        "near_zero_variance": near_zero,
        "low_importance": low,
        "weak_target_correlation": weak,
        "high_correlation_pairs": pairs,
        "features": _feature_report_entries(feature_names, selected, importance, target, reasons),
    }
    return report


def _selection_thresholds():
    """Return the fixed elimination thresholds used by the selector."""
    return {"near_zero_variance": 0.01, "pairwise_correlation": 0.95,
            "random_forest_importance": 0.001, "target_correlation": 0.05,
            "minimum_elimination_reasons": 2}


def _feature_report_entries(feature_names, selected, importance, target, reasons):
    """Build per-feature importance and elimination entries."""
    return [{
        "name": feature,
        "random_forest_importance": float(importance[feature]),
        "target_correlation": float(target[feature]),
        "elimination_reasons": sorted(reasons[feature]),
        "selected": feature in selected,
    } for feature in feature_names]


def write_selected_splits(
    full_dir: Path,
    output_dir: Path,
    selected: list[str],
    mode: str,
) -> dict[str, int]:
    """Apply the selected manifest to every predefined split."""
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    columns = [*METADATA_COLUMNS, *selected, LABEL_COLUMN]
    for split in SPLITS:
        source = full_dir / f"dataset_full_{split}.csv"
        target = output_dir / f"dataset_{mode}_{split}.csv"
        frame = pd.read_csv(source)
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise ValueError(f"{source} lacks selected-schema columns: {missing}")
        frame[columns].to_csv(target, index=False)
        counts[split] = len(frame)
    return counts


def parse_args() -> argparse.Namespace:
    """Parse feature selection options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=(BACKEND_ROOT / "config" / "feature_manifests" / "url_corpus"
                 / "quality_v4" / "selected.json"),
    )
    parser.add_argument("--mode", choices=CANONICAL_MODES, default="selected")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Select features from train only and write manifest plus split files."""
    args = parse_args()
    train_path = args.full_dir / "dataset_full_train.csv"
    train = pd.read_csv(train_path)
    selected, report = select_features(train, args.seed)
    report["mode"], report["feature_policy"], report["feature_names"] = args.mode, FEATURE_POLICY_VERSION, selected
    _write_manifest(args.manifest_path, args.mode, selected, report)
    counts = write_selected_splits(args.full_dir, args.output_dir, selected, args.mode)
    report["rows"] = counts
    report_path = args.output_dir / f"{args.mode}_selection_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame(report["features"]).to_csv(
        args.output_dir / f"{args.mode}_feature_selection.csv", index=False
    )
    print(json.dumps({"mode": args.mode, "features": len(selected), "rows": counts,
                      "manifest": str(args.manifest_path)}, ensure_ascii=False))


def _write_manifest(path, mode, selected, report):
    """Persist selected feature names and their audit report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"mode": mode, "feature_policy": FEATURE_POLICY_VERSION,
               "features": selected, "selection": report}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
