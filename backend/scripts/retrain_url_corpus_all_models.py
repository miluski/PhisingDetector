"""Train corrected URL-corpus models into a staged directory.

This script never overwrites ``backend/models``. It consumes predefined
domain-disjoint train/validation/test CSVs, calibrates one decision threshold
on validation data, and writes model payloads plus a machine-readable report.
Run later on a machine with enough CPU/RAM; this module is intentionally not
executed by the application startup path.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import tldextract
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config.feature_policy import (
    FEATURE_POLICY_VERSION,
    MISSING_INDICATORS,
    MISSING_POLICY,
    NETWORK_MODE,
    WHOIS_ENABLED,
    WHOIS_FEATURES,
)
from config.feature_modes import CANONICAL_MODES, SUPPORTED_MODES, normalize_mode
from config.feature_names_config import FeatureConfig
from utils.benign_url_categories import (
    BENIGN_CATEGORY_PATHS,
    classify_benign_category,
    count_benign_categories,
    is_calibration_url,
)

DEFAULT_FEATURE_DIR = BACKEND_ROOT / "data" / "url_corpus" / "quality_v4_repaired_features"
DEFAULT_OUTPUT_DIR = BACKEND_ROOT / "models" / "staged_url_corpus_v4"
TLD_EXTRACTOR = tldextract.TLDExtract(
    cache_dir=None,
    fallback_to_snapshot=True,
    suffix_list_urls=(),
)
MODELS = ("random_forest", "xgboost", "mlp", "svm", "logistic_regression")
MODES = CANONICAL_MODES
METADATA_COLUMNS = {"source_index", "url", "registered_domain", "missing_whois"}
FORBIDDEN_FEATURES = {"asn_ip"}


def artifact_filename(model_name: str, mode: str) -> str:
    """Return the staged artifact filename for a model and feature mode."""
    mode = normalize_mode(mode)
    if model_name in {"xgboost", "mlp", "svm"}:
        return f"{model_name}_model_{mode}_url_corpus_corrected.pkl"
    return f"{model_name}_{mode}_url_corpus_corrected.pkl"


def registered_domain(url: str) -> str:
    """Extract a registrable domain for split-leakage checks."""
    value = str(url or "")
    candidate = value if "://" in value else f"http://{value}"
    try:
        host = (urlsplit(candidate).hostname or "").strip(".").lower()
    except ValueError:
        host = candidate.split("//", 1)[-1].split("/", 1)[0].rsplit("@", 1)[-1]
        host = host.strip(".").lower()
    extracted = TLD_EXTRACTOR(host)
    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}".lower()
    return host


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Validate policy-sensitive columns and return model feature names."""
    forbidden = FORBIDDEN_FEATURES & set(frame.columns)
    if forbidden:
        raise ValueError(
            "Stale feature dataset contains forbidden legacy/network features: "
            f"{sorted(forbidden)}. Rebuild features with current policy."
        )
    if WHOIS_ENABLED:
        missing_whois = set(WHOIS_FEATURES) - set(frame.columns)
        if missing_whois:
            raise ValueError(
                "WHOIS opt-in is enabled, but feature CSV lacks RDAP/WHOIS columns: "
                f"{sorted(missing_whois)}. Re-extract all splits with URL_WHOIS_ENABLED=1."
            )
    return [
        column
        for column in frame.columns
        if column not in METADATA_COLUMNS and column != "phishing"
    ]


def load_frames(
    feature_dir: Path, mode: str, minimum_category_samples: int = 100
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Load and validate the three predefined split files for one mode."""
    frames = _read_mode_frames(feature_dir, mode)
    columns = feature_columns(frames["train"])
    _validate_mode_schema(frames, mode, columns)
    _validate_category_coverage(frames, minimum_category_samples)
    return frames, columns


def _read_mode_frames(feature_dir, mode):
    """Read train, validation and test CSVs for one feature mode."""
    mode = normalize_mode(mode)
    return {
        split: pd.read_csv(feature_dir / f"dataset_{mode}_{split}.csv")
        for split in ("train", "val", "test")
    }


def _validate_mode_schema(frames, mode, columns):
    """Ensure split columns and values match the shared feature policy."""
    expected = set(FeatureConfig.get_feature_names(mode))
    actual = set(columns)
    _validate_expected_columns(mode, expected, actual, columns)
    _validate_missing_indicators(mode, columns)
    _validate_split_frames(frames, mode, columns)


def _validate_expected_columns(mode, expected, actual, columns):
    """Validate mode schema and non-empty feature selection."""
    if actual != expected:
        raise ValueError(f"{mode} dataset does not match the shared feature schema: "
                         f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    if not columns:
        raise ValueError(f"No feature columns found for mode={mode}")


def _validate_missing_indicators(mode, columns):
    """Require all explicit missing-value indicators in the dataset."""
    missing = set(MISSING_INDICATORS) - set(columns)
    if missing:
        raise ValueError(
            f"{mode} dataset lacks missing indicators: {sorted(missing)}. "
            "Rebuild all feature CSVs before training."
        )


def _validate_split_frames(frames, mode, columns):
    """Ensure every split has identical schema and no NaNs."""
    for split, frame in frames.items():
        if feature_columns(frame) != columns:
            raise ValueError(f"Feature schema mismatch in {mode}/{split}")
        if frame[columns].isna().any().any():
            raise ValueError(f"NaN values found in {mode}/{split}")


def _validate_category_coverage(frames, minimum):
    """Require every split to contain enough benign examples per category."""
    if minimum < 0:
        raise ValueError("minimum_category_samples must be non-negative")
    failures = {}
    for split, frame in frames.items():
        counts = count_benign_categories(frame.to_dict("records"))
        missing = {category: count for category, count in counts.items() if count < minimum}
        if missing:
            failures[split] = missing
    if failures:
        raise ValueError(f"Insufficient benign category coverage: {failures}")


def verify_group_disjoint(frames: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Verify that registered domains do not cross split boundaries."""
    groups: dict[str, set[str]] = {}
    for split, frame in frames.items():
        groups[split] = {registered_domain(url) for url in frame["url"]}
    overlaps = {
        f"{left}_{right}": len(groups[left] & groups[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    if any(overlaps.values()):
        raise ValueError(f"Registered-domain overlap detected: {overlaps}")
    return {split: len(values) for split, values in groups.items()}


def is_root_url(url: str) -> bool:
    """Return whether a URL points to a domain root without query data."""
    try:
        parsed = urlsplit(str(url or ""))
    except ValueError:
        # Malformed Unicode/IPv6-like hosts still need a deterministic metric result.
        return False
    return parsed.path in {"", "/"} and not parsed.query and not parsed.fragment


def metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float, urls: pd.Series) -> dict[str, Any]:
    """Compute classification, AUC and hard-negative metrics."""
    predictions = (probabilities >= threshold).astype(int)
    benign_roots, hard_negatives = _metric_masks(y_true, urls)
    category_results = _category_metrics(y_true, predictions, urls)
    return {
        "threshold": float(threshold),
        **_classification_metrics(y_true, predictions, probabilities),
        **_hard_negative_metrics(predictions, benign_roots, hard_negatives),
        **_category_metric_payload(category_results),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    }


def _classification_metrics(y_true, predictions, probabilities):
    """Return general thresholded and ranking metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, probabilities)),
    }


def _hard_negative_metrics(predictions, benign_roots, hard_negatives):
    """Return root and hard-negative false-positive rates."""
    return {
        "benign_root_false_positive_rate": _mask_rate(predictions, benign_roots),
        "benign_root_samples": int(benign_roots.sum()),
        "benign_hard_negative_false_positive_rate": _mask_rate(predictions, hard_negatives),
        "benign_hard_negative_samples": int(hard_negatives.sum()),
    }


def _category_metric_payload(category_results):
    """Return category false-positive rates and sample counts."""
    return {
        "benign_category_false_positive_rate": {
            category: item["false_positive_rate"] for category, item in category_results.items()
        },
        "benign_category_samples": {
            category: item["samples"] for category, item in category_results.items()
        },
    }


def _metric_masks(y_true, urls):
    """Return benign-root and benign-hard-negative masks."""
    labels = np.asarray(y_true) == 0
    return labels & urls.map(is_root_url).to_numpy(), labels & urls.map(is_hard_negative_url).to_numpy()


def _mask_rate(predictions, mask):
    """Return positive prediction rate for a mask."""
    return float(predictions[mask].mean()) if mask.any() else 0.0


def _category_metrics(y_true, predictions, urls):
    """Calculate false-positive rate and sample count for each benign category."""
    labels = np.asarray(y_true) == 0
    results = {}
    for category in sorted(BENIGN_CATEGORY_PATHS):
        mask = labels & urls.map(
            lambda url: classify_benign_category(url) == category
        ).to_numpy()
        results[category] = {
            "false_positive_rate": _mask_rate(predictions, mask),
            "samples": int(mask.sum()),
        }
    return results


def is_hard_negative_url(url: str) -> bool:
    """Identify benign-looking URL paths used for threshold calibration."""
    return is_calibration_url(url)


def calibrate_threshold(frame: pd.DataFrame, probabilities: np.ndarray) -> float:
    """Choose a validation threshold constrained by root and hard-negative FPR."""
    candidates = _threshold_candidates(probabilities)
    masks = _calibration_masks(frame)
    acceptable, fallback = _rank_thresholds(frame, probabilities, candidates, masks)
    if not fallback:
        return 0.5
    return float(-max(acceptable or fallback)[2])


def _threshold_candidates(probabilities):
    """Return deterministic threshold candidates for calibration."""
    return np.unique(np.concatenate([np.linspace(0.05, 0.95, 181), probabilities]))


def _calibration_masks(frame):
    """Build root and hard-negative masks for a validation frame."""
    labels = frame["phishing"].to_numpy()
    urls = frame["url"].map
    return (
        (labels == 0) & urls(is_root_url).to_numpy(),
        (labels == 0) & urls(is_hard_negative_url).to_numpy(),
        _category_masks(labels, frame["url"]),
    )


def _category_masks(labels, urls):
    """Build benign masks for every category used in threshold calibration."""
    return {
        category: (labels == 0) & urls.map(
            lambda url: classify_benign_category(url) == category
        ).to_numpy()
        for category in sorted(BENIGN_CATEGORY_PATHS)
    }


def _rank_thresholds(frame, probabilities, candidates, masks):
    """Rank thresholds and keep candidates satisfying hard-negative limits."""
    root_mask, hard_negative_mask, category_masks = masks
    if not hard_negative_mask.any():
        raise ValueError("Validation set has no benign hard negatives")
    acceptable, fallback = [], []
    for threshold in candidates:
        item, root_fpr, hard_fpr, category_fprs = _threshold_item(
            frame, probabilities, threshold, root_mask, hard_negative_mask, category_masks
        )
        fallback.append(item)
        if root_fpr <= 0.10 and hard_fpr <= 0.10 and all(
            value <= 0.10 for value in category_fprs.values()
        ):
            acceptable.append(item)
    return acceptable, fallback


def _threshold_item(frame, probabilities, threshold, root_mask, hard_mask, category_masks):
    """Return ranking tuple and FPRs for one threshold."""
    prediction = (probabilities >= threshold).astype(int)
    root_fpr = float(prediction[root_mask].mean()) if root_mask.any() else 0.0
    hard_fpr = float(prediction[hard_mask].mean())
    category_fprs = {
        category: _mask_rate(prediction, mask)
        for category, mask in category_masks.items() if mask.any()
    }
    score = float(f1_score(frame["phishing"], prediction, zero_division=0))
    recall = float(recall_score(frame["phishing"], prediction, zero_division=0))
    return (score, recall, -float(threshold)), root_fpr, hard_fpr, category_fprs


def build_model(model_name: str, seed: int = 42):
    """Create the configured estimator for one model name."""
    builders = {
        "random_forest": _random_forest,
        "xgboost": _xgboost,
        "logistic_regression": _logistic_regression,
        "svm": _svm,
    }
    if model_name not in builders:
        raise ValueError(f"Unknown estimator: {model_name}")
    return builders[model_name](seed)


def _random_forest(seed):
    """Build the random forest estimator."""
    return RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced", random_state=seed, n_jobs=-1,
    )


def _xgboost(seed):
    """Build the XGBoost estimator."""
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.08,
        min_child_weight=1, subsample=0.85, colsample_bytree=0.85,
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        random_state=seed, n_jobs=-1,
    )
def _logistic_regression(seed):
    """Build the logistic regression estimator."""
    return LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=seed)


def _svm(seed):
    """Build the probabilistic SVM estimator."""
    return SVC(C=10.0, kernel="rbf", gamma="scale", probability=True, random_state=seed)


def train_mlp(X_train, y_train, X_val, y_val, seed: int):
    """Train the Keras MLP with deterministic early stopping."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    model = _build_mlp(tf, X_train.shape[1])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100,
              batch_size=256, callbacks=_mlp_callbacks(tf), verbose=2)
    return model


def _build_mlp(tf, feature_count):
    """Create the configured MLP architecture."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(feature_count,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
    return model


def _mlp_callbacks(tf):
    """Return deterministic validation-based MLP callbacks."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=12, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max", patience=5, factor=0.5, min_lr=1e-7
        ),
    ]


def probabilities(model_name: str, model, scaler, frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Return positive-class probabilities for a fitted model payload."""
    values = frame[columns]
    if scaler is not None:
        values = scaler.transform(values)
    if model_name == "mlp":
        return np.asarray(model.predict(values, verbose=0)).reshape(-1)
    return np.asarray(model.predict_proba(values)[:, 1]).reshape(-1)


def _training_frames(model_name, frames, columns):
    """Prepare train and validation matrices, scaling models that need it."""
    train = frames["train"]
    val = frames["val"]
    scaler = None
    X_train, X_val = train[columns], val[columns]
    if model_name in {"logistic_regression", "svm", "mlp"}:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
    return train, val, scaler, X_train, X_val


def _fit_variant(model_name, X_train, y_train, X_val, y_val, seed):
    """Fit one configured estimator and return elapsed training seconds."""
    started = time.perf_counter()
    if model_name == "mlp":
        model = train_mlp(X_train, y_train, X_val, y_val, seed)
    else:
        model = build_model(model_name, seed)
        model.fit(X_train, y_train)
    return model, time.perf_counter() - started


def _build_payload(model_name, mode, model, scaler, columns, threshold, val_metrics, test_metrics, seconds):
    """Build the serialized model payload."""
    params = model.get_config() if model_name == "mlp" else model.get_params()
    return {
        "model": model, "scaler": scaler, "feature_names": columns,
        "best_params": params, "decision_threshold": threshold, "metrics": test_metrics,
        "validation_metrics": val_metrics, "training_seconds": seconds,
        "dataset": "url_corpus_corrected", "mode": mode,
        "feature_policy": FEATURE_POLICY_VERSION, "network_mode": NETWORK_MODE,
        "missing_policy": MISSING_POLICY,
    }


def _write_payload(payload, output_dir, model_name, mode):
    """Persist one model payload in its staged model directory."""
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / artifact_filename(model_name, mode)
    with path.open("wb") as file:
        pickle.dump(payload, file)


def _quality_gate(test_metrics):
    """Return whether test metrics satisfy the minimum deployment gate."""
    category_rates = test_metrics["benign_category_false_positive_rate"].values()
    return bool(test_metrics["auc_roc"] >= 0.80 and test_metrics["recall"] >= 0.60
                and test_metrics["benign_root_false_positive_rate"] <= 0.20
                and all(rate <= 0.20 for rate in category_rates))


def train_variant(
    model_name: str, mode: str, frames: dict[str, pd.DataFrame],
    columns: list[str], output_dir: Path, seed: int,
) -> dict[str, Any]:
    """Train, calibrate, evaluate and stage one model/mode variant."""
    train, val, scaler, X_train, X_val = _training_frames(model_name, frames, columns)
    model, training_seconds = _fit_variant(model_name, X_train, train["phishing"].to_numpy(),
                                           X_val, val["phishing"].to_numpy(), seed)
    val_probabilities = probabilities(model_name, model, scaler, val, columns)
    threshold = calibrate_threshold(val, val_probabilities)
    val_metrics = metrics(val["phishing"], val_probabilities, threshold, val["url"])
    test = frames["test"]
    test_probabilities = probabilities(model_name, model, scaler, test, columns)
    test_metrics = metrics(test["phishing"], test_probabilities, threshold, test["url"])
    _write_payload(_build_payload(model_name, mode, model, scaler, columns, threshold,
                                  val_metrics, test_metrics, training_seconds), output_dir, model_name, mode)
    return _report_row(model_name, mode, columns, threshold, training_seconds, val_metrics, test_metrics)


def _report_row(model_name, mode, columns, threshold, seconds, val_metrics, test_metrics):
    """Build the machine-readable report row for one variant."""
    return {
        "model": model_name, "mode": mode, "features": len(columns),
        "training_seconds": seconds, "decision_threshold": threshold,
        "validation": val_metrics, "test": test_metrics,
        "quality_gate_passed": _quality_gate(test_metrics),
    }


def parse_args() -> argparse.Namespace:
    """Parse retraining command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--modes", nargs="+", choices=SUPPORTED_MODES, default=list(MODES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-category-samples", type=int, default=100)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite-staged", action="store_true")
    return parser.parse_args()


def _validate_output_dir(output_dir, overwrite):
    """Ensure staged output is not accidentally overwritten."""
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise SystemExit(f"Output exists: {output_dir}. Use a new path or --overwrite-staged.")
    output_dir.mkdir(parents=True, exist_ok=True)


def _train_requested(args):
    """Train all requested modes and models and collect split reports."""
    reports, groups = [], {}
    for mode in args.modes:
        frames, columns = load_frames(args.feature_dir, mode, args.minimum_category_samples)
        groups[mode] = verify_group_disjoint(frames)
        for model_name in args.models:
            print(f"\nTRAIN {model_name}/{mode}: {len(frames['train']):,} rows, {len(columns)} features")
            reports.append(train_variant(model_name, mode, frames, columns, args.output_dir, args.seed))
    return reports, groups


def _validate_requested(args):
    """Validate requested feature modes without fitting any model."""
    for mode in args.modes:
        frames, columns = load_frames(args.feature_dir, mode, args.minimum_category_samples)
        groups = verify_group_disjoint(frames)
        print(f"CHECK {mode}: {len(columns)} cech, domeny={groups}")


def _write_reports(args, reports, groups):
    """Write JSON and CSV retraining reports."""
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_dir": str(args.feature_dir.resolve()), "output_dir": str(args.output_dir.resolve()),
        "group_disjoint_counts": groups, "models": reports,
        "all_quality_gates_passed": all(item["quality_gate_passed"] for item in reports),
        "production_models_changed": False, "feature_policy": FEATURE_POLICY_VERSION,
        "network_mode": NETWORK_MODE, "missing_policy": MISSING_POLICY,
    }
    (args.output_dir / "retraining_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_metrics_csv(args.output_dir, reports)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _write_metrics_csv(output_dir, reports):
    """Write the compact metrics CSV consumed by comparison tooling."""
    fields = ["model", "mode", "features", "training_seconds", "decision_threshold", "quality_gate_passed"]
    with (output_dir / "retraining_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: item[key] for key in fields} for item in reports)


def main() -> None:
    """Coordinate staged retraining and report generation."""
    args = parse_args()
    if args.check_only:
        _validate_requested(args)
        return
    _validate_output_dir(args.output_dir, args.overwrite_staged)
    reports, groups = _train_requested(args)
    _write_reports(args, reports, groups)


if __name__ == "__main__":
    main()
