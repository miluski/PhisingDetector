"""Retrain staged URL-corpus Random Forest models with quality gates."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from utils.split_dataset_loader import feature_columns
from config.feature_policy import FEATURE_POLICY_VERSION, MISSING_INDICATORS, MISSING_POLICY, NETWORK_MODE


UNSTABLE_NETWORK_FEATURES = {
    "time_response", "domain_spf", "time_domain_activation",
    "time_domain_expiration", "qty_ip_resolved", "qty_nameservers",
    "qty_mx_servers", "ttl_hostname", "tls_ssl_certificate", "qty_redirects",
}


def is_root_url(value: str) -> bool:
    """Return whether a URL is a domain root without query or fragment."""
    parsed = urlsplit(str(value or ""))
    return parsed.path in {"", "/"} and not parsed.query and not parsed.fragment


def evaluate(
    model, frame: pd.DataFrame, columns: list[str], threshold: float = 0.5
) -> dict[str, float]:
    """Calculate classification and benign-root metrics for a frame."""
    X = frame[columns]
    y = frame["phishing"]
    probability = model.predict_proba(X)[:, 1]
    prediction = probability > threshold
    benign_roots = (frame["phishing"] == 0) & frame["url"].map(is_root_url)
    root_fpr = float((prediction[benign_roots] == 1).mean()) if benign_roots.any() else 0.0
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction)),
        "recall": float(recall_score(y, prediction)),
        "f1": float(f1_score(y, prediction)),
        "auc_roc": float(roc_auc_score(y, probability)),
        "benign_root_false_positive_rate": root_fpr,
        "benign_root_samples": int(benign_roots.sum()),
    }


def calibrate_threshold(model, frame: pd.DataFrame, columns: list[str]) -> float:
    """Choose the highest-recall validation threshold under root FPR limit."""
    probability = model.predict_proba(frame[columns])[:, 1]
    y = frame["phishing"].to_numpy()
    benign_roots = (frame["phishing"] == 0) & frame["url"].map(is_root_url)
    candidates = []
    for threshold in np.linspace(0.01, 0.5, 200):
        prediction = probability > threshold
        root_fpr = float(prediction[benign_roots].mean())
        if root_fpr <= 0.10:
            candidates.append((recall_score(y, prediction), float(threshold)))
    if not candidates:
        raise ValueError("No validation threshold satisfies benign root FPR <= 0.10")
    return max(candidates)[1]


def train_variant(feature_dir, model_dir, staged_dir, mode):
    """Train one corrected model variant and write its staged payload."""
    current_path, current, train, val, test = _load_variant(feature_dir, model_dir, mode)
    columns = _validated_columns(train)
    model, params = _fit_variant(train, columns)
    threshold = calibrate_threshold(model, val, columns)
    val_metrics, test_metrics = evaluate(model, val, columns, threshold), evaluate(model, test, columns, threshold)
    baseline = evaluate(
        current["model"], test, list(current["feature_names"]),
        float(current.get("decision_threshold", 0.5)),
    )
    passed = _quality_gate(test_metrics, baseline)
    staged_path = _save_variant(
        staged_dir, current_path.name, model, columns, params, threshold,
        val_metrics, test_metrics, baseline, passed,
    )
    return {"mode": mode, "current_model": str(current_path), "staged_model": str(staged_path),
            "baseline_test_metrics": baseline, "validation_metrics": val_metrics,
            "test_metrics": test_metrics, "quality_gate_passed": passed}


def _load_variant(feature_dir, model_dir, mode):
    """Load current payload and three feature splits for one mode."""
    current_path = model_dir / f"random_forest_{mode}_url_corpus.pkl"
    with current_path.open("rb") as file:
        current = pickle.load(file)
    frames = [pd.read_csv(feature_dir / f"dataset_{mode}_{split}.csv") for split in ("train", "val", "test")]
    return current_path, current, *frames


def _validated_columns(train):
    """Validate policy-sensitive columns and return stable model features."""
    if "asn_ip" in train.columns:
        raise ValueError("Stale feature dataset contains forbidden columns: ['asn_ip']")
    columns = [column for column in feature_columns(train) if column not in UNSTABLE_NETWORK_FEATURES]
    missing = set(MISSING_INDICATORS) - set(columns)
    if missing:
        raise ValueError(f"Missing indicators absent from dataset: {missing}")
    return columns


def _fit_variant(train, columns):
    """Fit the corrected random forest and return its parameters."""
    params = {"n_estimators": 500, "max_depth": None, "max_features": "sqrt",
              "min_samples_leaf": 2, "min_samples_split": 2, "class_weight": {0: 1, 1: 4}}
    model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    model.fit(train[columns], train["phishing"])
    return model, params


def _quality_gate(metrics, baseline):
    """Compare corrected metrics with the current production baseline."""
    return bool(metrics["auc_roc"] >= baseline["auc_roc"] + 0.05 and metrics["recall"] >= baseline["recall"]
                and metrics["accuracy"] >= baseline["accuracy"] and metrics["benign_root_false_positive_rate"] <= 0.12
                and metrics["benign_root_false_positive_rate"] <= baseline["benign_root_false_positive_rate"] - 0.50)


def _save_variant(staged_dir, filename, model, columns, params, threshold, val_metrics, test_metrics, baseline, passed):
    """Serialize one staged corrected random-forest payload."""
    payload = {"model": model, "scaler": None, "feature_names": columns, "best_params": params,
               "decision_threshold": threshold, "metrics": test_metrics, "validation_metrics": val_metrics,
               "baseline_test_metrics": baseline, "quality_gate_passed": passed,
               "feature_policy": FEATURE_POLICY_VERSION, "network_mode": NETWORK_MODE, "missing_policy": MISSING_POLICY}
    staged_dir.mkdir(parents=True, exist_ok=True)
    path = staged_dir / filename
    with path.open("wb") as file:
        pickle.dump(payload, file)
    return path


def parse_args() -> argparse.Namespace:
    """Parse corrected random-forest retraining options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument(
        "--model-dir", type=Path,
        default=BACKEND_DIR / "models" / "random_forest",
    )
    parser.add_argument("--staged-dir", type=Path, required=True)
    parser.add_argument("--promote", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Retrain requested modes, optionally promote passing artifacts."""
    args = parse_args()
    reports = [train_variant(args.feature_dir, args.model_dir, args.staged_dir, mode) for mode in ("full", "optimized")]
    all_passed = all(report["quality_gate_passed"] for report in reports)
    if args.promote and all_passed:
        for report in reports:
            shutil.copy2(report["staged_model"], report["current_model"])
            report["promoted"] = True
    elif args.promote:
        for report in reports:
            report["promoted"] = False

    output = {"all_quality_gates_passed": all_passed, "variants": reports}
    report_path = args.staged_dir / "quality_report.json"
    report_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
