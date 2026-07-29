"""Recheck staged URL-corpus models on grouped, external and real URL cases."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.url_feature_service import URLFeatureService
from config.project_paths import RESULTS_ROOT
from config.feature_names_config import FeatureConfig
from config.feature_policy import FEATURE_POLICY_VERSION, MISSING_POLICY, MISSING_VALUE, NETWORK_MODE
from config.feature_modes import CANONICAL_MODES, SUPPORTED_MODES, normalize_mode


DEFAULT_MODEL_ROOT = BACKEND_ROOT / "models" / "staged_url_corpus_v4"
DEFAULT_FEATURE_DIR = BACKEND_ROOT / "data" / "url_corpus" / "quality_v4_repaired_features"
DEFAULT_EXTERNAL_RAW = (
    BACKEND_ROOT / "data" / "url_corpus" /
    "external_test_large_no_domain_overlap_splits" / "raw_test.csv"
)
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "url_corpus" / "recheck_v4"
MODELS = ("random_forest", "xgboost", "mlp", "svm", "logistic_regression")
MODES = CANONICAL_MODES


def artifact_filename(model_name: str, mode: str) -> str:
    """Return the staged artifact filename for one model variant."""
    mode = normalize_mode(mode)
    if model_name in {"xgboost", "mlp", "svm"}:
        return f"{model_name}_model_{mode}_url_corpus_corrected.pkl"
    return f"{model_name}_{mode}_url_corpus_corrected.pkl"


def production_artifact_filename(model_name: str, mode: str) -> str:
    """Return the artifact filename written by URL-corpus notebooks."""
    mode = normalize_mode(mode)
    if model_name in {"xgboost", "mlp", "svm"}:
        return f"{model_name}_model_{mode}_url_corpus.pkl"
    return f"{model_name}_{mode}_url_corpus.pkl"


def load_payload(model_root: Path, model_name: str, mode: str) -> dict[str, Any]:
    """Load and validate a staged or notebook-produced model payload."""
    path = _resolve_artifact_path(model_root, model_name, mode)
    with path.open("rb") as file:
        payload = pickle.load(file)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Model artifact has no calibrated payload: {path}")
    return payload


def _resolve_artifact_path(model_root, model_name, mode):
    """Resolve corrected staged and production notebook artifact names."""
    directory = model_root / model_name
    candidates = (
        directory / artifact_filename(model_name, mode),
        directory / production_artifact_filename(model_name, mode),
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing model artifact; checked: {candidates}")


def predict(payload: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    """Predict after aligning external features to the trained schema."""
    columns = list(payload["feature_names"])
    values = align_features(frame, columns)
    scaler = payload.get("scaler")
    if scaler is not None:
        values = scaler.transform(values)
    model = payload["model"]
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(values)[:, 1]).reshape(-1)
    return np.asarray(model.predict(values, verbose=0)).reshape(-1)


def align_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add unavailable extractor features using the shared missing-value policy."""
    values = frame.copy()
    for column in columns:
        if column not in values:
            values[column] = MISSING_VALUE
    return values[columns].fillna(MISSING_VALUE)


def evaluate(payload: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    """Calculate thresholded classification metrics for one labelled frame."""
    probabilities = predict(payload, frame)
    threshold = float(payload.get("decision_threshold", 0.5))
    predictions = (probabilities >= threshold).astype(int)
    labels = frame["phishing"].to_numpy()
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    benign_total = int(matrix[0, 0] + matrix[0, 1])
    result = classification_metrics(labels, predictions, matrix, benign_total, threshold, len(frame))
    if len(np.unique(labels)) == 2:
        result["auc_roc"] = float(roc_auc_score(labels, probabilities))
    return result


def classification_metrics(labels, predictions, matrix, benign_total, threshold, rows):
    """Build scalar metrics from predictions and a binary confusion matrix."""
    false_positive_rate = float(matrix[0, 1] / benign_total) if benign_total else 0.0
    return {"threshold": threshold, "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "false_positive_rate": false_positive_rate,
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "specificity": float(matrix[0, 0] / benign_total) if benign_total else None,
            "confusion_matrix": matrix.tolist(), "rows": int(rows)}


def build_external_features(raw_path: Path, output_path: Path, mode: str) -> pd.DataFrame:
    """Extract and cache features for an independent URL file."""
    raw = pd.read_csv(raw_path)
    label_column = "label" if "label" in raw.columns else "phishing"
    service = URLFeatureService(apply_network_defaults=False)
    rows = []
    for index, row in raw.iterrows():
        features = service.extract(row["url"], mode)
        features["url"] = row["url"]
        features["source_index"] = row.get("source_index", index)
        features["phishing"] = int(row[label_column])
        rows.append(features)
        if (index + 1) % 1000 == 0:
            print(f"EXTERNAL FEATURES {mode}: {index + 1:,}/{len(raw):,}")
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def cached_features_are_compatible(frame: pd.DataFrame, payloads: dict[str, dict[str, Any]]) -> bool:
    """Check that a cached feature frame covers every model schema."""
    required = {
        feature
        for payload in payloads.values()
        for feature in payload["feature_names"]
    }
    return required.issubset(frame.columns) and _deterministic_features_valid(
        frame, required
    )


def _deterministic_features_valid(frame, required) -> bool:
    """Reject missing placeholders for features that cannot be unavailable."""
    if "domain_in_ip" not in required:
        return True
    values = pd.to_numeric(frame["domain_in_ip"], errors="coerce")
    return bool(values.isin([0, 1]).all())


REAL_CASES = [
    ("safe", "https://gmail.com/", 0),
    ("safe", "https://github.com/", 0),
    ("safe", "https://paypal.com/", 0),
    ("phishing", "https://bankofamerica-verify.com/secure/login?id=12345&token=abc", 1),
    ("phishing", "https://paypal-security-center.net/secure/login", 1),
    ("phishing", "http://192.0.2.10/secure/login.php?session=reset", 1),
]


def run_real_cases(payload: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    """Evaluate the fixed real-world phishing and benign URL smoke cases."""
    service = URLFeatureService(apply_network_defaults=False)
    columns = list(payload["feature_names"])
    rows = []
    for name, url, expected in REAL_CASES:
        features = service.extract(url, mode)
        frame = pd.DataFrame([features])
        probability = float(predict(payload, frame)[0])
        threshold = float(payload.get("decision_threshold", 0.5))
        rows.append({
            "case": name,
            "url": url,
            "expected": expected,
            "prediction": int(probability >= threshold),
            "phishing_probability": probability,
            "correct": int((probability >= threshold) == bool(expected)),
            "features": len(columns),
        })
    return rows


def parse_args() -> argparse.Namespace:
    """Parse recheck inputs and output locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--external-raw", type=Path, default=DEFAULT_EXTERNAL_RAW)
    _add_output_arguments(parser)
    parser.add_argument(
        "--temporal-raw", type=Path, default=None,
        help="Optional dated, independent-source temporal test CSV",
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--modes", nargs="+", choices=SUPPORTED_MODES, default=list(MODES))
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--skip-real-cases", action="store_true")
    return parser.parse_args()


def _add_output_arguments(parser):
    """Register report and reusable feature-cache output arguments."""
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--external-cache-dir", type=Path, default=None,
        help="Reuse compatible external feature caches independently of report output.",
    )


def _load_mode_inputs(args, mode, external_cache, temporal_cache):
    """Load grouped data, model payloads and optional independent test frames."""
    mode = normalize_mode(mode)
    grouped = pd.read_csv(args.feature_dir / f"dataset_{mode}_test.csv")
    payloads = {name: load_payload(args.model_root, name, mode) for name in args.models}
    validate_grouped_schema(grouped, payloads, mode)
    external = load_optional_frame(args.external_raw, external_cache, mode, payloads, not args.skip_external)
    temporal = load_optional_frame(args.temporal_raw, temporal_cache, mode, payloads, args.temporal_raw is not None)
    return grouped, payloads, external, temporal


def validate_grouped_schema(frame, payloads, mode):
    """Reject a grouped test file that cannot represent a trained model."""
    for name, payload in payloads.items():
        missing = sorted(set(payload["feature_names"]) - set(frame.columns))
        if missing:
            raise ValueError(f"Grouped test schema mismatch for {name}/{mode}: {missing}")


def load_optional_frame(raw_path, cache_dir, mode, payloads, enabled):
    """Load a compatible feature cache or build it from raw URLs."""
    mode = normalize_mode(mode)
    if not enabled or raw_path is None:
        return None
    cache_path = cache_dir / f"{'external' if cache_dir.name == 'external_features' else 'temporal'}_{mode}.csv"
    if cache_path.exists():
        frame = pd.read_csv(cache_path)
        if cached_features_are_compatible(frame, payloads):
            return frame
        print(f"FEATURE CACHE STALE {mode}: rebuilding feature file")
    projected = _project_full_feature_cache(cache_dir, mode, payloads)
    if projected is not None:
        projected.to_csv(cache_path, index=False)
        print(f"FEATURE CACHE PROJECTED FROM FULL {mode}: {cache_path}")
        return projected
    frame = build_external_features(raw_path, cache_path, mode)
    if not cached_features_are_compatible(frame, payloads):
        raise ValueError(f"Feature schema mismatch after rebuild: mode={mode}")
    return frame


def _project_full_feature_cache(cache_dir, mode, payloads):
    """Project a valid full cache onto a smaller model feature schema."""
    if mode == "full":
        return None
    full_path = cache_dir / f"{_cache_prefix(cache_dir)}_full.csv"
    if not full_path.exists():
        return None
    frame = pd.read_csv(full_path)
    if not cached_features_are_compatible(frame, payloads):
        return None
    columns = _projected_cache_columns(frame, mode)
    return frame[columns].copy()


def _cache_prefix(cache_dir: Path) -> str:
    """Return the filename prefix for external or temporal caches."""
    return "external" if cache_dir.name == "external_features" else "temporal"


def _projected_cache_columns(frame, mode):
    """Return metadata, canonical features and label available in a full cache."""
    requested = [
        "url", "source_index", *FeatureConfig.get_feature_names(mode), "phishing"
    ]
    return [column for column in requested if column in frame.columns]


def evaluate_mode(args, mode, external_cache, temporal_cache):
    """Evaluate all selected models for one dataset mode."""
    mode = normalize_mode(mode)
    grouped, payloads, external, temporal = _load_mode_inputs(args, mode, external_cache, temporal_cache)
    metrics, real_rows = [], []
    for name, payload in payloads.items():
        metrics.extend(evaluate_frames(name, mode, payload, grouped, external, temporal))
        if not args.skip_real_cases:
            real_rows.extend({"model": name, "mode": mode, **row} for row in run_real_cases(payload, mode))
    return metrics, real_rows


def evaluate_frames(name, mode, payload, grouped, external, temporal):
    """Evaluate one payload on every available frame."""
    frames = [("corrected_grouped_test", grouped), ("external_no_domain_overlap", external),
              ("temporal_independent_source", temporal)]
    return [{"dataset": label, "model": name, "mode": mode, **evaluate(payload, frame)}
            for label, frame in frames if frame is not None]


def write_metric_files(output_dir, metric_rows, real_rows):
    """Persist CSV metric and real-case reports."""
    if metric_rows:
        path = output_dir / "recheck_metrics.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=sorted({key for row in metric_rows for key in row}))
            writer.writeheader()
            writer.writerows(metric_rows)
    content = "" if not real_rows else pd.DataFrame(real_rows).to_csv(index=False)
    (output_dir / "real_url_cases.csv").write_text(content, encoding="utf-8")


def build_report(args, metric_rows, real_rows):
    """Build the JSON report and quality gates."""
    external = [row for row in metric_rows if row["dataset"] == "external_no_domain_overlap"]
    all_correct = all(row["correct"] for row in real_rows) if real_rows else None
    quality = bool(external) and all(
        row.get("auc_roc", 0) >= 0.80 and row.get("specificity", 0) >= 0.80
        for row in external
    )
    return {"model_root": str(args.model_root.resolve()), "metrics": metric_rows, "real_cases": real_rows,
            "all_real_cases_correct": all_correct, "feature_policy": FEATURE_POLICY_VERSION,
            "network_mode": NETWORK_MODE, "missing_policy": MISSING_POLICY,
            "metric_descriptions": metric_descriptions(),
            "external_quality_gate_passed": quality, "promotion_ready": bool(real_rows and all_correct and quality)}


def metric_descriptions():
    """Return metric definitions stored with every recheck report."""
    return {"false_positive_rate": "FP/(TN+FP) among benign URLs.",
            "specificity": "TN/(TN+FP) among benign URLs.",
            "recall": "TP/(TP+FN) among phishing URLs.",
            "interpretation": "FPR and recall depend on the decision threshold; AUC evaluates ranking.",}


def write_summary(output_dir, report):
    """Write a concise Polish interpretation of external metrics."""
    rows = [row for row in report["metrics"] if row["dataset"] == "external_no_domain_overlap"]
    lines = ["# URL Corpus Model Recheck Summary", "",
             "FPR is the share of benign URLs incorrectly classified as phishing: "
             "`FP / (TN + FP)`.", "",
             "| Model | Mode | AUC-ROC | Recall | FPR | Specificity | F1 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    lines.extend(format_summary_row(row) for row in rows)
    lines.extend(["", "## Final assessment", "",
                  "- External quality gate: "
                  f"**{_status(report['external_quality_gate_passed'], 'passed', 'failed')}**.",
                  f"- Promotion readiness: **{_status(report['promotion_ready'], 'yes', 'no')}**."])
    (output_dir / "recheck_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _status(value, true_text="passed", false_text="failed"):
    """Return a Polish status label for a boolean report field."""
    return true_text if value else false_text


def format_summary_row(row):
    """Format one external metric row as Markdown."""
    return (f"| {row['model']} | {row['mode']} | {row.get('auc_roc', 0):.4f} | "
            f"{row['recall']:.2%} | {row['false_positive_rate']:.2%} | "
            f"{row['specificity']:.2%} | {row['f1']:.4f} |")


def main() -> None:
    """Coordinate feature loading, evaluation and report writing."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    caches = _cache_directories(args)
    results = [evaluate_mode(args, mode, *caches) for mode in args.modes]
    metrics = [row for rows, _ in results for row in rows]
    real_rows = [row for _, rows in results for row in rows]
    write_metric_files(args.output_dir, metrics, real_rows)
    report = build_report(args, metrics, real_rows)
    report_path = args.output_dir / "recheck_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_summary(args.output_dir, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _cache_directories(args):
    """Resolve reusable external and report-local temporal feature caches."""
    external = args.external_cache_dir or args.output_dir / "external_features"
    return external, args.output_dir / "temporal_features"


if __name__ == "__main__":
    main()
