"""Pure measurement and validation helpers for research notebooks."""

from __future__ import annotations

import hashlib
import json
from time import perf_counter_ns
from typing import Any, Callable
from urllib.parse import urlparse

import numpy as np
from sklearn.metrics import roc_auc_score


SCALER_ALGORITHMS = {"logistic_regression", "svm", "mlp"}


def measure_ns(
    operation: Callable[[], Any],
    warmups: int,
    repetitions: int,
    clock: Callable[[], int] = perf_counter_ns,
) -> list[int]:
    """Measure completed calls after excluded warm-up calls."""
    for _ in range(warmups):
        operation()
    samples = []
    for _ in range(repetitions):
        started = clock()
        operation()
        samples.append(clock() - started)
    return samples


def timing_summary_ns(samples: list[int], batch_size: int = 1) -> dict[str, float]:
    """Summarize nanosecond samples as latency and throughput."""
    values = np.asarray(samples, dtype=float) / 1_000_000
    mean_ms = float(values.mean())
    return {
        "mean_ms": mean_ms,
        "median_ms": float(np.median(values)),
        "std_ms": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "p95_ms": float(np.percentile(values, 95)),
        "ms_per_url": mean_ms / batch_size,
        "urls_per_second": 1000.0 * batch_size / mean_ms,
    }


def validate_staged_payload(
    payload: dict[str, Any],
    expected_columns: list[str],
    algorithm: str,
    mode: str,
    feature_policy: str,
) -> dict[str, Any]:
    """Validate scientific artifact schema, preprocessing and threshold."""
    _require_payload_keys(payload)
    if list(payload["feature_names"]) != expected_columns:
        raise ValueError(f"Feature schema mismatch for {algorithm}/{mode}")
    _validate_payload_metadata(payload, mode, feature_policy)
    _validate_scaler(payload, algorithm)
    threshold = float(payload["decision_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Decision threshold outside [0, 1]: {threshold}")
    return {"features": len(expected_columns), "threshold": threshold,
            "payload_mode": payload["mode"]}


def validate_active_payload(
    payload: dict[str, Any],
    expected_columns: list[str],
    algorithm: str,
) -> dict[str, Any]:
    """Validate active application payload and preprocessing schema."""
    _require_active_payload_keys(payload)
    if list(payload["feature_names"]) != expected_columns:
        raise ValueError(f"Active feature schema mismatch for {algorithm}")
    _validate_scaler(payload, algorithm)
    threshold = float(payload.get("decision_threshold", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Decision threshold outside [0, 1]: {threshold}")
    return {"features": len(expected_columns), "threshold": threshold}


def _require_active_payload_keys(payload: dict[str, Any]) -> None:
    required = {"model", "scaler", "feature_names"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Active payload lacks keys: {sorted(missing)}")


def _require_payload_keys(payload: dict[str, Any]) -> None:
    required = {"model", "scaler", "feature_names", "decision_threshold"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Artifact payload lacks keys: {sorted(missing)}")


def _validate_payload_metadata(
    payload: dict[str, Any], mode: str, feature_policy: str,
) -> None:
    accepted_modes = {mode, "optimized_v2"} if mode == "selected" else {mode}
    if payload.get("mode") not in accepted_modes:
        raise ValueError(f"Artifact mode mismatch: {payload.get('mode')!r}")
    if payload.get("feature_policy") != feature_policy:
        raise ValueError(f"Artifact feature policy mismatch: {payload.get('feature_policy')!r}")


def _validate_scaler(payload: dict[str, Any], algorithm: str) -> None:
    scaler_present = payload.get("scaler") is not None
    if scaler_present != (algorithm in SCALER_ALGORITHMS):
        raise ValueError(f"Unexpected scaler policy for {algorithm}")


def classifier_probabilities(model: Any, values: Any) -> np.ndarray:
    """Return phishing probabilities without preprocessing."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(values)[:, 1]).reshape(-1)
    if callable(model):
        output = model(values, training=False)
        output = output.numpy() if hasattr(output, "numpy") else output
        return np.asarray(output).reshape(-1)
    return np.asarray(model.predict(values, verbose=0)).reshape(-1)


def algorithm_complexity(algorithm: str, model: Any) -> dict[str, Any]:
    """Extract algorithm-specific structural complexity."""
    extractors = {
        "logistic_regression": _linear_complexity,
        "random_forest": _forest_complexity,
        "xgboost": _xgboost_complexity,
        "svm": _svm_complexity,
        "mlp": _mlp_complexity,
    }
    return extractors[algorithm](model)


def _linear_complexity(model: Any) -> dict[str, Any]:
    coefficients = int(np.asarray(model.coef_).size)
    intercepts = int(np.asarray(model.intercept_).size)
    return {
        "coefficients": coefficients,
        "intercepts": intercepts,
        "trainable_parameters": coefficients + intercepts,
    }


def _tree_counts(tree: Any) -> tuple[int, int, int]:
    nodes = int(tree.node_count)
    leaves = int(np.count_nonzero(tree.children_left == -1))
    return nodes, leaves, int(tree.max_depth)


def _forest_complexity(model: Any) -> dict[str, Any]:
    counts = [_tree_counts(estimator.tree_) for estimator in model.estimators_]
    values = np.asarray(counts)
    return {
        "trees": len(counts),
        "nodes": int(values[:, 0].sum()),
        "leaves": int(values[:, 1].sum()),
        "mean_depth": float(values[:, 2].mean()),
        "max_depth": int(values[:, 2].max()),
    }


def _xgboost_tree_counts(node: dict[str, Any], depth: int = 0) -> tuple[int, int, int]:
    children = node.get("children", [])
    if not children:
        return 1, 1, depth
    results = [_xgboost_tree_counts(child, depth + 1) for child in children]
    return 1 + sum(item[0] for item in results), sum(item[1] for item in results), max(item[2] for item in results)


def _xgboost_complexity(model: Any) -> dict[str, Any]:
    trees = [json.loads(item) for item in model.get_booster().get_dump(dump_format="json")]
    counts = [_xgboost_tree_counts(tree) for tree in trees]
    return {
        "trees": len(trees),
        "nodes": sum(item[0] for item in counts),
        "leaves": sum(item[1] for item in counts),
        "max_depth": max(item[2] for item in counts),
    }


def _svm_complexity(model: Any) -> dict[str, Any]:
    return {
        "kernel": str(model.kernel),
        "support_vectors": int(model.support_.size),
        "support_vectors_by_class": json.dumps(model.n_support_.astype(int).tolist()),
    }


def _mlp_complexity(model: Any) -> dict[str, Any]:
    neurons = [int(layer.units) for layer in model.layers if hasattr(layer, "units")]
    parameters = sum(int(np.prod(weight.shape)) for weight in model.trainable_weights)
    return {
        "layers": len(model.layers),
        "neurons": int(sum(neurons)),
        "neurons_by_layer": json.dumps(neurons),
        "trainable_parameters": parameters,
    }


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    """Calculate binary classification measures from fixed predictions."""
    labels = np.asarray(labels, dtype=int)
    predicted = np.asarray(probabilities) >= threshold
    tn, fp, fn, tp = _confusion_counts(labels, predicted)
    return {
        "accuracy": _ratio(tp + tn, len(labels)),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "false_positive_rate": _ratio(fp, fp + tn),
        "specificity": _ratio(tn, tn + fp),
    }


def _confusion_counts(labels: np.ndarray, predicted: np.ndarray) -> tuple[int, int, int, int]:
    positive = labels == 1
    negative = ~positive
    return (
        int(np.sum(negative & ~predicted)),
        int(np.sum(negative & predicted)),
        int(np.sum(positive & ~predicted)),
        int(np.sum(positive & predicted)),
    )


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def stratified_bootstrap(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    repetitions: int = 1000,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Return percentile intervals from class-stratified row resampling."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    rng = np.random.default_rng(seed)
    samples = [_bootstrap_once(labels, probabilities, threshold, groups, rng) for _ in range(repetitions)]
    return _bootstrap_intervals(samples)


def _bootstrap_once(labels, probabilities, threshold, groups, rng):
    indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
    return binary_metrics(labels[indices], probabilities[indices], threshold)


def _bootstrap_intervals(samples: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    return {
        metric: tuple(np.percentile([row[metric] for row in samples], [2.5, 97.5]).astype(float))
        for metric in samples[0]
    }


def quality_gate(
    metrics: dict[str, float],
    auc_minimum: float = 0.80,
    specificity_minimum: float = 0.80,
) -> dict[str, Any]:
    """Evaluate existing external generalization thresholds."""
    auc_passed = metrics["roc_auc"] >= auc_minimum
    specificity_passed = metrics["specificity"] >= specificity_minimum
    return {
        "roc_auc_minimum": auc_minimum,
        "specificity_minimum": specificity_minimum,
        "roc_auc_passed": auc_passed,
        "specificity_passed": specificity_passed,
        "passed": auc_passed and specificity_passed,
    }


def prediction_checksum(labels: np.ndarray, probabilities: np.ndarray) -> str:
    """Hash labels and deterministic probabilities for reproducibility."""
    digest = hashlib.sha256()
    digest.update(np.asarray(labels, dtype=np.int8).tobytes())
    digest.update(np.asarray(probabilities, dtype=np.float64).tobytes())
    return digest.hexdigest()


def validate_safe_url(url: str) -> None:
    """Reject every benchmark URL except HTTPS example.com."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "example.com":
        raise ValueError("Benchmark URL must use https://example.com")


def recommend_realtime(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select best eligible active algorithm using declared tie-break order."""
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        return None
    key = lambda row: (-row["f1"], row["false_positive_rate"], row["api_p95_ms"], row["artifact_bytes"])
    return min(eligible, key=key)
