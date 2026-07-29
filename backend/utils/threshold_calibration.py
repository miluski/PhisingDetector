"""Threshold and model-selection helpers for phishing experiments.

The hyperparameter search ranks estimators on validation folds, while the
decision threshold is calibrated separately on the held-out validation split.
The final test split is never used to choose a threshold.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import f1_score, recall_score, roc_auc_score, roc_curve


DEFAULT_TARGET_FALSE_POSITIVE_RATE = 0.02
DEFAULT_MAX_FALSE_POSITIVE_RATE = 0.10
DEFAULT_MIN_RECALL = 0.80


def false_positive_rate(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    """Return FP/(TN+FP), i.e. the false-positive rate for class 0."""
    labels = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    benign = labels == 0
    return float(predictions[benign].mean()) if benign.any() else 0.0


def calibrate_threshold(
    y_validation: Iterable[int],
    probabilities: Iterable[float],
    *,
    target_false_positive_rate: float = DEFAULT_TARGET_FALSE_POSITIVE_RATE,
    max_false_positive_rate: float = DEFAULT_MAX_FALSE_POSITIVE_RATE,
    min_recall: float = DEFAULT_MIN_RECALL,
) -> dict[str, float | bool]:
    """Select a validation threshold under target and hard FPR constraints.

    The lower target FPR is a safety margin for distribution shift. If it
    cannot retain the required recall, calibration may use the hard FPR
    limit. Invalid operating points raise ``ValueError``.
    """
    y_true, scores = _calibration_arrays(y_validation, probabilities)
    _validate_inputs(y_true, scores, target_false_positive_rate, max_false_positive_rate, min_recall)
    return _select_threshold(
        y_true, scores, target_false_positive_rate,
        max_false_positive_rate, min_recall,
    )


def _calibration_arrays(y_validation, probabilities):
    """Convert public calibration inputs into aligned numeric arrays."""
    y_true = np.asarray(y_validation).astype(int)
    scores = np.asarray(probabilities, dtype=float).reshape(-1)
    return y_true, scores


def _select_threshold(y_true, scores, target_fpr, max_fpr, min_recall):
    """Select the best threshold from the preferred or acceptable tier."""
    preferred, acceptable = _rank_thresholds(
        y_true, scores, target_fpr, max_fpr, min_recall,
    )
    if not acceptable:
        raise ValueError("No validation threshold satisfies FPR and recall constraints")
    selected = max(preferred) if preferred else _most_conservative(acceptable)
    return _calibration_result(
        y_true, scores, selected, bool(preferred), target_fpr, max_fpr, min_recall,
    )


def _most_conservative(candidates):
    """Minimise FPR when the safety target cannot retain required recall."""
    return max(candidates, key=lambda item: (item[2], item[0], item[1], item[3]))


def _validate_inputs(y_true, scores, target_fpr, max_fpr, min_recall):
    """Reject malformed calibration inputs and contradictory constraints."""
    if len(y_true) != len(scores) or not len(y_true):
        raise ValueError("Validation labels and probabilities must have equal non-zero length")
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Validation data must contain both binary classes")
    if not np.isfinite(scores).all():
        raise ValueError("Validation probabilities must be finite")
    if not 0 <= target_fpr <= max_fpr <= 1 or not 0 <= min_recall <= 1:
        raise ValueError("Threshold calibration constraints must be within [0, 1]")


def _rank_thresholds(y_true, scores, target_fpr, max_fpr, min_recall):
    """Rank thresholds satisfying the safety target and hard constraints."""
    preferred, acceptable = [], []
    for item, fpr, recall in _threshold_statistics(y_true, scores):
        if fpr <= max_fpr and recall >= min_recall:
            acceptable.append(item)
            if fpr <= target_fpr:
                preferred.append(item)
    return preferred, acceptable


def _threshold_statistics(y_true, scores):
    """Return F1-aware ranking data for every distinct operating point."""
    false_rates, recalls, thresholds = roc_curve(
        y_true, scores, drop_intermediate=False,
    )
    f1_values = _curve_f1(y_true, false_rates, recalls)
    values = zip(f1_values, recalls, false_rates, thresholds, strict=True)
    return [
        ((float(f1), float(recall), -float(fpr), -float(threshold)), fpr, recall)
        for f1, recall, fpr, threshold in values
    ]


def _curve_f1(y_true, false_rates, recalls):
    """Calculate F1 values from ROC-curve rates and class counts."""
    positives = float((y_true == 1).sum())
    true_positives = recalls * positives
    false_positives = false_rates * float((y_true == 0).sum())
    denominators = true_positives + false_positives
    precisions = np.divide(
        true_positives, denominators, out=np.zeros_like(recalls),
        where=denominators != 0,
    )
    f1_values = np.divide(
        2 * precisions * recalls, precisions + recalls,
        out=np.zeros_like(recalls), where=(precisions + recalls) != 0,
    )
    return f1_values


def _calibration_result(
    y_true, scores, selected, margin_satisfied, target_fpr, max_fpr, min_recall,
):
    """Build the public calibration report for the selected threshold."""
    threshold = -selected[3]
    predictions = (scores >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "validation_fpr": false_positive_rate(y_true, predictions),
        "validation_recall": float(recall_score(y_true, predictions, zero_division=0)),
        "validation_f1": float(f1_score(y_true, predictions, zero_division=0)),
        "constraint_satisfied": True,
        "safety_margin_satisfied": margin_satisfied,
        "target_fpr": float(target_fpr),
        "max_fpr": float(max_fpr),
        "min_recall": float(min_recall),
    }


def model_selection_score(estimator, X, y) -> float:
    """FPR-aware scorer for GridSearchCV using the estimator's 0.5 decision.

    AUC remains the primary ranking signal; the penalty discourages a model
    that labels most benign URLs as phishing. Threshold calibration is still
    performed afterwards on the held-out validation split.
    """
    probabilities = predict_probabilities(estimator, X)
    predictions = (probabilities >= 0.5).astype(int)
    fpr = false_positive_rate(y, predictions)
    recall = float(recall_score(y, predictions, zero_division=0))
    auc = float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else 0.0
    return auc + 0.25 * recall - 0.50 * fpr


def predict_probabilities(estimator, X) -> np.ndarray:
    """Return a comparable [0, 1] score for sklearn/Keras-like estimators."""
    if hasattr(estimator, "predict_proba"):
        values = np.asarray(estimator.predict_proba(X)[:, 1]).reshape(-1)
    elif hasattr(estimator, "decision_function"):
        values = np.asarray(estimator.decision_function(X)).reshape(-1)
        values = 1.0 / (1.0 + np.exp(-values))
    else:
        values = np.asarray(estimator.predict(X)).reshape(-1)
    return values.astype(float)
