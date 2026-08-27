from unittest.mock import Mock

import numpy as np
import pytest

from utils.research_benchmark import (
    algorithm_complexity,
    binary_metrics,
    measure_ns,
    quality_gate,
    recommend_realtime,
    stratified_bootstrap,
    timing_summary_ns,
    validate_active_payload,
    validate_safe_url,
    validate_staged_payload,
)


def test_measurement_excludes_warmup_and_uses_requested_repetitions():
    operation = Mock()
    clock = Mock(side_effect=[0, 10, 20, 50, 80, 130])
    assert measure_ns(operation, 2, 3, clock) == [10, 30, 50]
    assert operation.call_count == 5


def test_timing_summary_reports_linear_p95_and_throughput():
    summary = timing_summary_ns([1_000_000, 2_000_000, 3_000_000], batch_size=2)
    assert summary["mean_ms"] == pytest.approx(2.0)
    assert summary["median_ms"] == pytest.approx(2.0)
    assert summary["p95_ms"] == pytest.approx(2.9)
    assert summary["ms_per_url"] == pytest.approx(1.0)
    assert summary["urls_per_second"] == pytest.approx(1000.0)


def test_staged_payload_validation_catches_schema_and_scaler_contracts():
    payload = {
        "model": object(),
        "scaler": object(),
        "feature_names": ["a", "b"],
        "decision_threshold": 0.4,
        "mode": "full",
        "feature_policy": "v4",
    }
    assert validate_staged_payload(payload, ["a", "b"], "svm", "full", "v4")["features"] == 2
    with pytest.raises(ValueError, match="schema"):
        validate_staged_payload(payload, ["b", "a"], "svm", "full", "v4")
    with pytest.raises(ValueError, match="scaler"):
        validate_staged_payload({**payload, "scaler": None}, ["a", "b"], "svm", "full", "v4")


def test_selected_payload_accepts_audited_historical_mode_alias():
    payload = {
        "model": object(),
        "scaler": None,
        "feature_names": ["a"],
        "decision_threshold": 0.5,
        "mode": "optimized_v2",
        "feature_policy": "v4",
    }
    result = validate_staged_payload(payload, ["a"], "random_forest", "selected", "v4")
    assert result["payload_mode"] == "optimized_v2"


def test_active_payload_validation_uses_legacy_default_threshold():
    payload = {
        "model": object(),
        "scaler": object(),
        "feature_names": ["a", "b"],
    }
    result = validate_active_payload(payload, ["a", "b"], "svm")
    assert result == {"features": 2, "threshold": 0.5}
    with pytest.raises(ValueError, match="schema"):
        validate_active_payload(payload, ["b", "a"], "svm")


def test_random_forest_complexity_counts_nodes_leaves_and_depth():
    tree = Mock(node_count=5, children_left=np.array([1, -1, 3, -1, -1]), max_depth=2)
    model = Mock(estimators_=[Mock(tree_=tree), Mock(tree_=tree)])
    result = algorithm_complexity("random_forest", model)
    assert result == {
        "trees": 2,
        "nodes": 10,
        "leaves": 6,
        "mean_depth": 2.0,
        "max_depth": 2,
    }


def test_binary_metrics_and_quality_gate_cover_failure_boundary():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.9, 0.7])
    metrics = binary_metrics(labels, probabilities, 0.5)
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(1.0)
    assert quality_gate({"roc_auc": 0.8, "specificity": 0.8})["passed"]
    assert not quality_gate(metrics)["passed"]


def test_stratified_bootstrap_is_seeded_and_keeps_finite_intervals():
    labels = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.6, 0.4, 0.8, 0.9])
    first = stratified_bootstrap(labels, probabilities, 0.5, repetitions=30, seed=42)
    second = stratified_bootstrap(labels, probabilities, 0.5, repetitions=30, seed=42)
    assert first == second
    assert all(0.0 <= lower <= upper <= 1.0 for lower, upper in first.values())


def test_safe_url_contract_blocks_non_example_and_insecure_urls():
    validate_safe_url("https://example.com/account/verify?token=benchmark")
    with pytest.raises(ValueError):
        validate_safe_url("http://example.com")
    with pytest.raises(ValueError):
        validate_safe_url("https://example.org")


def test_realtime_recommendation_uses_declared_tie_break_order():
    base = {
        "eligible": True,
        "f1": 0.9,
        "false_positive_rate": 0.05,
        "api_p95_ms": 100.0,
        "artifact_bytes": 1000,
    }
    rows = [{**base, "algorithm": "slow"}, {**base, "algorithm": "fast", "api_p95_ms": 80.0}]
    assert recommend_realtime(rows)["algorithm"] == "fast"
