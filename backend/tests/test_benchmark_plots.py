from pathlib import Path

import pandas as pd

from utils.benchmark_plots import (
    _combined_realtime_size_matrix,
    _combined_runtime_size_matrix,
    _realtime_figure_size,
    _runtime_latency_figure,
    _size_category,
    plot_generalization_results,
    plot_heldout_results,
    plot_realtime_results,
    plot_runtime_results,
)


LABELS = {
    "logistic_regression": "Regresja Logistyczna",
    "random_forest": "Random Forest",
}


def test_realtime_plot_separates_classifier_and_api_scales(tmp_path: Path):
    frame = pd.DataFrame([
        _assessment("logistic_regression", 0.2, 1200.0, True),
        _assessment("random_forest", 20.0, 2400.0, False),
    ])
    plot_realtime_results(frame, tmp_path, LABELS, 100.0, 2000.0)
    assert (tmp_path / "pl" / "realtime_p95.png").exists()
    assert (tmp_path / "eng" / "realtime_p95.png").exists()
    assert (tmp_path / "pl" / "realtime_quality.png").exists()
    assert not (tmp_path / "pl" / "realtime_p95_url_corpus.png").exists()
    assert not (tmp_path / "pl" / "realtime_p95_original.png").exists()


def test_realtime_plot_separates_artifact_families(tmp_path: Path):
    frame = pd.DataFrame([
        _assessment("logistic_regression", 0.2, 1200.0, True),
        _assessment("random_forest", 20.0, 1800.0, True),
        {**_assessment("logistic_regression", 0.3, 1300.0, True), "dataset": "url_corpus"},
        {**_assessment("random_forest", 22.0, 1900.0, True), "dataset": "url_corpus"},
    ])
    plot_realtime_results(frame, tmp_path, LABELS, 100.0, 2000.0)
    assert (tmp_path / "pl" / "realtime_p95.png").exists()
    assert (tmp_path / "eng" / "active_artifact_size.png").exists()
    assert (tmp_path / "pl" / "realtime_classifier_p95_url_corpus.png").exists()
    assert (tmp_path / "pl" / "realtime_api_p95_url_corpus.png").exists()
    assert (tmp_path / "eng" / "realtime_api_p95_url_corpus.png").exists()
    assert not (tmp_path / "pl" / "realtime_p95_url_corpus.png").exists()


def test_realtime_latency_uses_compact_page_size():
    assert _realtime_figure_size(1) == (11.5, 3.6)
    assert _realtime_figure_size(2) == (11.5, 7.2)


def test_runtime_plot_creates_bilingual_readable_panels(tmp_path: Path):
    runtime = pd.DataFrame([
        _runtime(algorithm, mode, scope)
        for mode in ("full", "optimized")
        for algorithm in LABELS
        for scope in ("classifier_only", "scaler_and_classifier")
    ])
    loads = pd.DataFrame([
        _load(algorithm, mode)
        for mode in ("full", "optimized")
        for algorithm in LABELS
    ])
    plot_runtime_results(runtime, loads, tmp_path, LABELS)
    assert (tmp_path / "pl" / "prediction_p95_batch_1.png").exists()
    assert (tmp_path / "eng" / "artifact_size.png").exists()
    assert not (tmp_path / "pl" / "prediction_p95_batch_1_full.png").exists()


def test_runtime_latency_uses_one_linear_axis():
    from matplotlib import pyplot as plt

    frame = pd.DataFrame([
        _runtime(algorithm, mode, scope)
        for mode in ("full", "optimized")
        for algorithm in LABELS
        for scope in ("classifier_only", "scaler_and_classifier")
    ])
    figure = _runtime_latency_figure(frame, LABELS)
    assert len(figure.axes) == 1
    assert all(axis.get_xscale() == "linear" for axis in figure.axes)
    assert len(figure.axes[0].patches) == 4
    assert all(axis.get_xlabel() == "P95 czasu klasyfikacji [ms]"
               for axis in figure.axes)
    plt.close(figure)


def test_quality_plots_create_heatmaps_and_transitions(tmp_path: Path):
    metrics = pd.DataFrame([
        _metric(algorithm, mode, dataset)
        for mode in ("full", "selected")
        for algorithm in LABELS
        for dataset in ("internal", "external")
    ])
    differences = pd.DataFrame([
        _generalization_difference(algorithm, mode, metric)
        for mode in ("full", "selected")
        for algorithm in LABELS
        for metric in _metric_names()
    ])
    plot_generalization_results(metrics, differences, tmp_path, LABELS)
    assert (tmp_path / "pl" / "external_metric_heatmap.png").exists()
    assert (tmp_path / "eng" / "specificity_full_internal_external.png").exists()
    assert (tmp_path / "pl" / "specificity_selected_internal_external.png").exists()
    assert not (tmp_path / "pl" / "specificity_internal_external.png").exists()
    assert not (tmp_path / "pl" / "external_metric_heatmap_full.png").exists()


def test_heldout_plot_creates_variant_comparison(tmp_path: Path):
    metrics = pd.DataFrame([
        _metric(algorithm, mode, "held_out")
        for mode in ("full", "optimized")
        for algorithm in LABELS
    ])
    differences = pd.DataFrame([
        _variant_difference(algorithm, metric)
        for algorithm in LABELS
        for metric in _metric_names()
    ])
    plot_heldout_results(metrics, differences, tmp_path, LABELS)
    assert (tmp_path / "pl" / "held_out_metric_heatmap.png").exists()
    assert (tmp_path / "eng" / "held_out_false_positive_rate.png").exists()
    assert (tmp_path / "eng" / "optimized_full_difference_heatmap.png").exists()
    assert not (tmp_path / "pl" / "held_out_metric_heatmap_full.png").exists()


def test_benchmark_plots_use_bars_and_linear_scales():
    source = (
        Path(__file__).resolve().parents[1] / "utils" / "benchmark_plots.py"
    ).read_text()
    assert ".scatter(" not in source
    assert ".hlines(" not in source
    assert 'set_xscale("log")' not in source
    assert "skala logarytmiczna" not in source
    assert "kolor = przedział" not in source
    assert "_add_context" not in source
    assert ".set_title(" not in source
    assert ".suptitle(" not in source


def test_artifact_size_categories_keep_all_scales_visible():
    assert _size_category(float("nan")) == -1
    assert _size_category(0.005) == 0
    assert _size_category(6.5) == 1
    assert _size_category(305.0) == 2


def test_runtime_artifact_matrix_combines_both_sources():
    original = _loads_for_modes("full", "optimized")
    url_corpus = _loads_for_modes("full", "selected")
    matrix = _combined_runtime_size_matrix(
        {"original": original, "url_corpus": url_corpus}, LABELS
    )
    assert matrix.shape == (2, 4)
    assert list(matrix.index) == list(LABELS.values())
    assert list(matrix.columns)[-1] == (
        "URL Corpus\nPo selekcji cech\n48 z 110 cech"
    )


def test_algorithm_names_match_existing_notebooks():
    source = (
        Path(__file__).resolve().parents[1] / "utils" / "benchmark_plots.py"
    ).read_text()
    assert "Las losowy" not in source
    assert "Maszyna wektorów nośnych" not in source
    assert "Perceptron wielowarstwowy" not in source


def test_realtime_artifact_matrix_combines_both_sources():
    original = pd.DataFrame([
        _assessment(algorithm, 1.0, 2.0, True) for algorithm in LABELS
    ])
    url_corpus = original.assign(dataset="url_corpus")
    matrix = _combined_realtime_size_matrix(
        {"original": original, "url_corpus": url_corpus}, LABELS
    )
    assert matrix.shape == (2, 2)
    assert not matrix.isna().any().any()


def _loads_for_modes(*modes):
    return pd.DataFrame([
        _load(algorithm, mode) for mode in modes for algorithm in LABELS
    ])


def _assessment(algorithm, classifier_p95, api_p95, eligible):
    return {
        "dataset": "current",
        "algorithm": algorithm,
        "classifier_p95_ms": classifier_p95,
        "api_p95_ms": api_p95,
        "eligible": eligible,
        "f1": 0.95,
        "false_positive_rate": 0.02,
        "artifact_bytes": 100_000,
    }


def _runtime(algorithm, mode, scope):
    return {
        "algorithm": algorithm,
        "mode": mode,
        "scope": scope,
        "batch_size": 1,
        "p95_ms": 0.5 if algorithm == "logistic_regression" else 15.0,
    }


def _load(algorithm, mode):
    return {
        "algorithm": algorithm,
        "mode": mode,
        "p95_ms": 5.0,
        "artifact_bytes": 100_000,
    }


def _metric(algorithm, mode, dataset):
    return {
        "algorithm": algorithm,
        "mode": mode,
        "dataset": dataset,
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.93,
        "f1": 0.935,
        "roc_auc": 0.97,
        "specificity": 0.96 if dataset == "internal" else 0.78,
        "false_positive_rate": 0.04,
    }


def _metric_names():
    return ("accuracy", "precision", "recall", "f1", "roc_auc", "specificity")


def _generalization_difference(algorithm, mode, metric):
    return {
        "algorithm": algorithm,
        "mode": mode,
        "metric": metric,
        "internal_minus_external": 0.05,
    }


def _variant_difference(algorithm, metric):
    return {
        "algorithm": algorithm,
        "metric": metric,
        "optimized_minus_full": 0.001,
    }
