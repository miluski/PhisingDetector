"""Readable scientific plots for phased model testing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

from utils.notebook_plot_style import (
    CLASS_COLORS,
    HEATMAP_CMAPS,
    MODEL_COLORS,
    configure_axis_limit_ticks,
    configure_count_colorbar,
    configure_numeric_x_axis,
    dataset_display_name,
    polish_number,
)


def plot_quality_by_test_size(
    series: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    y_label: str,
):
    """Plot one bounded quality measure against test-set size."""
    figure, axis = _model_line_figure(series)
    configure_numeric_x_axis(axis, "Liczba próbek w zbiorze testowym")
    axis.set_ylabel(y_label)
    axis.set_ylim(0.0, 1.0)
    configure_axis_limit_ticks(axis, "y")
    _finish_line_axis(axis)
    return figure


def plot_error_rate_by_test_size(
    series: Mapping[str, tuple[Sequence[float], Sequence[float]]],
):
    """Plot model error percentages against test-set size."""
    figure, axis = _model_line_figure(series)
    configure_numeric_x_axis(axis, "Liczba próbek w zbiorze testowym")
    axis.set_ylabel("Odsetek błędnie sklasyfikowanych próbek [%]")
    axis.set_ylim(0.0, _percentage_axis_upper(_series_values(series)))
    configure_axis_limit_ticks(axis, "y")
    _finish_line_axis(axis)
    return figure


def _series_values(series) -> np.ndarray:
    """Return all vertical observations from a collection of model series."""
    values = [np.asarray(item[1], dtype=float) for item in series.values()]
    return np.concatenate(values)


def _percentage_axis_upper(values) -> float:
    """Return a rounded zero-based percentage ceiling with visible reserve."""
    maximum = max(float(np.nanmax(values)), 0.0)
    step = 5.0 if maximum < 40.0 else 10.0
    reserved = maximum * 1.15
    return min(100.0, max(step, float(np.ceil(reserved / step) * step)))


def _model_line_figure(series):
    """Create a consistently styled multi-model line figure."""
    figure, axis = plt.subplots(figsize=(10, 7))
    for model_name, (x_values, y_values) in series.items():
        axis.plot(
            x_values, y_values, marker="o", linewidth=2.3,
            label=model_name, color=MODEL_COLORS.get(model_name, "#555555"),
        )
    return figure, axis


def _finish_line_axis(axis) -> None:
    """Apply the shared line-chart legend and grid."""
    axis.legend(loc="best", title="Algorytm klasyfikacji")
    axis.grid(True, alpha=0.25)


def plot_confusion_matrix(matrix, dataset_type: str):
    """Plot one confusion matrix with semantic labels and count legend."""
    figure, axis = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap=HEATMAP_CMAPS[dataset_type],
        vmin=0.0, vmax=float(np.asarray(matrix).sum()),
        ax=axis, xticklabels=["Brak phishingu", "Phishing"],
        yticklabels=["Brak phishingu", "Phishing"],
        cbar_kws={"label": "Liczba próbek"},
    )
    configure_count_colorbar(axis)
    axis.set_xlabel("Klasa przewidziana")
    axis.set_ylabel("Klasa rzeczywista")
    return figure


def plot_metric_heatmap(metrics, dataset_type: str):
    """Plot comparable quality measures in one annotated matrix."""
    figure, axis = plt.subplots(figsize=(11, 7))
    sns.heatmap(
        metrics, annot=True, fmt=".4f", cmap=HEATMAP_CMAPS[dataset_type],
        vmin=0.0, vmax=1.0, ax=axis,
        cbar_kws={"label": "Wartość miary jakości klasyfikacji"},
    )
    axis.set_xlabel("Miara jakości klasyfikacji")
    axis.set_ylabel("Algorytm klasyfikacji")
    return figure


def plot_probability_ranking(labels, values, dataset_type: str):
    """Plot a readable ranking of probability-valued model results."""
    figure, axis = plt.subplots(figsize=(12, 8))
    positions = np.arange(len(labels))
    colors = [_ranking_color(label) for label in labels]
    bars = axis.barh(positions, values, color=colors)
    _annotate_probability_bars(axis, bars, values)
    axis.set_yticks(positions, labels)
    axis.set_xlabel("Pole pod krzywą ROC")
    axis.set_ylabel("Algorytm klasyfikacji i proporcja klas")
    axis.set_xlim(0.0, 1.0)
    configure_axis_limit_ticks(axis, "x")
    axis.invert_yaxis()
    axis.legend(
        handles=_ranking_legend(labels), loc="lower center",
        bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False,
    )
    axis.grid(True, axis="x", alpha=0.25)
    return figure


def _annotate_probability_bars(axis, bars, values) -> None:
    """Place exact ranking values inside bars without extending the scale."""
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            float(value) - 0.01,
            bar.get_y() + bar.get_height() / 2.0,
            polish_number(float(value), 4),
            ha="right",
            va="center",
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
        )


def _ranking_color(label: str) -> str:
    """Return the stable algorithm color encoded in a ranking label."""
    model_name = label.split(" | ", maxsplit=1)[0]
    return MODEL_COLORS.get(model_name, "#555555")


def _ranking_legend(labels):
    """Return one semantic legend item for each represented algorithm."""
    model_names = list(dict.fromkeys(
        label.split(" | ", maxsplit=1)[0] for label in labels
    ))
    colors = [MODEL_COLORS.get(name, "#555555") for name in model_names]
    return _model_legend(model_names, colors)


def plot_count_ranking(labels, values, *, total: bool = False):
    """Plot integer error counts with model colors and exact labels."""
    figure, axis = plt.subplots(figsize=(11, 7))
    colors = [MODEL_COLORS.get(label, "#555555") for label in labels]
    bars = axis.barh(labels, values, color=colors)
    axis.bar_label(bars, labels=[polish_number(value, 0) for value in values], padding=4)
    axis.set_xlabel(_count_axis_label(total))
    axis.set_ylabel("Algorytm klasyfikacji")
    axis.set_xlim(0.0, _count_axis_upper(values))
    configure_axis_limit_ticks(axis, "x")
    axis.legend(handles=_model_legend(labels, colors), loc="best")
    axis.grid(True, axis="x", alpha=0.25)
    return figure


def _model_legend(labels, colors):
    """Return explicit model-color legend entries."""
    return [
        Patch(facecolor=color, label=label)
        for label, color in zip(labels, colors, strict=True)
    ]


def _count_axis_label(total: bool) -> str:
    """Return a precise error-count axis label."""
    if total:
        return "Łączna liczba błędnie sklasyfikowanych próbek"
    return "Liczba błędnie sklasyfikowanych próbek"


def _count_axis_upper(values) -> float:
    """Add enough right-side reserve for integer value labels."""
    maximum = max(float(np.max(values)), 1.0)
    return maximum * 1.16


def plot_outcome_shares(models, correct, errors):
    """Plot normalized correct and incorrect classification shares."""
    figure, axis = plt.subplots(figsize=(11, 7))
    correct_share, error_share = _outcome_percentages(correct, errors)
    axis.barh(
        models, correct_share, color=CLASS_COLORS["safe"],
        label="Poprawnie sklasyfikowane",
    )
    axis.barh(
        models, error_share, left=correct_share,
        color=CLASS_COLORS["phishing"], label="Błędnie sklasyfikowane",
    )
    _annotate_outcome_shares(axis, correct_share, error_share)
    axis.set_xlabel("Udział wyników klasyfikacji [%]")
    axis.set_ylabel("Algorytm klasyfikacji")
    axis.set_xlim(0.0, 100.0)
    axis.legend(loc="best")
    axis.grid(True, axis="x", alpha=0.25)
    return figure


def remove_obsolete_average_metric_figures(plot_root: str | Path) -> None:
    """Remove superseded one-metric bar charts from prior notebook runs."""
    metric_keys = ("accuracy", "precision", "recall", "f1", "roc_auc")
    for dataset_type in ("full", "optimized"):
        dataset_dir = Path(plot_root) / "summary" / dataset_type
        for metric_key in metric_keys:
            (dataset_dir / f"average_{metric_key}.png").unlink(missing_ok=True)


def remove_legacy_phase_testing_figures(plot_root: str | Path) -> None:
    """Remove figures superseded by the localized one-chart file layout."""
    root = Path(plot_root)
    for path in root.rglob("confusion_matrix_*.png"):
        path.unlink(missing_ok=True)
    patterns = (
        "confusion_matrices_*.png",
        "misclassification_distribution_*.png",
        "phase_statistics.png",
        "average_metrics.png",
        "size_impact.png",
    )
    for pattern in patterns:
        for path in root.rglob(pattern):
            if not {"pl", "eng"}.intersection(path.parts):
                path.unlink(missing_ok=True)


def _outcome_percentages(correct, errors):
    """Convert classification counts into row-wise percentages."""
    correct_values = np.asarray(correct, dtype=float)
    error_values = np.asarray(errors, dtype=float)
    totals = correct_values + error_values
    return correct_values / totals * 100.0, error_values / totals * 100.0


def _annotate_outcome_shares(axis, correct, errors) -> None:
    """Write percentages inside both normalized bar segments."""
    for position, (correct_value, error_value) in enumerate(
        zip(correct, errors, strict=True)
    ):
        if correct_value > 0.0:
            axis.text(
                correct_value / 2, position,
                f"{polish_number(correct_value, 2)}%", ha="center", va="center",
            )
        if error_value > 0.0:
            axis.text(
                correct_value + error_value / 2, position,
                f"{polish_number(error_value, 2)}%", ha="center", va="center",
            )


def plot_quality_metrics_by_test_size(
    all_series: Mapping[str, Mapping[str, tuple[Sequence[float], Sequence[float]]]],
    metric_specs: Sequence[tuple[str, str, str]],
    dataset_type: str,
):
    """Plot multiple quality metrics against test-set size in a single figure."""
    num_metrics = len(metric_specs)
    if num_metrics == 0:
        return plt.figure()

    rows, cols = (2, 2) if num_metrics > 1 else (1, 1)
    figure, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), sharex=True)
    if num_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (metric_key, metric_label, _) in enumerate(metric_specs):
        axis = axes[i]
        series = {
            model_name: data
            for model_name, data in all_series.get(metric_key, {}).items()
        }

        for model_name, (x_values, y_values) in series.items():
            axis.plot(
                x_values, y_values, marker="o", linewidth=2.0,
                label=model_name, color=MODEL_COLORS.get(model_name, "#555555"),
            )

        configure_numeric_x_axis(axis, "Liczba próbek w zbiorze testowym")
        axis.set_ylabel(metric_label)
        axis.set_ylim(0.0, 1.0)
        configure_axis_limit_ticks(axis, "y")
        _finish_line_axis(axis)

    # Hide unused subplots
    for i in range(num_metrics, len(axes)):
        axes[i].set_visible(False)

    figure.suptitle(
        f"Wpływ rozmiaru zbioru na jakość modeli ({dataset_display_name(dataset_type)})",
        fontsize=16,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    return figure
