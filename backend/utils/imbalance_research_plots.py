"""Scientific visual contract for the class-imbalance experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

from utils.notebook_plot_style import (
    CLASS_COLORS,
    CLASS_LABELS,
    HEATMAP_CMAPS,
    MUTED_NEUTRAL_COLOR,
    configure_axis_limit_ticks,
    configure_count_colorbar,
    polish_number,
)


BALANCING_VARIANT_COLORS = {
    "bez balansowania": "#4477AA",
    "niedopróbkowanie": "#228833",
    "nadpróbkowanie": "#EE7733",
    "syntetyczne nadpróbkowanie": "#AA3377",
}

BALANCING_VARIANT_STYLES = {
    "bez balansowania": "-",
    "niedopróbkowanie": "--",
    "nadpróbkowanie": "-.",
    "syntetyczne nadpróbkowanie": ":",
}


def plot_class_distribution(labels, values, *, percentage: bool):
    """Plot class counts or percentages using semantic class colors."""
    figure, axis = plt.subplots(figsize=(8, 5))
    colors = [CLASS_COLORS["safe"], CLASS_COLORS["phishing"]]
    bars = axis.bar(labels, values, color=colors)
    axis.set_xlabel("Klasa")
    axis.set_ylabel(_distribution_axis_label(percentage))
    axis.set_ylim(0.0, _distribution_upper(values, percentage))
    configure_axis_limit_ticks(axis, "y")
    axis.bar_label(bars, labels=_distribution_labels(values, percentage), padding=4)
    axis.legend(handles=_class_legend(labels, colors), loc="best")
    axis.grid(True, axis="y", alpha=0.25)
    return figure


def _distribution_axis_label(percentage: bool) -> str:
    """Return the vertical label for a class-distribution chart."""
    return "Udział próbek [%]" if percentage else "Liczba próbek"


def _distribution_upper(values, percentage: bool) -> float:
    """Return a zero-based upper limit with annotation reserve."""
    if percentage:
        return 100.0
    return max(float(np.max(values)) * 1.15, 1.0)


def _distribution_labels(values, percentage: bool) -> list[str]:
    """Format exact values displayed above distribution bars."""
    decimals, suffix = (2, "%") if percentage else (0, "")
    return [f"{polish_number(float(value), decimals)}{suffix}" for value in values]


def _class_legend(labels, colors) -> list[Patch]:
    """Return explicit entries for the semantic class colors."""
    return [
        Patch(facecolor=color, label=label)
        for label, color in zip(labels, colors, strict=True)
    ]


def plot_precision_recall_curves(
    curves: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    average_precision: Mapping[str, float],
    baseline: float,
):
    """Plot balancing variants with redundant color and line encodings."""
    figure, axis = plt.subplots(figsize=(10, 7))
    for variant, (recall, precision) in curves.items():
        axis.plot(
            recall, precision, linewidth=2.3,
            color=BALANCING_VARIANT_COLORS[variant],
            linestyle=BALANCING_VARIANT_STYLES[variant],
            label=f"{variant}: {polish_number(average_precision[variant], 3)}",
        )
    _draw_precision_baseline(axis, baseline)
    _configure_precision_recall_axis(axis)
    return figure


def _draw_precision_baseline(axis, baseline: float) -> None:
    """Draw the prevalence baseline with a neutral reference style."""
    axis.axhline(
        baseline, color=MUTED_NEUTRAL_COLOR, linewidth=2.0,
        linestyle=(0, (7, 3)), label=f"Poziom bazowy: {polish_number(baseline, 3)}",
    )


def _configure_precision_recall_axis(axis) -> None:
    """Apply conventional probability limits and a compact legend."""
    axis.set_xlabel("Czułość")
    axis.set_ylabel("Precyzja")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    configure_axis_limit_ticks(axis, "x", floor=0.0, ceiling=1.0)
    configure_axis_limit_ticks(axis, "y", floor=0.0, ceiling=1.0)
    axis.legend(
        loc="lower left", title="Wariant balansowania (średnia precyzja)",
        framealpha=0.95,
    )
    axis.grid(True, alpha=0.25)


def shared_confusion_maximum(matrices: Mapping[str, np.ndarray]) -> float:
    """Return one common color maximum for comparable confusion matrices."""
    if not matrices:
        raise ValueError("At least one confusion matrix is required")
    return max(float(np.max(matrix)) for matrix in matrices.values())


def plot_imbalance_confusion_matrix(matrix, maximum: float):
    """Plot a confusion matrix on the experiment-wide count color scale."""
    figure, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap=HEATMAP_CMAPS["full"],
        vmin=0.0, vmax=maximum, ax=axis,
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        cbar_kws={"label": "Liczba próbek"},
    )
    configure_count_colorbar(axis)
    axis.set_xlabel("Klasa przewidziana")
    axis.set_ylabel("Klasa rzeczywista")
    return figure


def remove_legacy_imbalance_figures(plot_root: str | Path) -> None:
    """Remove obsolete nonlocalized figures from previous notebook runs."""
    dataset_root = Path(plot_root) / "full"
    for path in dataset_root.glob("*.png"):
        path.unlink()
