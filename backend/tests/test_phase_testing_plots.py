"""Tests for the shared phase-testing visual contract."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from utils.phase_testing_plots import (
    plot_confusion_matrix,
    plot_count_ranking,
    plot_error_rate_by_test_size,
    plot_metric_heatmap,
    plot_outcome_shares,
    plot_probability_ranking,
    plot_quality_by_test_size,
    remove_legacy_phase_testing_figures,
)
from utils.figure_localization import translate_figure_text


@pytest.fixture(autouse=True)
def close_figures():
    """Close figures created by each visual contract test."""
    yield
    plt.close("all")


def test_quality_curve_uses_full_probability_scale() -> None:
    """Avoid exaggerating clustered scores with a truncated vertical axis."""
    series = _quality_series()
    figure = plot_quality_by_test_size(series, "Miara F1")
    axis = figure.axes[0]
    assert axis.get_ylim() == (0.0, 1.0)
    _assert_axis_contract(axis)


def test_error_curve_uses_zero_based_scale_with_reserve() -> None:
    """Keep the baseline visible without wasting the percentage domain."""
    figure = plot_error_rate_by_test_size(_error_series())
    axis = figure.axes[0]
    assert axis.get_ylim()[0] == 0.0
    assert 5.4 < axis.get_ylim()[1] < 100.0
    _assert_axis_contract(axis)


def test_confusion_matrix_has_labels_and_color_legend() -> None:
    """Describe both matrix axes and preserve its count color scale."""
    figure = plot_confusion_matrix(np.array([[91, 9], [7, 93]]), "full")
    _assert_axis_contract(figure.axes[0], legend_required=False)
    assert len(figure.axes) == 2
    assert figure.axes[1].get_ylabel() == "Liczba próbek"
    assert figure.axes[0].collections[0].cmap.name == "Blues"


def test_confusion_matrices_use_one_comparable_color_scale() -> None:
    """Keep matrix colors independent of the dataset variant."""
    matrix = np.array([[91, 9], [7, 93]])
    full = plot_confusion_matrix(matrix, "full")
    optimized = plot_confusion_matrix(matrix, "optimized")
    assert full.axes[0].collections[0].cmap.name == "Blues"
    assert optimized.axes[0].collections[0].cmap.name == "Blues"


def test_metric_heatmap_uses_full_probability_range() -> None:
    """Keep heatmap intensity comparable without magnifying differences."""
    metrics = pd.DataFrame(
        [[0.95, 0.96], [0.97, 0.98]],
        index=["SVM", "MLP"],
        columns=["Miara F1", "Dokładność klasyfikacji"],
    )
    figure = plot_metric_heatmap(metrics, "optimized")
    colorbar = figure.axes[0].collections[0].colorbar
    assert (colorbar.vmin, colorbar.vmax) == (0.0, 1.0)
    _assert_axis_contract(figure.axes[0], legend_required=False)


def test_probability_ranking_uses_zero_based_bars() -> None:
    """Use honest zero-based bars with exact bounded score labels."""
    values = np.array([0.9712, 0.9654, 0.9587])
    figure = plot_probability_ranking(["MLP", "SVM", "XGBoost"], values, "full")
    axis = figure.axes[0]
    assert axis.get_xlim() == (0.0, 1.0)
    assert len(axis.patches) == len(values)
    assert not axis.collections
    assert len(axis.texts) == len(values)
    _assert_axis_contract(axis)


def test_error_count_ranking_uses_integer_ticks() -> None:
    """Format sample counts without fractional tick labels."""
    figure = plot_count_ranking(["MLP", "SVM"], [3, 7])
    axis = figure.axes[0]
    figure.canvas.draw()
    labels = [label.get_text() for label in axis.get_xticklabels()]
    assert all("," not in label for label in labels)
    _assert_axis_contract(axis)


def test_outcome_chart_exposes_small_error_share() -> None:
    """Normalize outcomes so that a small error segment remains measurable."""
    figure = plot_outcome_shares(["MLP", "SVM"], [990, 970], [10, 30])
    axis = figure.axes[0]
    assert axis.get_xlim() == (0.0, 100.0)
    assert any("1,00%" in text.get_text() for text in axis.texts)
    _assert_axis_contract(axis)


def test_phase_plot_labels_translate_completely() -> None:
    """Translate the phase-specific scientific labels used in English plots."""
    assert translate_figure_text(
        "Algorytm klasyfikacji i proporcja klas"
    ) == "Classification algorithm and class proportion"
    assert translate_figure_text(
        "Udział wyników klasyfikacji [%]"
    ) == "Share of classification outcomes [%]"
    assert translate_figure_text(
        "Liczba próbek w zbiorze testowym"
    ) == "Number of samples in the test set"


def test_legacy_phase_figures_are_removed_without_touching_localized_files(
    tmp_path,
) -> None:
    """Remove only obsolete duplicates outside language directories."""
    legacy = tmp_path / "phase_5" / "full" / "phase_statistics.png"
    localized = tmp_path / "phase_5" / "full" / "pl" / "phase_statistics.png"
    redundant = tmp_path / "phase_5" / "full" / "pl" / "confusion_matrix_mlp.png"
    legacy.parent.mkdir(parents=True)
    localized.parent.mkdir(parents=True)
    legacy.touch()
    localized.touch()
    redundant.touch()

    remove_legacy_phase_testing_figures(tmp_path)

    assert not legacy.exists()
    assert localized.exists()
    assert not redundant.exists()


def _quality_series():
    """Return clustered quality observations for two models."""
    return {
        "MLP": ([1000, 5000, 10000], [0.951, 0.958, 0.962]),
        "SVM": ([1000, 5000, 10000], [0.944, 0.952, 0.956]),
    }


def _error_series():
    """Return representative error percentages for two models."""
    return {
        "MLP": ([1000, 5000, 10000], [4.9, 4.2, 3.8]),
        "SVM": ([1000, 5000, 10000], [5.4, 4.8, 4.3]),
    }


def _assert_axis_contract(axis, *, legend_required: bool = True) -> None:
    """Assert labels, absence of embedded title and optional legend."""
    assert axis.get_xlabel()
    assert axis.get_ylabel()
    assert not axis.get_title()
    if legend_required:
        assert axis.get_legend() is not None
