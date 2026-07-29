"""Tests for the class-imbalance visual contract."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex

from utils.figure_localization import translate_figure_text
from utils.imbalance_research_plots import (
    BALANCING_VARIANT_COLORS,
    plot_class_distribution,
    plot_imbalance_confusion_matrix,
    plot_precision_recall_curves,
    remove_legacy_imbalance_figures,
    shared_confusion_maximum,
)
from utils.notebook_plot_style import CLASS_COLORS


@pytest.fixture(autouse=True)
def close_figures():
    """Close figures created by each visual contract test."""
    yield
    plt.close("all")


def test_class_distribution_uses_semantic_colors_and_legend() -> None:
    """Keep non-phishing and phishing colors stable across notebooks."""
    figure = plot_class_distribution(
        ["Brak phishingu", "Phishing"], [46_400, 24_517], percentage=False,
    )
    axis = figure.axes[0]
    colors = [to_hex(patch.get_facecolor()) for patch in axis.patches]
    expected = [to_hex(CLASS_COLORS["safe"]), to_hex(CLASS_COLORS["phishing"])]
    assert colors == expected
    _assert_axis_contract(axis)


def test_precision_recall_curves_use_color_and_line_style() -> None:
    """Distinguish overlapping balancing variants by two encodings."""
    curves = _precision_recall_curves()
    scores = {variant: 0.96 for variant in curves}
    figure = plot_precision_recall_curves(curves, scores, 0.346)
    axis = figure.axes[0]
    variant_lines = axis.lines[:4]
    assert len({line.get_linestyle() for line in variant_lines}) == 4
    colors = [to_hex(line.get_color()) for line in variant_lines]
    expected = [to_hex(color) for color in BALANCING_VARIANT_COLORS.values()]
    assert colors == expected
    assert axis.get_xlim() == (0.0, 1.0)
    assert axis.get_ylim() == (0.0, 1.0)
    _assert_axis_contract(axis)


def test_confusion_matrices_share_count_color_scale() -> None:
    """Use one scale maximum so equal colors retain equal meanings."""
    matrices = {
        "bez balansowania": np.array([[100, 10], [5, 85]]),
        "nadpróbkowanie": np.array([[90, 20], [4, 86]]),
    }
    maximum = shared_confusion_maximum(matrices)
    figure = plot_imbalance_confusion_matrix(matrices["nadpróbkowanie"], maximum)
    axis = figure.axes[0]
    assert maximum == 100.0
    assert axis.collections[0].get_clim() == (0.0, 100.0)
    assert figure.axes[1].get_ylabel() == "Liczba próbek"
    _assert_axis_contract(axis, legend_required=False)


def test_legacy_imbalance_cleanup_preserves_localized_results(tmp_path) -> None:
    """Delete stale root images without deleting PL and ENG outputs."""
    full = tmp_path / "full"
    localized = full / "pl"
    localized.mkdir(parents=True)
    legacy, current = full / "old.png", localized / "current.png"
    legacy.touch()
    current.touch()
    remove_legacy_imbalance_figures(tmp_path)
    assert not legacy.exists()
    assert current.exists()


def test_embedded_polish_decimal_is_translated() -> None:
    """Convert decimal commas inside English legend text."""
    label = "Wariant balansowania (średnia precyzja): 0,961"
    assert translate_figure_text(label) == (
        "Balancing variant (average precision): 0.961"
    )


def _precision_recall_curves():
    """Return four representative overlapping precision-recall curves."""
    values = {
        "bez balansowania": 0.00,
        "niedopróbkowanie": 0.01,
        "nadpróbkowanie": 0.02,
        "syntetyczne nadpróbkowanie": 0.03,
    }
    recall = np.array([0.0, 0.5, 1.0])
    return {
        variant: (recall, np.array([1.0, 0.9 - offset, 0.35]))
        for variant, offset in values.items()
    }


def _assert_axis_contract(axis, *, legend_required: bool = True) -> None:
    """Assert complete labels, no embedded title and optional legend."""
    assert axis.get_xlabel()
    assert axis.get_ylabel()
    assert not axis.get_title()
    if legend_required:
        assert axis.get_legend() is not None
