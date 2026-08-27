"""Tests for complete and unambiguous feature-name localization."""

import ast
import json
import re
from pathlib import Path

from config.feature_names_config import FeatureConfig
from utils.feature_names import (
    FEATURE_NAMES,
    FEATURE_NAMES_EN,
    get_readable_name,
)
from utils.figure_localization import (
    apply_academic_plotly_style,
    english_plotly_copy,
    translate_figure_text,
)
from utils.notebook_plot_style import (
    METRIC_LABELS,
    MODEL_COLORS,
    PARAMETER_LABELS,
    dataset_display_name,
)


POLISH_CHARACTERS = re.compile(r"[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]")
POLISH_PLOT_TERMS = re.compile(
    r"\b(odchylenie|standardowe|zbiór|uczacy|uczący|walidacyjny|testowy|"
    r"pełny|zoptymalizowany|liczba|wartość|cecha|czas|średnia|próbka|"
    r"próbek|próbki|błąd|dokładność|czułość|precyzja|klasa|rzeczywista|"
    r"przewidziana|prawdopodobieństwo|próg|uczenie|ważność|rozkład|"
    r"udział|wynik|fałsz|ścieżka|domena|plik|parametry|brak|faza|"
    r"głębokość|liście|warstwa|warstwy|wejściowa|waga|epok|ukrytych|"
    r"balansowania|bezpieczne|regresja|logistyczna|parametr|redukcja)\b",
    re.IGNORECASE,
)
PLOT_METHODS = {
    "set_xlabel", "set_ylabel", "set_title", "text", "annotate", "bar",
    "barh", "plot", "scatter", "axhline", "axvline", "add_bar",
    "add_annotation", "update_layout", "update_xaxes", "update_yaxes",
    "legend", "fill_between",
}
PLOT_TEXT_KEYWORDS = {
    "label", "name", "text", "hovertemplate", "title", "ticktext",
    "xaxis_title", "yaxis_title", "subplot_titles", "xticklabels",
    "yticklabels",
}


def test_feature_name_dictionaries_have_identical_keys():
    """Require an English label for every Polish feature label."""
    assert set(FEATURE_NAMES_EN) == set(FEATURE_NAMES)


def test_every_model_feature_has_both_localized_names():
    """Cover full, optimized and selected model schemas."""
    for mode in ("full", "optimized", "selected"):
        for feature in FeatureConfig.get_feature_names(mode):
            assert feature in FEATURE_NAMES
            assert feature in FEATURE_NAMES_EN


def test_english_feature_names_do_not_contain_polish_fragments():
    """Reject partially translated feature names."""
    forbidden = ("Liczba", "Domena", "Ścieżka", "Plik", "Parametry", "Brak")
    for label in FEATURE_NAMES_EN.values():
        assert not POLISH_CHARACTERS.search(label)
        assert not any(fragment in label for fragment in forbidden)


def test_readable_name_selects_requested_language():
    """Return stable labels for both supported language identifiers."""
    feature = "qty_slash_directory"

    assert get_readable_name(feature, "pl") == 'Ścieżka: Liczba "/"'
    assert get_readable_name(feature, "en") == 'Path: Number of "/" characters'
    assert get_readable_name(feature, "eng") == 'Path: Number of "/" characters'


def test_figure_translation_covers_every_feature_name():
    """Translate every Polish feature label using the canonical dictionary."""
    for feature, polish_label in FEATURE_NAMES.items():
        assert translate_figure_text(polish_label) == FEATURE_NAMES_EN[feature]


def test_plotly_copy_translates_nested_feature_labels():
    """Translate feature names in Plotly axes, traces and annotations."""
    import plotly.graph_objects as go

    polish = FEATURE_NAMES["qty_slash_directory"]
    figure = go.Figure(go.Bar(x=[1], y=[polish], name="Zbiór uczący"))
    figure.update_layout(xaxis_title=f'Wartość cechy: "{polish}"')

    translated = english_plotly_copy(figure)

    assert translated.data[0].y[0] == FEATURE_NAMES_EN["qty_slash_directory"]
    assert translated.data[0].name == "Training set"
    assert translated.layout.xaxis.title.text == (
        'Feature value: "Path: Number of "/" characters"'
    )


def test_academic_plotly_style_remains_valid_after_translation():
    """Preserve valid CSS colors in localized academic figures."""
    import plotly.graph_objects as go

    figure = go.Figure(go.Bar(x=[1], y=["Legalny URL"]))
    apply_academic_plotly_style(figure)
    translated = english_plotly_copy(figure)

    assert translated.layout.font.family == "DejaVu Serif"
    assert translated.layout.legend.bgcolor == "rgba(0, 0, 0, 0)"


def test_standard_deviation_legend_is_translated():
    """Translate uncertainty-band terminology used in learning curves."""
    assert translate_figure_text("± odchylenie standardowe") == (
        "± standard deviation"
    )


def test_english_numeric_labels_use_unambiguous_thousands():
    """Keep English plot ticks readable without locale grouping ambiguity."""
    assert translate_figure_text("1 000") == "1000"
    assert translate_figure_text("0,01") == "0.01"
    assert translate_figure_text("Średnia: 1 000 GB") == "Mean: 1000 GB"


def test_visible_plot_literals_do_not_remain_in_polish():
    """Audit axes, legends and annotations in shared and source notebooks."""
    for source in _plot_sources():
        for text in _plot_text_literals(source):
            translated = translate_figure_text(text)
            assert not POLISH_PLOT_TERMS.search(translated), text


def test_dynamic_metric_model_parameter_and_dataset_labels_are_translated():
    """Audit labels produced from dictionaries and runtime data values."""
    labels = [
        *METRIC_LABELS.values(),
        *PARAMETER_LABELS.values(),
        *MODEL_COLORS,
        dataset_display_name("full", 110),
        dataset_display_name("optimized", 48),
        "50% phishing, 50% bezpieczne",
        "bez balansowania",
    ]
    for label in labels:
        assert not POLISH_PLOT_TERMS.search(translate_figure_text(label)), label


def test_runtime_comparison_labels_have_expected_english_wording():
    """Keep frequently plotted dynamic labels precise and stable."""
    assert translate_figure_text(
        "Regresja Logistyczna | 50% phishing, 50% bezpieczne"
    ) == "Logistic Regression | 50% phishing, 50% non-phishing"
    assert translate_figure_text("Pełny zbiór (110 cech)") == (
        "Full dataset (110 features)"
    )


def test_benchmark_algorithm_labels_are_fully_localized():
    assert translate_figure_text("Regresja logistyczna") == "Logistic regression"
    assert translate_figure_text("Las losowy") == "Random forest"
    assert translate_figure_text("Maszyna wektorów nośnych (SVM)") == (
        "Support vector machine (SVM)"
    )
    assert translate_figure_text("Perceptron wielowarstwowy (MLP)") == (
        "Multilayer perceptron (MLP)"
    )
    assert translate_figure_text("Spełnia wszystkie kryteria") == "Meets all criteria"
    assert translate_figure_text("Próg opóźnienia") == "Latency threshold"
    assert translate_figure_text(
        "Wariant po selekcji cech (48 z 110)"
    ) == "Feature-selected variant (48 of 110)"


def _plot_sources():
    """Return shared plot source and code from canonical notebooks."""
    backend = Path(__file__).resolve().parents[1]
    yield (backend / "utils" / "notebook_plot_style.py").read_text()
    yield (backend / "utils" / "benchmark_plots.py").read_text()
    for path in sorted((backend / "notebooks" / "original").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                yield "".join(cell.get("source", []))


def _plot_text_literals(source):
    """Yield string literals passed to plot text APIs."""
    tree = ast.parse(source)
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        name = getattr(call.func, "attr", getattr(call.func, "id", ""))
        nodes = list(call.args) if name in PLOT_METHODS else []
        nodes.extend(
            keyword.value for keyword in call.keywords
            if keyword.arg in PLOT_TEXT_KEYWORDS
        )
        for node in nodes:
            yield from _string_literals(node)


def _string_literals(node):
    """Yield all strings nested in one expression."""
    for item in ast.walk(node):
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            yield item.value
