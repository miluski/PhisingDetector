"""Readable bilingual figures for supplementary benchmark notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from utils.notebook_plot_style import (
    MAX_TWO_DECIMALS,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    TERTIARY_COLOR,
    metric_display_name,
    polish_number,
    save_comparison_figure,
)


MODE_LABELS = {
    "full": "Pełny wariant cech",
    "selected": "Wariant po selekcji cech (48 z 110)",
    "optimized": "Zoptymalizowany wariant cech",
}
MODE_COLORS = {
    "full": PRIMARY_COLOR,
    "selected": SECONDARY_COLOR,
    "optimized": SECONDARY_COLOR,
}
DATASET_LABELS = {
    "current": "Zbiór Original",
    "url_corpus": "Zbiór URL Corpus",
}
RUNTIME_ARTIFACT_LABELS = {
    ("original", "full"): "Original\nPełny zestaw cech\n111 cech",
    ("original", "optimized"): "Original\nPo optymalizacji\n74 z 111 cech",
    ("url_corpus", "full"): "URL Corpus\nPełny zestaw cech\n110 cech",
    ("url_corpus", "selected"): "URL Corpus\nPo selekcji cech\n48 z 110 cech",
}
REALTIME_ARTIFACT_LABELS = {
    "original": "Original\nAplikacja\n74 z 111 cech",
    "url_corpus": "URL Corpus\nAplikacja\n48 z 110 cech",
}
QUALITY_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc", "specificity"]
SIZE_COLORS = ListedColormap(["#eeeeee", "#b7e4c7", "#90caf9", "#ffcc80"])


def plot_runtime_results(
    runtime: pd.DataFrame,
    loads: pd.DataFrame,
    output_dir: Path,
    labels: dict[str, str],
    artifact_groups: dict[str, pd.DataFrame] | None = None,
) -> None:
    """Create one comparative figure for every runtime research question."""
    stems = ("prediction_p95_batch_1", "deserialization_p95", "artifact_size")
    _clear_plots(output_dir, stems)
    batch_one = runtime[runtime["batch_size"] == 1]
    _plot_runtime_latency(batch_one, output_dir, labels)
    _plot_runtime_artifacts(loads, artifact_groups, output_dir, labels)


def _plot_runtime_artifacts(loads, groups, output_dir, labels):
    _plot_grouped_mode_bar(
        loads, "p95_ms", "Czas deserializacji P95 [ms]",
        output_dir / "deserialization_p95.png", labels,
    )
    matrix = _runtime_size_matrix(loads, groups, labels)
    _plot_size_matrix(matrix, output_dir / "artifact_size.png")


def _plot_runtime_latency(frame, output_dir, labels):
    figure = _runtime_latency_figure(frame, labels)
    path = output_dir / "prediction_p95_batch_1.png"
    save_comparison_figure(figure, path)


def _runtime_latency_figure(frame, labels):
    """Return classifier-only latency on one shared linear scale."""
    selected = frame[frame["scope"] == "classifier_only"]
    modes = _modes(frame)
    figure, axis = plt.subplots(figsize=(11, 5.8))
    _finish_runtime_axis(axis, selected, labels, modes)
    figure.tight_layout(pad=1.2)
    return figure


def _finish_runtime_axis(axis, frame, labels, modes):
    """Draw classifier latency for every algorithm on one axis."""
    _draw_mode_bars(axis, frame, "p95_ms", labels, modes)
    maximum = frame["p95_ms"].max()
    axis.set_xlim(0.0, float(maximum) * 1.24)
    axis.set_xlabel("P95 czasu klasyfikacji [ms]")
    axis.set_ylabel("Algorytm klasyfikacji")
    axis.legend(frameon=False)
    axis.grid(axis="x", alpha=0.3)


def _plot_grouped_mode_bar(frame, column, x_label, path, labels):
    modes = _modes(frame)
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    _draw_mode_bars(axis, frame, column, labels, modes)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Algorytm klasyfikacji")
    axis.legend(frameon=False)
    save_comparison_figure(figure, path)


def _draw_mode_bars(axis, frame, column, labels, modes):
    positions = np.arange(len(labels))
    width = 0.72 / len(modes)
    for index, mode in enumerate(modes):
        values = _mode_values(frame, mode, column, labels)
        offset = (index - (len(modes) - 1) / 2) * width
        bars = axis.barh(
            positions + offset, values, width,
            color=MODE_COLORS[mode], label=MODE_LABELS[mode],
        )
        _label_bars(axis, bars, values, 3)
    axis.set_yticks(positions, list(labels.values()))
    axis.grid(axis="x", alpha=0.25)
    axis.margins(x=0.16)


def _mode_values(frame, mode, column, labels):
    selected = frame[frame["mode"] == mode].set_index("algorithm")
    return selected.loc[list(labels), column].tolist()


def _mode_size_matrix(frame, column, labels):
    modes = _modes(frame)
    matrix = frame.pivot(index="algorithm", columns="mode", values=column)
    matrix = matrix.loc[list(labels), modes]
    matrix.index = list(labels.values())
    matrix.columns = [MODE_LABELS[mode] for mode in modes]
    return matrix


def _runtime_size_matrix(loads, groups, labels):
    if groups:
        return _combined_runtime_size_matrix(groups, labels)
    sized = loads.assign(artifact_mb=loads["artifact_bytes"] / 1_000_000)
    return _mode_size_matrix(sized, "artifact_mb", labels)


def _combined_runtime_size_matrix(groups, labels):
    columns = {}
    for family, frame in groups.items():
        for mode in _modes(frame):
            label = RUNTIME_ARTIFACT_LABELS[(family, mode)]
            values = frame[frame["mode"] == mode].set_index("algorithm")
            columns[label] = values["artifact_bytes"] / 1_000_000
    matrix = pd.DataFrame(columns).reindex(list(labels))
    matrix.index = list(labels.values())
    return matrix


def _plot_size_matrix(matrix, path):
    categories = matrix.map(_size_category)
    annotations = matrix.map(_size_annotation)
    height = max(4.5, 1.2 + 0.75 * len(matrix))
    width = max(9.5, 3.5 + 2.4 * len(matrix.columns))
    figure, axis = plt.subplots(figsize=(width, height))
    sns.heatmap(
        categories, annot=annotations, fmt="", cmap=SIZE_COLORS,
        vmin=-1.5, vmax=2.5, linewidths=1.5, linecolor="white",
        cbar_kws={"ticks": [-1, 0, 1, 2]}, ax=axis,
    )
    plt.setp(axis.get_xticklabels(), rotation=0, ha="center")
    _finish_size_matrix(axis)
    save_comparison_figure(figure, path)


def _finish_size_matrix(axis):
    colorbar = axis.collections[0].colorbar
    colorbar.set_ticklabels([
        "brak artefaktu", "poniżej 1 MB",
        "od 1 do 10 MB", "powyżej 10 MB",
    ])
    axis.set_xlabel("Wariant lub rodzina artefaktu")
    axis.set_ylabel("Algorytm klasyfikacji")


def _size_category(value):
    if pd.isna(value):
        return -1
    if value < 1:
        return 0
    return 1 if value <= 10 else 2


def _size_annotation(value):
    if pd.isna(value):
        return "brak artefaktu"
    return f"{_compact_number(value, 3)} MB"


def plot_generalization_results(
    metrics: pd.DataFrame,
    differences: pd.DataFrame,
    output_dir: Path,
    labels: dict[str, str],
) -> None:
    """Create joint full-versus-selected generalization figures."""
    stems = (
        "external_metric_heatmap", "specificity_internal_external",
        "specificity_full_internal_external",
        "specificity_selected_internal_external",
        "generalization_difference_heatmap",
    )
    _clear_plots(output_dir, stems)
    external = metrics[metrics["dataset"] == "external"]
    _plot_external_metrics(external, output_dir, labels)
    _plot_specificity(metrics, output_dir, labels)
    _plot_generalization_differences(differences, output_dir, labels)


def _plot_external_metrics(frame, output_dir, labels):
    columns = ["roc_auc", "specificity", "f1"]
    matrix = _comparison_matrix(frame, columns, labels)
    figure, axis = plt.subplots(figsize=(9.5, 7.2))
    _draw_heatmap(axis, matrix, 0.65, 1.0, "Wartość metryki", 3)
    axis.set_xlabel("Metryka na zbiorze zewnętrznym")
    axis.set_ylabel("Algorytm i wariant cech")
    path = output_dir / "external_metric_heatmap.png"
    save_comparison_figure(figure, path)


def _plot_specificity(frame, output_dir, labels):
    for mode in _modes(frame):
        figure, axis = plt.subplots(figsize=(10.5, 4.8))
        _draw_specificity_mode(axis, frame[frame["mode"] == mode], labels, mode)
        _specificity_legend(axis)
        path = output_dir / f"specificity_{mode}_internal_external.png"
        save_comparison_figure(figure, path)


def _draw_specificity_mode(axis, frame, labels, mode):
    data = frame.set_index(["algorithm", "dataset"])
    internal = [data.loc[(name, "internal"), "specificity"] for name in labels]
    external = [data.loc[(name, "external"), "specificity"] for name in labels]
    positions = np.arange(len(labels))
    _draw_dataset_bars(axis, positions, internal, external)
    axis.axvline(0.8, color="#222222", linestyle="--", linewidth=1.2)
    axis.set_yticks(positions, list(labels.values()))
    axis.set_xlim(0, 1.02)
    axis.xaxis.set_major_formatter(MAX_TWO_DECIMALS)
    axis.set_xlabel("Specyficzność")
    axis.set_ylabel(f"Algorytm — {MODE_LABELS[mode]}")
    axis.grid(axis="x", alpha=0.25)


def _draw_dataset_bars(axis, positions, internal, external):
    width = 0.34
    first = axis.barh(
        positions - width / 2, internal, width,
        color=PRIMARY_COLOR, label="Zbiór wewnętrzny",
    )
    second = axis.barh(
        positions + width / 2, external, width,
        color=SECONDARY_COLOR, label="Zbiór zewnętrzny",
    )
    _label_bars(axis, first, internal, 3)
    _label_bars(axis, second, external, 3)


def _specificity_legend(axis):
    handles, names = axis.get_legend_handles_labels()
    handles.append(Line2D([], [], color="#222222", linestyle="--"))
    names.append("Próg 0,80")
    axis.legend(
        handles, names, loc="center left",
        bbox_to_anchor=(1.01, 0.5), frameon=False,
    )


def _plot_generalization_differences(frame, output_dir, labels):
    matrix = _difference_matrix(frame, "internal_minus_external", labels)
    limit = max(0.01, float(matrix.abs().to_numpy().max()))
    figure, axis = plt.subplots(figsize=(12.0, 7.2))
    _draw_heatmap(axis, matrix, -limit, limit, "Wewnętrzny − zewnętrzny", 3, 0)
    axis.set_xlabel("Różnica: zbiór wewnętrzny − zbiór zewnętrzny")
    axis.set_ylabel("Algorytm i wariant cech")
    path = output_dir / "generalization_difference_heatmap.png"
    save_comparison_figure(figure, path)


def plot_heldout_results(
    metrics: pd.DataFrame,
    differences: pd.DataFrame,
    output_dir: Path,
    labels: dict[str, str],
) -> None:
    """Create joint full-versus-optimized held-out figures."""
    stems = ("held_out_metric_heatmap", "held_out_false_positive_rate",
             "optimized_full_difference_heatmap")
    _clear_plots(output_dir, stems)
    _plot_heldout_metrics(metrics, output_dir, labels)
    _plot_grouped_mode_bar(
        metrics, "false_positive_rate",
        "Odsetek fałszywych alarmów (FPR) — mniej oznacza lepiej",
        output_dir / "held_out_false_positive_rate.png", labels,
    )
    _plot_variant_differences(differences, output_dir, labels)


def _plot_heldout_metrics(frame, output_dir, labels):
    matrix = _comparison_matrix(frame, QUALITY_COLUMNS, labels)
    figure, axis = plt.subplots(figsize=(12.0, 7.2))
    _draw_heatmap(axis, matrix, 0.9, 1.0, "Wartość metryki", 3)
    axis.set_xlabel("Metryka na wydzielonym zbiorze testowym")
    axis.set_ylabel("Algorytm i wariant cech")
    path = output_dir / "held_out_metric_heatmap.png"
    save_comparison_figure(figure, path)


def _plot_variant_differences(frame, output_dir, labels):
    matrix = frame[frame["metric"].isin(QUALITY_COLUMNS)].pivot(
        index="algorithm", columns="metric", values="optimized_minus_full"
    ).loc[list(labels), QUALITY_COLUMNS]
    matrix.index = list(labels.values())
    matrix.columns = [metric_display_name(name) for name in matrix.columns]
    limit = max(0.001, float(matrix.abs().to_numpy().max()))
    figure, axis = plt.subplots(figsize=(10.5, 5.2))
    _draw_heatmap(axis, matrix, -limit, limit, "Zoptymalizowany − pełny", 4, 0)
    axis.set_xlabel("Metryka jakości")
    axis.set_ylabel("Algorytm klasyfikacji")
    save_comparison_figure(figure, output_dir / "optimized_full_difference_heatmap.png")


def _comparison_matrix(frame, columns, labels):
    modes = _modes(frame)
    order = [(algorithm, mode) for algorithm in labels for mode in modes]
    matrix = frame.set_index(["algorithm", "mode"]).loc[order, columns]
    matrix.index = _comparison_labels(order, labels)
    matrix.columns = [metric_display_name(name) for name in matrix.columns]
    return matrix


def _difference_matrix(frame, value_column, labels):
    modes = _modes(frame)
    order = [(algorithm, mode) for algorithm in labels for mode in modes]
    matrix = frame[frame["metric"].isin(QUALITY_COLUMNS)].pivot(
        index=["algorithm", "mode"], columns="metric", values=value_column,
    ).loc[order, QUALITY_COLUMNS]
    matrix.index = _comparison_labels(order, labels)
    matrix.columns = [metric_display_name(name) for name in matrix.columns]
    return matrix


def _comparison_labels(order, labels):
    return [f"{labels[algorithm]} — {MODE_LABELS[mode]}" for algorithm, mode in order]


def _draw_heatmap(axis, matrix, minimum, maximum, colorbar_label, decimals, center=None):
    annotations = matrix.map(lambda value: polish_number(float(value), decimals))
    sns.heatmap(
        matrix, annot=annotations, fmt="", center=center,
        vmin=minimum, vmax=maximum, cmap="vlag" if center == 0 else "YlGnBu",
        cbar_kws={"label": colorbar_label}, ax=axis,
    )
    plt.setp(axis.get_xticklabels(), rotation=25, ha="right")
    colorbar = axis.collections[0].colorbar
    colorbar.ax.yaxis.set_major_formatter(MAX_TWO_DECIMALS)
    colorbar.update_ticks()


def plot_realtime_results(
    assessment: pd.DataFrame,
    output_dir: Path,
    labels: dict[str, str],
    classifier_limit: float,
    api_limit: float,
    artifact_groups: dict[str, pd.DataFrame] | None = None,
) -> None:
    """Create one application-level figure for every measured dimension."""
    stems = (
        "realtime_p95", "realtime_p95_url_corpus",
        "realtime_classifier_p95_url_corpus", "realtime_api_p95_url_corpus",
        "realtime_quality", "active_artifact_size",
    )
    _clear_plots(output_dir, stems)
    _plot_realtime_latency(
        assessment, labels, output_dir / "realtime_p95.png",
        classifier_limit, api_limit,
    )
    _plot_url_corpus_latency(
        assessment, labels, output_dir, classifier_limit, api_limit,
    )
    _plot_realtime_quality(assessment, labels, output_dir / "realtime_quality.png")
    _plot_realtime_artifacts(assessment, artifact_groups, output_dir, labels)


def _plot_url_corpus_latency(frame, labels, output_dir, classifier_limit, api_limit):
    selected = frame[frame["dataset"] == "url_corpus"]
    if selected.empty:
        return
    _plot_classifier_p95(
        selected, labels, output_dir / "realtime_classifier_p95_url_corpus.png",
    )
    _plot_api_p95(
        selected, labels, output_dir / "realtime_api_p95_url_corpus.png", api_limit,
    )


def _plot_classifier_p95(frame, labels, path):
    """Plot classifier latency without compressing values against threshold."""
    figure, axis = plt.subplots(figsize=(9.5, 5.2))
    names = [labels[name] for name in frame["algorithm"]]
    values = frame["classifier_p95_ms"]
    bars = axis.barh(names, values, color=TERTIARY_COLOR)
    _label_bars(axis, bars, values, 3)
    axis.set_xlim(0.0, float(max(values)) * 1.22)
    axis.set_xlabel("P95 czasu klasyfikacji [ms]")
    axis.set_ylabel("Algorytm — Zbiór URL Corpus")
    axis.grid(axis="x", alpha=0.25)
    save_comparison_figure(figure, path)


def _plot_api_p95(frame, labels, path, limit):
    """Plot end-to-end API latency with its acceptance threshold."""
    figure, axis = plt.subplots(figsize=(9.5, 5.2))
    names = [labels[name] for name in frame["algorithm"]]
    values = frame["api_p95_ms"]
    colors = [TERTIARY_COLOR if value <= limit else SECONDARY_COLOR
              for value in values]
    _focused_api_bar(axis, names, values, limit, _axis_limit(values, limit), colors)
    axis.set_xlabel("P95 czasu pełnego żądania API [ms]")
    axis.set_ylabel("Algorytm — Zbiór URL Corpus")
    _add_realtime_legend(axis, loc="lower right", anchor=(0.98, 0.03))
    save_comparison_figure(figure, path)


def _focused_api_bar(axis, names, values, limit, maximum, colors):
    bars = axis.barh(names, values, color=colors)
    axis.axvline(limit, color="#222222", linestyle="--", linewidth=1.2)
    axis.grid(axis="x", alpha=0.25)
    axis.set_xlim(0, maximum)
    _label_threshold_bars(axis, bars, values, limit, maximum)


def _label_threshold_bars(axis, bars, values, limit, maximum):
    labels = [_compact_number(value, 1) for value in values]
    offset = maximum * 0.015
    for bar, value, label in zip(bars, values, labels):
        inside = float(value) < limit
        position = float(value) - offset if inside else float(value) + offset
        axis.text(
            position, bar.get_y() + bar.get_height() / 2, label,
            ha="right" if inside else "left", va="center",
            color="white" if inside else "#25344F",
        )


def _plot_realtime_artifacts(assessment, groups, output_dir, labels):
    matrix = _realtime_artifact_matrix(assessment, groups, labels)
    _plot_size_matrix(matrix, output_dir / "active_artifact_size.png")


def _plot_realtime_latency(frame, labels, path, classifier_limit, api_limit):
    datasets = frame["dataset"].drop_duplicates().tolist()
    figure, axes = plt.subplots(len(datasets), 2, squeeze=False,
                               figsize=_realtime_figure_size(len(datasets)))
    classifier_max = _axis_limit(frame["classifier_p95_ms"], classifier_limit)
    api_max = _axis_limit(frame["api_p95_ms"], api_limit)
    for row, dataset in enumerate(datasets):
        selected = frame[frame["dataset"] == dataset]
        names = [labels[name] for name in selected["algorithm"]]
        colors = _eligibility_colors(selected)
        _latency_bar(axes[row, 0], names, selected["classifier_p95_ms"],
                     classifier_limit, classifier_max, colors)
        _latency_bar(axes[row, 1], names, selected["api_p95_ms"],
                     api_limit, api_max, colors)
        _finish_latency_row(axes[row], dataset)
    _add_realtime_legend(axes[0, 0])
    save_comparison_figure(figure, path)


def _realtime_figure_size(dataset_count):
    return 11.5, 3.6 * dataset_count


def _latency_bar(axis, names, values, limit, maximum, colors):
    bars = axis.barh(names, values, color=colors)
    axis.axvline(limit, color="#222222", linestyle="--", linewidth=1.2)
    _label_bars(axis, bars, values, 1)
    axis.grid(axis="x", alpha=0.25)
    axis.set_xlim(0, maximum)


def _finish_latency_row(axes, dataset):
    axes[0].set_xlabel("Opóźnienie P95 klasyfikatora [ms]")
    axes[1].set_xlabel("Opóźnienie P95 pełnego przepływu API [ms]")
    axes[0].set_ylabel(f"Algorytm — {DATASET_LABELS[dataset]}")
    axes[1].tick_params(axis="y", labelleft=False)


def _axis_limit(values, threshold):
    return max(float(max(values)) * 1.18, threshold * 1.12)


def _plot_realtime_quality(frame, labels, path):
    datasets = frame["dataset"].drop_duplicates().tolist()
    figure, axes = plt.subplots(len(datasets), 2, squeeze=False,
                               figsize=(16.0, 4.8 * len(datasets)))
    fpr_limit = float(frame["false_positive_rate"].max()) * 1.15
    for row, dataset in enumerate(datasets):
        selected = frame[frame["dataset"] == dataset]
        names = [labels[name] for name in selected["algorithm"]]
        _quality_row(axes[row], selected, names, fpr_limit)
        axes[row, 0].set_ylabel(f"Algorytm — {DATASET_LABELS[dataset]}")
    save_comparison_figure(figure, path)


def _quality_row(axes, frame, names, fpr_limit):
    f1_bars = axes[0].barh(names, frame["f1"], color=PRIMARY_COLOR)
    fpr_bars = axes[1].barh(names, frame["false_positive_rate"], color=SECONDARY_COLOR)
    _label_bars(axes[0], f1_bars, frame["f1"], 3)
    _label_bars(axes[1], fpr_bars, frame["false_positive_rate"], 3)
    axes[0].set_xlabel("Miara F1 — więcej oznacza lepiej")
    axes[1].set_xlabel("FPR — mniej oznacza lepiej")
    axes[0].set_xlim(0, 1.06)
    axes[1].set_xlim(0, fpr_limit)
    axes[1].tick_params(axis="y", labelleft=False)
    for axis in axes:
        axis.grid(axis="x", alpha=0.25)


def _realtime_size_matrix(frame, labels):
    sized = frame.assign(artifact_mb=frame["artifact_bytes"] / 1_000_000)
    datasets = frame["dataset"].drop_duplicates().tolist()
    algorithms = [name for name in labels if name in set(frame["algorithm"])]
    matrix = sized.pivot(
        index="algorithm", columns="dataset", values="artifact_mb",
    ).reindex(index=algorithms, columns=datasets)
    matrix.index = [labels[name] for name in algorithms]
    matrix.columns = [DATASET_LABELS[name] for name in datasets]
    return matrix


def _realtime_artifact_matrix(assessment, groups, labels):
    if groups:
        return _combined_realtime_size_matrix(groups, labels)
    return _realtime_size_matrix(assessment, labels)


def _combined_realtime_size_matrix(groups, labels):
    columns = {}
    for family, frame in groups.items():
        dataset = "current" if family == "original" else "url_corpus"
        selected = frame[frame["dataset"] == dataset].set_index("algorithm")
        columns[REALTIME_ARTIFACT_LABELS[family]] = (
            selected["artifact_bytes"] / 1_000_000
        )
    matrix = pd.DataFrame(columns).reindex(list(labels))
    matrix.index = list(labels.values())
    return matrix


def _eligibility_colors(frame):
    return [TERTIARY_COLOR if value else SECONDARY_COLOR for value in frame["eligible"]]


def _add_realtime_legend(axis, loc="upper center", anchor=(0.67, 0.98)):
    handles = [
        Patch(color=TERTIARY_COLOR, label="Spełnia kryteria"),
        Patch(color=SECONDARY_COLOR, label="Nie spełnia kryteriów"),
        Line2D([], [], color="#222222", linestyle="--", label="Próg opóźnienia"),
    ]
    axis.legend(
        handles=handles, loc=loc, bbox_to_anchor=anchor, frameon=True,
        framealpha=0.9, edgecolor="none", fontsize=8,
    )


def _modes(frame):
    return frame["mode"].drop_duplicates().tolist()


def _clear_plots(output_dir, stems):
    for language in ("pl", "eng", "en"):
        directory = Path(output_dir) / language
        for stem in stems:
            for path in directory.glob(f"{stem}*.png"):
                path.unlink()


def _label_bars(axis, bars, values, decimals):
    labels = [_compact_number(value, decimals) for value in values]
    axis.bar_label(bars, labels=labels, padding=4)


def _compact_number(value, decimals):
    return polish_number(float(value), decimals).rstrip("0").rstrip(",")
