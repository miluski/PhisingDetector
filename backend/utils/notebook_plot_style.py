"""Shared visual contract for model-training notebooks.

Dataset identity belongs in output paths, not plot titles.  Colors encode the
same concept in every algorithm notebook.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MaxNLocator
from sklearn.metrics import precision_recall_curve, roc_curve

from utils.figure_localization import (
    english_matplotlib_copy,
    language_output_path,
    path_language,
    remove_legacy_language_files,
)


DATASET_SUBDIRS = {
    "Pełny zbiór": "full",
    "Zoptymalizowany zbiór": "optimized",
    "Pelny (111)": "full",
    "Zoptymalizowany (73)": "optimized",
    "full": "full",
    "optimized": "optimized",
}

PRIMARY_COLOR = "#4477AA"
SECONDARY_COLOR = "#EE7733"
TERTIARY_COLOR = "#228833"
# Blue variant keeps algorithm plots consistent with the model notebooks.
QUATERNARY_COLOR = "#3366CC"
FIFTH_COLOR = "#66CCEE"
NEUTRAL_COLOR = "#555555"
MUTED_NEUTRAL_COLOR = "#777777"

DATASET_COLORS = {"full": PRIMARY_COLOR, "optimized": SECONDARY_COLOR}
TRAIN_COLOR = PRIMARY_COLOR
VALIDATION_COLOR = SECONDARY_COLOR

TRAIN_DATA_LABEL = "Zbiór uczący"
VALIDATION_DATA_LABEL = "Zbiór walidacyjny"
TEST_DATA_LABEL = "Zbiór testowy"
SAMPLE_COUNT_AXIS_LABEL = "Liczba próbek"
PARAMETER_C_AXIS_LABEL = (
    "Wartość parametru C (C > 0; odwrotność siły regularyzacji)"
)
CPU_AXIS_LABEL = "Zmiana wykorzystania procesora względem poziomu bazowego [%]"
RAM_AXIS_LABEL = "Zmiana wykorzystania pamięci RAM względem poziomu bazowego [GB]"
RESOURCE_BASELINE_COLOR = MUTED_NEUTRAL_COLOR
RESOURCE_MEAN_COLOR = TERTIARY_COLOR
TRAINING_START_COLOR = "#222222"
TRAINING_END_COLOR = QUATERNARY_COLOR
TRAINING_START_STYLE = ":"
TRAINING_END_STYLE = "--"
FINALIZATION_LABEL = "Zapis modelu i zakończenie procesu"
TRAINING_END_LABEL = "Koniec uczenia / początek zapisu modelu"
FINALIZATION_ALPHA = 0.12


def polish_number(value: float, decimals: int = 2) -> str:
    """Format a number with Polish decimal and thousands separators."""
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "\0").replace(".", ",").replace("\0", " ")


def _compact_polish_number(value: float, _position=None) -> str:
    """Format an axis value with at most two decimal places."""
    return polish_number(value, 2).rstrip("0").rstrip(",")


MAX_TWO_DECIMALS = FuncFormatter(_compact_polish_number)
INTEGER_COUNT_FORMATTER = FuncFormatter(
    lambda value, _position: polish_number(value, 0)
)

CLASS_COLORS = {
    "safe": PRIMARY_COLOR,
    "phishing": SECONDARY_COLOR,
}

MODEL_COLORS = {
    "Regresja Logistyczna": PRIMARY_COLOR,
    "Random Forest": TERTIARY_COLOR,
    "XGBoost": SECONDARY_COLOR,
    "SVM": QUATERNARY_COLOR,
    "MLP": FIFTH_COLOR,
}

HEATMAP_CMAPS = {
    "full": "Blues",
    "optimized": "Blues",
}

METRIC_LABELS = {
    "accuracy": "Dokładność klasyfikacji",
    "precision": "Precyzja",
    "recall": "Czułość",
    "f1": "Miara F1",
    "roc_auc": "Pole pod krzywą ROC",
    "auc_roc": "Pole pod krzywą ROC",
    "false_positive_rate": "Odsetek fałszywych alarmów (FPR)",
    "specificity": "Specyficzność",
    "bce": "Binarna entropia krzyżowa",
    "error_rate": "Współczynnik błędu klasyfikacji",
}

TRAIN_TEST_METRIC_KEYS = {
    "accuracy": ("train_accuracy", "accuracy"),
    "precision": ("train_precision", "precision"),
    "recall": ("train_recall", "recall"),
    "false_positive_rate": (
        "train_false_positive_rate", "false_positive_rate",
    ),
    "error_rate": ("train_error_rate", "test_error_rate"),
    "f1": ("train_f1", "f1"),
    "roc_auc": ("train_auc_roc", "auc_roc"),
}

CLASSIFICATION_METRIC_GROUPS = {
    "jakosc": ("accuracy", "precision", "recall", "f1", "roc_auc"),
    "bledy": ("false_positive_rate", "error_rate"),
}

PARAMETER_LABELS = {
    "C": PARAMETER_C_AXIS_LABEL,
    "alpha": "Współczynnik regularyzacji alpha",
    "n_estimators": "Liczba drzew decyzyjnych",
    "max_depth": "Maksymalna głębokość drzewa",
    "learning_rate": "Współczynnik uczenia",
    "gamma": "Parametr gamma",
    "kernel": "Jądro funkcji decyzyjnej",
    "hidden_layer_sizes": "Rozmiar warstw ukrytych",
    "epochs": "Liczba epok uczenia",
}


def metric_display_name(metric_key: str) -> str:
    """Return a full Polish label for a metric key."""
    return METRIC_LABELS.get(metric_key, metric_key.replace("_", " ").capitalize())


def classification_metric_sets(result: dict) -> tuple[dict, dict]:
    """Return coherent train and test metrics from one evaluation result."""
    train_metrics = {
        metric: _validated_metric(result[train_key], train_key)
        for metric, (train_key, _) in TRAIN_TEST_METRIC_KEYS.items()
    }
    test_metrics = {
        metric: _validated_metric(result[test_key], test_key)
        for metric, (_, test_key) in TRAIN_TEST_METRIC_KEYS.items()
    }
    return train_metrics, test_metrics


def _validated_metric(value, name):
    """Return a finite probability metric or reject inconsistent results."""
    number = float(value)
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"Metric {name} must be within [0, 1]")
    return number


def parameter_display_name(parameter_name: str) -> str:
    """Return a descriptive Polish label for a model parameter."""
    return PARAMETER_LABELS.get(
        parameter_name, parameter_name.replace("_", " ").capitalize()
    )


def dataset_key(dataset_name: str) -> str:
    """Return canonical dataset key used by directories and color maps."""
    try:
        return DATASET_SUBDIRS[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset name: {dataset_name!r}") from exc


def dataset_display_name(
    dataset_name: str, feature_count: int | None = None
) -> str:
    """Return a Polish display label without encoding stale feature counts."""
    key = dataset_key(dataset_name)
    label = "Pełny zbiór" if key == "full" else "Zoptymalizowany zbiór"
    return f"{label} ({feature_count} cech)" if feature_count is not None else label


def dataset_plot_dir(base_dir: str | Path, dataset_name: str) -> Path:
    """Create and return dataset-specific plot directory."""
    path = Path(base_dir) / dataset_key(dataset_name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataset_figure(
    figure,
    base_dir: str | Path,
    dataset_name: str,
    filename: str,
    *,
    dpi: int = 300,
    display_in_notebook: bool = True,
) -> Path:
    """Save localized figure variants inside the dataset directory."""
    output_path = dataset_plot_dir(base_dir, dataset_name) / filename
    return save_comparison_figure(
        figure,
        output_path,
        dpi=dpi,
        display_in_notebook=display_in_notebook,
    )


def save_comparison_figure(
    figure,
    output_path: str | Path,
    *,
    dpi: int = 300,
    display_in_notebook: bool = True,
) -> Path:
    """Save Polish and English variants of a comparison figure."""
    output_path = Path(output_path)
    clear_figure_titles(figure)
    if path_language(output_path):
        return _save_matplotlib_figure(figure, output_path, dpi)
    return _save_bilingual_figure(
        figure,
        output_path,
        dpi,
        display_in_notebook,
    )


def _save_bilingual_figure(
    figure,
    output_path: Path,
    dpi: int,
    display_in_notebook: bool,
) -> Path:
    """Save and optionally display both localized figure variants."""
    polish_path = language_output_path(output_path, "pl")
    english_path = language_output_path(output_path, "eng")
    _save_matplotlib_figure(figure, polish_path, dpi)
    english_figure = english_matplotlib_copy(figure)
    _save_matplotlib_figure(english_figure, english_path, dpi)
    _display_bilingual_figures(
        figure,
        english_figure,
        display_in_notebook,
    )
    remove_legacy_language_files(output_path)
    return polish_path


def _save_matplotlib_figure(figure, output_path: Path, dpi: int) -> Path:
    """Save one Matplotlib figure at an explicit localized path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.2
    )
    return output_path


def _display_bilingual_figures(
    polish_figure,
    english_figure,
    display_in_notebook: bool,
) -> None:
    """Display both variants during execution in a Jupyter kernel."""
    if display_in_notebook and _running_in_notebook():
        from IPython.display import display

        display(polish_figure)
        display(english_figure)
    plt.close(polish_figure)
    plt.close(english_figure)


def _running_in_notebook() -> bool:
    """Return whether code is executing in an IPython notebook kernel."""
    try:
        shell = get_ipython()  # type: ignore[name-defined]
    except NameError:
        return False
    return shell.__class__.__name__ == "ZMQInteractiveShell"


def remove_dataset_figure(
    base_dir: str | Path, dataset_name: str, filename: str
) -> None:
    """Remove one obsolete dataset figure without touching other results."""
    path = Path(base_dir) / dataset_key(dataset_name) / filename
    path.unlink(missing_ok=True)


def clear_figure_titles(figure) -> None:
    """Remove embedded titles while preserving axes, legends and annotations."""
    for axis in figure.axes:
        axis.set_title("")
    suptitle = getattr(figure, "_suptitle", None)
    if suptitle is not None:
        suptitle.set_text("")


def shared_limits(
    series: Iterable[Iterable[float]],
    *,
    padding_ratio: float = 0.05,
    floor: float | None = None,
    ceiling: float | None = None,
) -> tuple[float, float]:
    """Return finite limits shared by separate figures.

    Padding uses full observed span. Constant series receive small stable span.
    """
    values = _finite_series(series)
    low, high = float(values.min()), float(values.max())
    span = max(high - low, max(abs(high), 1.0) * 0.1)
    padding = span * padding_ratio
    lower = low - padding if floor is None else max(floor, low - padding)
    upper = high + padding if ceiling is None else min(ceiling, high + padding)
    return _valid_limits(lower, upper, floor, ceiling)


def _valid_limits(lower, upper, floor, ceiling):
    """Return an ordered interval when clipping collapses a constant range."""
    if upper > lower:
        return lower, upper
    span = max(abs(lower), 1.0) * 0.05
    lower = lower - span if floor is None else max(floor, lower - span)
    upper = upper + span if ceiling is None else min(ceiling, upper + span)
    if upper <= lower:
        raise ValueError("Axis domain does not contain the supplied values")
    return lower, upper


def configure_dynamic_axis(
    axis,
    axis_name: str,
    series: Iterable[Iterable[float]],
    *,
    floor: float | None = None,
    ceiling: float | None = None,
    padding_ratio: float = 0.08,
    max_decimals: int | None = None,
) -> None:
    """Fit one numeric axis to observed data with domain-aware reserve."""
    limits = shared_limits(
        series, padding_ratio=padding_ratio, floor=floor, ceiling=ceiling
    )
    _set_compact_axis_ticks(
        axis, axis_name, limits, max_decimals=max_decimals
    )


def scientific_scale(
    values, *, include_zero: bool = False, symmetric: bool = False,
    reserve_ratio: float = 0.08, target_intervals: int = 6,
    rounded_bounds: bool = True, tick_step: float | None = None,
    tick_anchor: float = 0.0,
) -> tuple[float, float, np.ndarray]:
    """Return rounded limits and equally spaced ticks for Plotly figures."""
    finite = _finite_series((values,))
    lower, upper = _scientific_data_range(
        finite, include_zero, symmetric, reserve_ratio
    )
    if tick_step is not None:
        return _predefined_axis_ticks(
            lower, upper, tick_step, tick_anchor, rounded_bounds
        )
    if not rounded_bounds:
        return _interior_axis_ticks(lower, upper, target_intervals)
    return _equal_axis_ticks(
        lower, upper, target_intervals=target_intervals
    )


def _interior_axis_ticks(lower, upper, target_intervals):
    """Keep exact reserved bounds and place equal rounded ticks inside them."""
    step = _nice_axis_step(upper - lower, target_intervals)
    first = np.ceil(lower / step) * step
    last = np.floor(upper / step) * step
    ticks = np.arange(first, last + step * 0.5, step)
    if ticks.size < 2:
        ticks = np.linspace(lower, upper, 3)
    return float(lower), float(upper), _clean_ticks(ticks)


def _predefined_axis_ticks(lower, upper, step, anchor, rounded_bounds):
    """Build ticks from an explicit positive interval and optional anchor."""
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("Tick step must be a finite positive number")
    first = np.ceil((lower - anchor) / step) * step + anchor
    last = np.floor((upper - anchor) / step) * step + anchor
    if rounded_bounds:
        lower = np.floor((lower - anchor) / step) * step + anchor
        upper = np.ceil((upper - anchor) / step) * step + anchor
        first, last = lower, upper
    ticks = np.arange(first, last + step * 0.5, step)
    return float(lower), float(upper), _clean_ticks(ticks)


def _scientific_data_range(values, include_zero, symmetric, reserve_ratio):
    """Add proportional reserve before rounding scientific scale bounds."""
    lower, upper = float(values.min()), float(values.max())
    nonnegative, nonpositive = lower >= 0.0, upper <= 0.0
    if include_zero:
        lower, upper = min(lower, 0.0), max(upper, 0.0)
    if symmetric:
        bound = max(abs(lower), abs(upper), 1e-12)
        return -bound * (1.0 + reserve_ratio), bound * (1.0 + reserve_ratio)
    span = max(upper - lower, max(abs(lower), abs(upper)) * 0.1, 1e-12)
    lower, upper = lower - span * reserve_ratio, upper + span * reserve_ratio
    if nonnegative:
        lower = max(0.0, lower)
    if nonpositive:
        upper = min(0.0, upper)
    return lower, upper


def probability_limits(
    series: Iterable[Iterable[float]],
    *,
    padding_ratio: float = 0.08,
    minimum_span: float = 0.02,
) -> tuple[float, float]:
    """Return readable, bounded limits for a probability-valued line chart."""
    values = _finite_series(series)
    low = float(np.clip(values.min(), 0.0, 1.0))
    high = float(np.clip(values.max(), 0.0, 1.0))
    span = max((high - low) * (1.0 + 2.0 * padding_ratio), minimum_span)
    lower, upper = _bounded_span((low + high) / 2.0, span)
    return lower, upper


def _bounded_span(center: float, span: float) -> tuple[float, float]:
    """Fit a requested span inside the closed probability interval."""
    lower, upper = max(0.0, center - span / 2.0), min(1.0, center + span / 2.0)
    requested = min(span, 1.0)
    if upper - lower >= requested or np.isclose(upper - lower, requested):
        return lower, upper
    if lower == 0.0:
        return 0.0, min(1.0, span)
    return max(0.0, 1.0 - span), 1.0


def _finite_series(series):
    """Flatten finite numeric values from all input series."""
    arrays = [np.asarray(values, dtype=float).reshape(-1) for values in series]
    finite = [values[np.isfinite(values)] for values in arrays if values.size]
    if not finite:
        raise ValueError("At least one finite series is required")
    values = np.concatenate(finite)
    if not values.size:
        raise ValueError("At least one finite series is required")
    return values


def _resource_series(monitor, include_post_training):
    """Return zero-based time and baseline-adjusted resource arrays."""
    times = np.asarray(monitor.timestamps, dtype=float)
    cpu = np.asarray(monitor.cpu_percent, dtype=float)
    ram = np.asarray(monitor.ram_gb, dtype=float)
    baselines = _resource_baselines(monitor, cpu, ram)
    start, end = getattr(monitor, "training_start_time", None), getattr(monitor, "training_end_time", None)
    if start is not None and end is not None and not include_post_training:
        return _training_resource_series(
            times, cpu, ram, start, end, baselines
        )
    origin = float(times[0]) if times.size else 0.0
    return (
        times - origin,
        _resource_delta(cpu, baselines[0]),
        _resource_delta(ram, baselines[1]),
        origin,
        baselines,
    )


def _training_resource_series(times, cpu, ram, start, end, baselines):
    """Restrict a resource trace to real samples from the training phase."""
    mask = (times >= start) & (times <= end)
    return (
        times[mask] - start,
        _resource_delta(cpu[mask], baselines[0]),
        _resource_delta(ram[mask], baselines[1]),
        start,
        baselines,
    )


def _resource_delta(values, baseline):
    """Return a nonnegative increase above the measured idle baseline."""
    return np.maximum(np.asarray(values, dtype=float) - baseline, 0.0)


def _resource_baselines(monitor, cpu, ram):
    """Return robust CPU and RAM levels measured before training."""
    phases = np.asarray(getattr(monitor, "phases", []), dtype=object)
    if phases.size == cpu.size and np.any(phases == "baseline"):
        mask = phases == "baseline"
        return float(np.median(cpu[mask])), float(np.median(ram[mask]))
    times = np.asarray(getattr(monitor, "timestamps", []), dtype=float)
    start = getattr(monitor, "training_start_time", None)
    if start is not None and times.size == cpu.size and np.any(times < start):
        mask = times < start
        return float(np.median(cpu[mask])), float(np.median(ram[mask]))
    return float(cpu[0]), float(ram[0])


def resource_baseline_summary(monitor) -> str:
    """Explain the measured idle baseline using values from one trace."""
    cpu = np.asarray(monitor.cpu_percent, dtype=float)
    ram = np.asarray(monitor.ram_gb, dtype=float)
    cpu_baseline, ram_baseline = _resource_baselines(monitor, cpu, ram)
    phases = np.asarray(getattr(monitor, "phases", []), dtype=object)
    samples = int(np.sum(phases == "baseline"))
    cores = int(getattr(monitor, "num_cores", 1))
    return (
        f"Poziom bazowy to mediana {samples} próbek procesu izolowanego "
        "i jego procesów potomnych, zebranych bezpośrednio przed uczeniem "
        f"(CPU: {polish_number(cpu_baseline, 2)}% łącznej mocy {cores} "
        f"rdzeni logicznych; RAM: {polish_number(ram_baseline, 2)} GB). "
        "Wykres przedstawia nieujemną różnicę: pomiar minus poziom bazowy."
    )


def _draw_resource_axis(axis, times, values, mean, label, color, limits):
    """Draw one baseline-adjusted resource series and its training mean."""
    axis.plot(
        times, values, color=color, linewidth=2, alpha=0.9,
        label="Zmiana względem poziomu bazowego",
        drawstyle=_resource_drawstyle(label), clip_on=False, zorder=4,
    )
    axis.axhline(
        mean, color=RESOURCE_MEAN_COLOR, linestyle="--", linewidth=2.4,
        label="Średnia podczas uczenia", clip_on=False, zorder=8,
    )
    _configure_resource_axis(axis, times, label, limits)
    _annotate_resource_mean(axis, mean, label)
    _place_resource_legend(axis)


def _annotate_resource_mean(axis, mean, axis_label):
    """Place the mean value directly above its horizontal reference."""
    unit = "%" if axis_label == CPU_AXIS_LABEL else " GB"
    axis.annotate(
        f"Średnia: {_compact_polish_number(mean)}{unit}",
        xy=(0.5, mean), xycoords=("axes fraction", "data"),
        xytext=(0, 6), textcoords="offset points",
        ha="center", va="bottom", color=RESOURCE_MEAN_COLOR,
        fontweight="bold", zorder=9,
        bbox={
            "facecolor": "white", "edgecolor": RESOURCE_MEAN_COLOR,
            "linewidth": 0.8, "alpha": 0.95, "pad": 2.0,
        },
    )


def _resource_drawstyle(axis_label):
    """Represent interval-averaged CPU samples without linear interpolation."""
    return "steps-post" if axis_label == CPU_AXIS_LABEL else "default"


def _place_resource_legend(axis):
    """Place the resource legend above data and phase markers."""
    axis.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=3, frameon=False,
    )


def _configure_resource_axis(axis, times, label, limits):
    """Apply labels, scientific limits and equal ticks to a resource axis."""
    axis.set_xlabel("Czas trwania pomiaru [s]")
    axis.set_ylabel(label)
    duration = max(float(times[-1]), 0.001)
    _configure_resource_time_axis(axis, duration)
    _set_compact_axis_ticks(axis, "y", limits, max_decimals=2)
    axis.grid(True, alpha=0.3)


def _configure_resource_time_axis(axis, duration):
    """End the time axis at the final real measurement."""
    lower, upper, ticks = _interior_axis_ticks(
        0.0, duration, target_intervals=8
    )
    axis.set_xlim(lower, upper)
    axis.xaxis.set_major_locator(FixedLocator(ticks))
    axis.xaxis.set_major_formatter(_formatter_for_ticks(ticks, 2))
    _ensure_zero_boundary_tick(axis, "x")


def _add_phase_markers(axes, monitor, time_origin):
    """Mark training boundaries on a complete baseline-to-post-training plot."""
    start, end = monitor.training_start_time, monitor.training_end_time
    if start is None or end is None:
        return
    start, end = start - time_origin, end - time_origin
    _draw_training_markers(axes, start, end)
    axes[0].axvspan(
        start, end, color=TRAINING_START_COLOR, alpha=0.05,
        label="_nolegend_",
    )
    _add_finalization_spans(axes, monitor, time_origin, end)
    _place_resource_legend(axes[0])


def _draw_training_markers(axes, start, end):
    """Draw consistent beginning and end markers on resource axes."""
    for axis in axes:
        axis.axvline(
            start, color=TRAINING_START_COLOR,
            linestyle=TRAINING_START_STYLE, linewidth=1.8,
            clip_on=False, zorder=7,
        )
        axis.axvline(
            end, color=TRAINING_END_COLOR,
            linestyle=TRAINING_END_STYLE, linewidth=1.8,
            clip_on=False, zorder=7,
        )


def _add_finalization_spans(axes, monitor, time_origin, training_end):
    """Identify model serialization and worker teardown after fitting."""
    finalization_end = _finalization_end_time(monitor)
    if finalization_end is None:
        return
    finalization_end -= time_origin
    if finalization_end <= training_end:
        return
    for axis in axes:
        axis.axvspan(
            training_end, finalization_end,
            facecolor=TRAINING_END_COLOR, alpha=FINALIZATION_ALPHA,
            edgecolor=TRAINING_END_COLOR, linewidth=0.8,
            label=FINALIZATION_LABEL, zorder=1,
        )


def _finalization_end_time(monitor):
    """Return process exit or the last real post-training observation."""
    process_exit = _process_exit_time(monitor)
    if process_exit is not None:
        return process_exit
    post_training = [
        timestamp for timestamp, phase
        in zip(monitor.timestamps, monitor.phases)
        if phase == "post_training"
    ]
    return float(post_training[-1]) if post_training else None


def _process_exit_time(monitor):
    """Return the first post-training sample after worker memory is released."""
    samples = zip(
        monitor.timestamps, monitor.phases,
        monitor.cpu_percent, monitor.ram_gb,
    )
    for timestamp, phase, cpu, ram in samples:
        if phase == "post_training" and np.isclose([cpu, ram], 0.0).all():
            return float(timestamp)
    return None


def _draw_resource_series(axes, times, cpu, ram, means, dataset_name):
    """Draw CPU and RAM series with shared dataset styling."""
    color = DATASET_COLORS[dataset_key(dataset_name)]
    for values, key, label, axis in (
        (cpu, "cpu", CPU_AXIS_LABEL, axes[0]),
        (ram, "ram", RAM_AXIS_LABEL, axes[1]),
    ):
        floor = None if np.isclose(means[key], 0.0) else 0.0
        limits = shared_limits(
            (values, [means[key], 0.0]), padding_ratio=0.06, floor=floor
        )
        _draw_resource_axis(axis, times, values, means[key], label, color, limits)


def _resource_mean(values):
    """Return mean of finite plotted resource values."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else 0.0


def _resource_training_mask(monitor):
    """Return samples belonging to measured training interval."""
    times = np.asarray(monitor.timestamps, dtype=float)
    start = getattr(monitor, "training_start_time", None)
    end = getattr(monitor, "training_end_time", None)
    if start is None or end is None:
        return np.ones(times.size, dtype=bool)
    return (times >= start) & (times <= end)


def _resource_means(monitor, baselines):
    """Return means matching baseline-adjusted training samples."""
    mask = _resource_training_mask(monitor)
    cpu = np.maximum(np.asarray(monitor.cpu_percent)[mask] - baselines[0], 0.0)
    ram = np.maximum(np.asarray(monitor.ram_gb)[mask] - baselines[1], 0.0)
    return {
        "cpu": _resource_mean(cpu),
        "ram": _resource_mean(ram),
    }


def plot_resource_usage(monitor, stats: dict, dataset_name: str, *, include_post_training: bool = False):
    """Create separate CPU/RAM axes for one dataset and monitoring phase."""
    figure, axes = plt.subplots(1, 2, figsize=(18, 6))
    if not stats or not monitor.timestamps:
        return figure, axes
    times, cpu, ram, origin, baselines = _resource_series(
        monitor, include_post_training
    )
    if not len(times):
        return figure, axes
    means = _resource_means(monitor, baselines)
    _draw_resource_series(axes, times, cpu, ram, means, dataset_name)
    if include_post_training:
        _add_phase_markers(axes, monitor, origin)
    figure.tight_layout()
    return figure, axes


def _copy_resource_axis(source_axis, monitor, time_origin):
    """Copy one resource axis into an independent figure."""
    figure, axis = plt.subplots(figsize=(12, 6))
    _copy_lines(source_axis, axis)
    _copy_phase_patches(source_axis, axis)
    axis.set_xlim(source_axis.get_xlim())
    axis.set_ylim(source_axis.get_ylim())
    axis.set_xlabel(source_axis.get_xlabel())
    axis.set_ylabel(source_axis.get_ylabel())
    _copy_resource_mean_annotation(axis)
    axis.xaxis.set_major_formatter(MAX_TWO_DECIMALS)
    axis.yaxis.set_major_formatter(MAX_TWO_DECIMALS)
    axis.xaxis.set_major_locator(FixedLocator(source_axis.get_xticks()))
    axis.yaxis.set_major_locator(FixedLocator(source_axis.get_yticks()))
    axis.grid(True, alpha=0.3)
    _copy_phase_markers(axis, monitor, time_origin)
    _place_resource_legend(axis)
    figure.tight_layout()
    return figure


def _copy_resource_mean_annotation(axis):
    """Recreate the mean annotation after copying an independent axis."""
    mean_lines = [
        line for line in axis.lines
        if line.get_label() == "Średnia podczas uczenia"
    ]
    if not mean_lines:
        return
    mean = float(np.asarray(mean_lines[0].get_ydata(), dtype=float)[0])
    _annotate_resource_mean(axis, mean, axis.get_ylabel())


def _copy_lines(source_axis, target_axis):
    """Copy plotted lines while preserving visual semantics."""
    for line in source_axis.lines:
        if _is_phase_marker(line):
            continue
        if _is_horizontal_reference(line):
            _copy_horizontal_reference(line, target_axis)
        else:
            _copy_data_line(line, target_axis)


def _copy_data_line(line, axis):
    """Copy a measured data line without changing its coordinates."""
    axis.plot(
        line.get_xdata(), line.get_ydata(), color=line.get_color(),
        linestyle=line.get_linestyle(), linewidth=line.get_linewidth(),
        alpha=line.get_alpha(), label=line.get_label(),
        drawstyle=line.get_drawstyle(), clip_on=False,
        zorder=line.get_zorder(),
    )


def _copy_horizontal_reference(line, axis):
    """Copy an axhline so it spans the complete independent figure."""
    value = float(np.asarray(line.get_ydata(), dtype=float)[0])
    axis.axhline(
        value, color=line.get_color(), linestyle=line.get_linestyle(),
        linewidth=line.get_linewidth(), alpha=line.get_alpha(),
        label=line.get_label(), clip_on=False, zorder=line.get_zorder(),
    )


def _is_horizontal_reference(line) -> bool:
    """Return whether a line uses normalized x coordinates as an axhline."""
    x_values = np.asarray(line.get_xdata(), dtype=float)
    y_values = np.asarray(line.get_ydata(), dtype=float)
    return (
        len(y_values) == 2
        and np.ptp(y_values) == 0
        and np.array_equal(x_values, [0.0, 1.0])
    )


def _is_phase_marker(line) -> bool:
    """Identify an axvline encoded in Matplotlib's normalized y coordinates."""
    x_values = np.asarray(line.get_xdata(), dtype=float)
    y_values = np.asarray(line.get_ydata(), dtype=float)
    return len(x_values) == 2 and np.ptp(x_values) == 0 and np.array_equal(y_values, [0.0, 1.0])


def _copy_phase_markers(axis, monitor, time_origin) -> None:
    """Draw full-height training boundaries on a copied resource axis."""
    start = getattr(monitor, "training_start_time", None)
    end = getattr(monitor, "training_end_time", None)
    if start is None or end is None:
        return
    _draw_copied_phase_marker(
        axis, start - time_origin, TRAINING_START_COLOR,
        TRAINING_START_STYLE, "Początek uczenia",
    )
    _draw_copied_phase_marker(
        axis, end - time_origin, TRAINING_END_COLOR,
        TRAINING_END_STYLE, TRAINING_END_LABEL,
    )


def _draw_copied_phase_marker(axis, position, color, style, label) -> None:
    """Draw one visible full-height boundary on a copied resource axis."""
    axis.axvline(
        position, color=color, linestyle=style, linewidth=1.8,
        label=label, clip_on=False, zorder=7,
    )


def _copy_phase_patches(source_axis, target_axis):
    """Copy phase spans from a combined resource axis."""
    for patch in source_axis.patches:
        lower, upper = _patch_x_bounds(patch)
        target_axis.axvspan(
            lower, upper, facecolor=patch.get_facecolor(),
            edgecolor=patch.get_edgecolor(), linewidth=patch.get_linewidth(),
            alpha=patch.get_alpha(), label=patch.get_label(),
            zorder=patch.get_zorder(),
        )


def _patch_x_bounds(patch):
    """Return data-space horizontal bounds of an axis span patch."""
    if hasattr(patch, "get_x") and hasattr(patch, "get_width"):
        lower = float(patch.get_x())
        return lower, lower + float(patch.get_width())
    vertices = np.asarray(patch.get_xy(), dtype=float)
    return float(vertices[:, 0].min()), float(vertices[:, 0].max())


def plot_resource_usage_separate(
    monitor, stats: dict, dataset_name: str, *, include_post_training: bool = False,
):
    """Create independent CPU and RAM figures for one monitored process."""
    combined_figure, axes = plot_resource_usage(
        monitor,
        stats,
        dataset_name,
        include_post_training=include_post_training,
    )
    origin = _resource_time_origin(monitor, include_post_training)
    cpu_figure = _copy_resource_axis(axes[0], monitor, origin)
    ram_figure = _copy_resource_axis(axes[1], monitor, origin)
    plt.close(combined_figure)
    return cpu_figure, ram_figure


def _resource_time_origin(monitor, include_post_training):
    """Return the real sample used as zero on a resource time axis."""
    if include_post_training and monitor.timestamps:
        return float(monitor.timestamps[0])
    return float(getattr(monitor, "training_start_time", 0.0) or 0.0)


def plot_confusion_matrix(confusion_matrix, dataset_name: str):
    """Create standalone confusion-matrix figure for one dataset."""
    figure, axis = plt.subplots(figsize=(8, 7))
    key = dataset_key(dataset_name)
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap=HEATMAP_CMAPS[key],
        xticklabels=["Brak phishingu", "Phishing"],
        yticklabels=["Brak phishingu", "Phishing"],
        ax=axis,
        annot_kws={"size": 14},
        cbar_kws={"label": "Liczba próbek"},
    )
    configure_count_colorbar(axis)
    axis.set_xlabel("Klasa przewidziana")
    axis.set_ylabel("Klasa rzeczywista")
    figure.tight_layout(pad=1.2)
    return figure, axis


def configure_count_colorbar(axis) -> None:
    """Format a sample-count color bar using integer tick labels."""
    if not axis.collections or axis.collections[0].colorbar is None:
        return
    colorbar = axis.collections[0].colorbar
    colorbar.locator = MaxNLocator(nbins=6, integer=True)
    colorbar.formatter = INTEGER_COUNT_FORMATTER
    colorbar.update_ticks()


def plot_roc_curve(
    y_true,
    y_probability,
    auc_roc: float,
    dataset_name: str,
):
    """Create standalone ROC figure for one dataset."""
    figure, axis = plt.subplots(figsize=(8, 7))
    key = dataset_key(dataset_name)
    _draw_roc_axis(axis, y_true, y_probability, auc_roc, DATASET_COLORS[key])
    figure.tight_layout()
    return figure, axis


def _draw_roc_axis(axis, y_true, y_probability, auc_roc, color):
    """Draw ROC line and its fixed [0, 1] coordinate system."""
    fpr, tpr, _ = roc_curve(y_true, y_probability)
    _plot_roc_lines(axis, fpr, tpr, auc_roc, color)
    _configure_roc_axis(axis)


def _plot_roc_lines(axis, fpr, tpr, auc_roc, color) -> None:
    """Draw the observed ROC curve and random-classifier reference."""
    axis.plot(
        fpr,
        tpr,
        color=color,
        linewidth=3,
        label=f"Pole pod krzywą ROC = {polish_number(auc_roc, 4)}",
    )
    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color=NEUTRAL_COLOR,
        linestyle="--",
        linewidth=1.5,
        label="Klasyfikator losowy",
    )


def _configure_roc_axis(axis) -> None:
    """Apply fixed probability bounds and full Polish axis labels."""
    axis.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        xlabel="Odsetek wyników fałszywie dodatnich",
        ylabel="Odsetek wyników prawdziwie dodatnich",
    )
    axis.xaxis.set_major_formatter(MAX_TWO_DECIMALS)
    axis.yaxis.set_major_formatter(MAX_TWO_DECIMALS)
    configure_axis_limit_ticks(axis, "x")
    configure_axis_limit_ticks(axis, "y")
    axis.legend(loc="lower right")
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)


def plot_precision_recall_curve(
    y_true,
    y_probability,
    average_precision: float,
    dataset_name: str,
):
    """Create a precision-recall curve with the class-prevalence baseline."""
    figure, axis = plt.subplots(figsize=(8, 7))
    precision, recall, _ = precision_recall_curve(y_true, y_probability)
    color = DATASET_COLORS[dataset_key(dataset_name)]
    axis.plot(
        recall,
        precision,
        color=color,
        linewidth=3,
        label=f"Średnia precyzja = {polish_number(average_precision, 4)}",
    )
    _configure_precision_recall_axis(axis, y_true)
    figure.tight_layout()
    return figure, axis


def _configure_precision_recall_axis(axis, y_true) -> None:
    """Apply a fixed probability scale and prevalence reference line."""
    prevalence = float(np.mean(np.asarray(y_true) == 1))
    axis.axhline(
        prevalence,
        color=NEUTRAL_COLOR,
        linestyle="--",
        linewidth=1.5,
        label=f"Udział klasy phishing = {polish_number(prevalence, 4)}",
    )
    axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0), xlabel="Czułość", ylabel="Precyzja")
    axis.xaxis.set_major_formatter(MAX_TWO_DECIMALS)
    axis.yaxis.set_major_formatter(MAX_TWO_DECIMALS)
    configure_axis_limit_ticks(axis, "x")
    configure_axis_limit_ticks(axis, "y")
    axis.legend(loc="lower left")
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)


def plot_score_curve(
    x_values, train_mean, train_std, validation_mean,
    validation_std, *, x_label: str,
    y_label: str = "Pole pod krzywą ROC",
    y_limits: tuple[float, float] | None = None, logarithmic_x: bool = False,
):
    """Create one train/validation score figure with stable semantic colors."""
    figure, axis = plt.subplots(figsize=(9, 6))
    x_values = _score_x_values(x_values, x_label)
    _plot_score_series(axis, x_values, train_mean, train_std, TRAIN_DATA_LABEL, TRAIN_COLOR, "-")
    _plot_score_series(
        axis, x_values, validation_mean, validation_std,
        VALIDATION_DATA_LABEL, VALIDATION_COLOR, "--",
    )
    limits = y_limits or _automatic_score_limits(
        y_label, train_mean, train_std, validation_mean, validation_std,
    )
    _configure_score_axis(axis, x_label, y_label, limits, logarithmic_x)
    figure.tight_layout(pad=1.2)
    return figure, axis


def _score_x_values(x_values, x_label):
    """Represent averaged fold sizes as the nearest whole sample count."""
    values = np.asarray(x_values, dtype=float)
    if _is_sample_count_axis(x_label):
        return np.rint(values).astype(int)
    return values


def _plot_score_series(axis, x_values, mean, deviation, label, color, linestyle):
    """Draw one score line and its standard-deviation band."""
    values, spread = np.asarray(mean), np.asarray(deviation)
    axis.plot(
        x_values, values, label=label, color=color, linewidth=2,
        marker="o", linestyle=linestyle, clip_on=False, zorder=3,
    )
    lower = np.maximum(values - spread, 0.0)
    axis.fill_between(
        x_values, lower, values + spread, color=color, alpha=0.2, label="_nolegend_"
    )


def _score_legend_handles(axis):
    """Return line handles and one neutral swatch for the uncertainty bands."""
    band = Patch(
        facecolor=MUTED_NEUTRAL_COLOR,
        edgecolor="none",
        alpha=0.2,
        label="± odchylenie standardowe",
    )
    return [*axis.lines, band]


def _automatic_score_limits(y_label, train_mean, train_std, validation_mean, validation_std):
    """Fit score curves to their observations and mathematical domain."""
    train, train_spread = np.asarray(train_mean), np.asarray(train_std)
    validation, validation_spread = np.asarray(validation_mean), np.asarray(validation_std)
    series = (
        train - train_spread, train + train_spread,
        validation - validation_spread, validation + validation_spread,
    )
    if y_label in {"ROC-AUC", "Pole pod krzywą ROC"}:
        return probability_limits(series)
    values = _finite_series(series)
    floor = 0.0 if np.all(values >= 0.0) else None
    ceiling = 1.0 if _is_bounded_score(y_label) else None
    return shared_limits(series, padding_ratio=0.08, floor=floor, ceiling=ceiling)


def _is_bounded_score(y_label):
    """Return whether a score is mathematically restricted to zero–one."""
    return y_label in {
        "ROC-AUC",
        "Pole pod krzywą ROC",
        "Dokładność",
        "Dokładność klasyfikacji",
        "Precyzja",
        "Czułość",
        "Miara F1",
        "Współczynnik błędu klasyfikacji",
    }


def _configure_score_axis(axis, x_label, y_label, y_limits, logarithmic_x):
    """Apply labels, optional limits and scale to a score axis."""
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.legend(handles=_score_legend_handles(axis), loc="best")
    axis.grid(True, alpha=0.3)
    _set_compact_axis_ticks(axis, "y", y_limits)
    _configure_observed_x_axis(axis, x_label, logarithmic_x)


def _configure_observed_x_axis(axis, x_label, logarithmic_x):
    """Configure an experiment axis using its observed x values."""
    configure_numeric_x_axis(
        axis,
        x_label,
        include_zero=_is_parameter_c_axis(x_label),
        logarithmic_x=logarithmic_x,
    )


def configure_numeric_x_axis(
    axis, x_label, *, include_zero=False, logarithmic_x=False,
    tick_step=None, tick_anchor=0.0,
):
    """Label a numeric axis using equal scientific intervals and reserve."""
    values = _observed_x_values(axis)
    axis.set_xlabel(x_label)
    if not values.size:
        return
    if logarithmic_x:
        _configure_logarithmic_x_axis(
            axis, values, show_observed_ticks=_is_parameter_c_axis(x_label)
        )
        return
    lower, upper, ticks = _numeric_x_scale(
        values, x_label, include_zero, tick_step, tick_anchor
    )
    _apply_numeric_x_ticks(axis, x_label, lower, upper, ticks)


def _apply_numeric_x_ticks(axis, x_label, lower, upper, ticks) -> None:
    """Apply numeric limits, labels and an explicit zero boundary tick."""
    axis.set_xlim(lower, upper)
    axis.xaxis.set_major_locator(FixedLocator(ticks))
    axis.xaxis.set_major_formatter(_numeric_x_formatter(x_label, ticks))
    _ensure_zero_boundary_tick(axis, "x")


def _numeric_x_formatter(x_label, ticks):
    """Use zero decimal places for axes representing sample counts."""
    if _is_sample_count_axis(x_label):
        return INTEGER_COUNT_FORMATTER
    return _formatter_for_ticks(ticks, 2)


def _numeric_x_scale(values, x_label, include_zero, tick_step, tick_anchor):
    """Return limits and ticks appropriate for the represented quantity."""
    lower, upper = _reserved_linear_range(values, include_zero)
    if tick_step is not None:
        return _predefined_axis_ticks(
            lower, upper, tick_step, tick_anchor, rounded_bounds=True
        )
    if _is_sample_count_axis(x_label) and not include_zero:
        return _sample_count_scale(values)
    if _is_parameter_c_axis(x_label) and not include_zero:
        return _interior_axis_ticks(lower, upper, target_intervals=6)
    return _interior_axis_ticks(lower, upper, target_intervals=9)


def _sample_count_scale(values, maximum_ticks=10):
    """Frame sample observations with readable, equal integer intervals."""
    integer_values = _integer_sample_values(values)
    first, last = float(integer_values[0]), float(integer_values[-1])
    if np.isclose(first, last):
        return first - 1.0, last + 1.0, integer_values
    span = last - first
    reserve = max(span * 0.03, 1.0)
    lower, upper = max(0.0, first - reserve), last + reserve
    step = _nice_integer_step(span, maximum_ticks - 1)
    first_tick = np.ceil(lower / step) * step
    last_tick = np.floor(upper / step) * step
    ticks = np.arange(first_tick, last_tick + step * 0.5, step)
    return lower, upper, _clean_ticks(ticks)


def _integer_sample_values(values):
    """Validate that values representing sample counts are whole numbers."""
    rounded = np.rint(np.asarray(values, dtype=float))
    if not np.allclose(values, rounded, atol=1e-9):
        raise ValueError("Sample counts must be integers")
    return rounded


def _is_sample_count_axis(x_label):
    """Return whether an axis represents a number of observations."""
    return x_label.startswith(SAMPLE_COUNT_AXIS_LABEL) or (
        "Liczba" in x_label and "próbek" in x_label
    )


def _is_parameter_c_axis(x_label):
    """Return whether an axis represents the strictly positive C parameter."""
    return x_label.startswith("Wartość parametru C")


def configure_axis_limit_ticks(
    axis, axis_name, *, floor=None, ceiling=None, max_decimals=None
):
    """Replace irregular endpoint ticks with equal scientific intervals."""
    if axis_name == "x" and max_decimals is None:
        max_decimals = 2
    lower, upper = _clipped_axis_limits(axis, axis_name, floor, ceiling)
    is_count = _is_sample_count_axis(_axis_label(axis, axis_name))
    lower, upper, ticks = _axis_tick_spec(
        lower, upper, is_count, floor, ceiling
    )
    formatter = _axis_tick_formatter(ticks, is_count, max_decimals)
    _apply_fixed_axis_ticks(
        axis, axis_name, lower, upper, ticks, formatter
    )


def _clipped_axis_limits(axis, axis_name, floor, ceiling):
    """Read axis limits and clip them to an optional semantic domain."""
    getter = axis.get_xlim if axis_name == "x" else axis.get_ylim
    lower, upper = getter()
    lower = max(floor, lower) if floor is not None else lower
    upper = min(ceiling, upper) if ceiling is not None else upper
    return lower, upper


def _axis_tick_spec(lower, upper, is_count, floor, ceiling):
    """Build ticks for either sample counts or continuous quantities."""
    if is_count:
        return _integer_count_axis_scale(lower, upper)
    lower, upper, ticks = _equal_axis_ticks(lower, upper)
    return _clip_axis_ticks(lower, upper, ticks, floor, ceiling)


def _axis_tick_formatter(ticks, is_count, max_decimals):
    """Choose an integer formatter for counts and a numeric one otherwise."""
    if is_count:
        return INTEGER_COUNT_FORMATTER
    return _formatter_for_ticks(ticks, max_decimals)


def _apply_fixed_axis_ticks(
    axis, axis_name, lower, upper, ticks, formatter
):
    """Apply fixed limits, locations and labels to one numeric axis."""
    setter = axis.set_xlim if axis_name == "x" else axis.set_ylim
    setter(lower, upper)
    target = axis.xaxis if axis_name == "x" else axis.yaxis
    target.set_major_locator(FixedLocator(ticks))
    target.set_major_formatter(formatter)
    _ensure_zero_boundary_tick(axis, axis_name)


def _axis_label(axis, axis_name):
    """Return the visible label for the selected Matplotlib axis."""
    return axis.get_xlabel() if axis_name == "x" else axis.get_ylabel()


def _integer_count_axis_scale(lower, upper, target_intervals=6):
    """Cover a sample-count range with equal, integer-valued ticks."""
    lower, upper = float(np.floor(lower)), float(np.ceil(upper))
    step = _nice_integer_step(upper - lower, target_intervals)
    rounded_lower = np.floor(lower / step) * step
    rounded_upper = np.ceil(upper / step) * step
    if lower >= 0.0:
        rounded_lower = max(0.0, rounded_lower)
    ticks = np.arange(rounded_lower, rounded_upper + step * 0.5, step)
    return rounded_lower, rounded_upper, _clean_ticks(ticks)


def _nice_integer_step(span, target_intervals):
    """Choose a conventional positive integer interval for count ticks."""
    rough = max(float(span) / target_intervals, 1.0)
    magnitude = 10.0 ** np.floor(np.log10(rough))
    normalized = rough / magnitude
    multiplier = next(
        value for value in (1.0, 2.0, 5.0, 10.0)
        if normalized <= value
    )
    return max(1, int(round(multiplier * magnitude)))


def _set_compact_axis_ticks(
    axis, axis_name, limits, *, max_decimals=None
):
    """Keep reserved bounds exact and place regular ticks inside them."""
    lower, upper, ticks = _interior_axis_ticks(*limits, target_intervals=6)
    setter = axis.set_xlim if axis_name == "x" else axis.set_ylim
    target = axis.xaxis if axis_name == "x" else axis.yaxis
    setter(lower, upper)
    target.set_major_locator(FixedLocator(ticks))
    target.set_major_formatter(_formatter_for_ticks(ticks, max_decimals))
    _ensure_zero_boundary_tick(axis, axis_name)


def _ensure_zero_boundary_tick(axis, axis_name) -> None:
    """Explicitly label zero whenever it is a numeric axis boundary."""
    getter = axis.get_xlim if axis_name == "x" else axis.get_ylim
    lower, _ = getter()
    if not np.isclose(lower, 0.0):
        return
    target = axis.xaxis if axis_name == "x" else axis.yaxis
    ticks = np.asarray(target.get_majorticklocs(), dtype=float)
    if np.any(np.isclose(ticks, 0.0)):
        return
    target.set_major_locator(FixedLocator(np.insert(ticks, 0, 0.0)))


def _clip_axis_ticks(lower, upper, ticks, floor, ceiling):
    """Clip rounded bounds to a meaningful numeric domain."""
    lower = max(floor, lower) if floor is not None else lower
    upper = min(ceiling, upper) if ceiling is not None else upper
    visible = ticks[(ticks >= lower) & (ticks <= upper)]
    if visible.size < 2:
        return _interior_axis_ticks(lower, upper, target_intervals=6)
    return lower, upper, _clean_ticks(visible)


def _equal_axis_ticks(lower, upper, *, target_intervals=6):
    """Return rounded bounds and equally spaced ticks covering an interval."""
    if np.isclose([lower, upper], [0.0, 1.0]).all():
        return 0.0, 1.0, np.linspace(0.0, 1.0, 6)
    if np.isclose([lower, upper], [0.0, 100.0]).all():
        return 0.0, 100.0, np.linspace(0.0, 100.0, 6)
    step = _nice_axis_step(upper - lower, target_intervals)
    rounded_lower = np.floor(lower / step) * step
    rounded_upper = np.ceil(upper / step) * step
    if lower >= 0.0:
        rounded_lower = max(0.0, rounded_lower)
    ticks = np.arange(rounded_lower, rounded_upper + step * 0.5, step)
    return float(rounded_lower), float(rounded_upper), _clean_ticks(ticks)


def _nice_axis_step(span, target_intervals):
    """Choose a conventional 1–2–2.5–4–5–10 tick step."""
    rough = max(float(span) / target_intervals, np.finfo(float).eps)
    magnitude = 10.0 ** np.floor(np.log10(rough))
    normalized = rough / magnitude
    multiplier = next(
        value for value in (1.0, 2.0, 2.5, 4.0, 5.0, 10.0)
        if normalized <= value
    )
    return float(multiplier * magnitude)


def _clean_ticks(ticks):
    """Remove floating-point noise from generated tick locations."""
    cleaned = np.asarray([float(f"{value:.12g}") for value in ticks])
    threshold = max(float(np.max(np.abs(cleaned))), 1.0) * 1e-12
    cleaned[np.abs(cleaned) < threshold] = 0.0
    return cleaned


def _formatter_for_ticks(ticks, max_decimals=None):
    """Return a consistent formatter precise enough for one equal tick step."""
    if max_decimals == 2:
        return _two_decimal_axis_formatter(ticks)
    decimals = _tick_decimals(ticks)
    if max_decimals is not None:
        decimals = min(decimals, max_decimals)
    return FuncFormatter(
        lambda value, _position: polish_number(value, decimals)
    )


def _two_decimal_axis_formatter(ticks):
    """Use compact decimals or precise scientific notation when required."""
    tick_format = plotly_tick_format(ticks)
    if tick_format == ".2e":
        return FuncFormatter(_scientific_polish_number)
    return MAX_TWO_DECIMALS


def plotly_tick_format(ticks) -> str:
    """Choose a consistent Plotly format without hiding a small tick step."""
    values = np.asarray(ticks, dtype=float)
    if values.size == 0 or np.allclose(values, np.round(values), atol=1e-9):
        return ".0f"
    if np.unique(np.round(values, 2)).size == values.size:
        return ".2~f"
    return ".2e"


def _scientific_polish_number(value, _position=None):
    """Format very small axis values with a two-decimal mantissa."""
    if np.isclose(value, 0.0, atol=1e-15):
        return "0"
    mantissa, exponent = f"{value:.2e}".split("e")
    superscript = str(int(exponent)).translate(
        str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    )
    return f"{mantissa.replace('.', ',')}×10{superscript}"


def _tick_decimals(ticks):
    """Return the fewest decimals that represent the common tick step."""
    values = np.asarray(ticks, dtype=float)
    if values.size < 2:
        return 2
    step = abs(float(values[1] - values[0]))
    for decimals in range(7):
        scaled = step * 10 ** decimals
        if np.isclose(scaled, round(scaled), rtol=0.0, atol=1e-8):
            return decimals
    return 6


def _observed_x_values(axis):
    """Return sorted finite x values represented by plotted lines."""
    arrays = [np.asarray(line.get_xdata(), dtype=float) for line in axis.lines]
    finite = [values[np.isfinite(values)] for values in arrays if values.size]
    if not finite:
        return np.asarray([], dtype=float)
    return np.unique(np.concatenate(finite))


def _reserved_linear_range(values, include_zero):
    """Use exact observed bounds unless zero is a real requested reference."""
    first, last = float(values[0]), float(values[-1])
    if include_zero:
        return 0.0, max(last * 1.05, 1.0)
    if not np.isclose(first, last):
        return first, last
    reserve = max(abs(first) * 0.05, 1e-9)
    return first - reserve, last + reserve


def _configure_logarithmic_x_axis(axis, values, show_observed_ticks=False):
    """Configure a logarithmic experiment axis with multiplicative reserve."""
    positive = values[values > 0.0]
    if not positive.size:
        raise ValueError("A logarithmic axis requires positive x values")
    factor = 10.0 ** 0.08
    axis.set_xscale("log")
    axis.set_xlim(float(positive[0] / factor), float(positive[-1] * factor))
    locator = (
        FixedLocator(positive)
        if show_observed_ticks
        else LogLocator(base=10.0, numticks=8)
    )
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(FuncFormatter(_format_log_tick))


def _format_log_tick(value, _position=None):
    """Format logarithmic ticks without scientific-notation exponents."""
    if value == 0.0:
        return "0"
    decimals = max(0, min(6, int(np.ceil(-np.log10(abs(value))))))
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text.replace(".", ",")


def plot_final_classification_error(train_error: float, test_error: float):
    """Compare final classification error on training and held-out test data."""
    figure, axis = plt.subplots(figsize=(7, 6))
    labels = [TRAIN_DATA_LABEL, TEST_DATA_LABEL]
    values = [float(train_error), float(test_error)]
    bars = _train_test_bars(axis, labels, values)
    axis.set_xlabel("Zbiór użyty do obliczenia miary")
    axis.set_ylabel("Współczynnik błędu klasyfikacji")
    configure_dynamic_axis(axis, "y", ([0.0], values), floor=0.0, ceiling=1.0)
    axis.grid(True, axis="y", alpha=0.3)
    _annotate_bars(axis, bars, values, decimals=4)
    axis.legend(handles=_train_test_legend_handles(), frameon=False)
    figure.tight_layout()
    return figure, axis


def _train_test_legend_handles():
    """Return semantic legend entries for training and test data."""
    return [
        Patch(facecolor=TRAIN_COLOR, label=TRAIN_DATA_LABEL),
        Patch(facecolor=VALIDATION_COLOR, label=TEST_DATA_LABEL),
    ]


def _train_test_bars(axis, labels, values):
    """Draw train/test bars using stable semantic colors."""
    return axis.bar(
        labels,
        values,
        color=[TRAIN_COLOR, VALIDATION_COLOR],
        alpha=0.85,
    )


def plot_final_train_test_metric_groups(
    train_metrics: dict, test_metrics: dict,
) -> dict:
    """Create two readable train/test classification-metric figures."""
    return {
        group: plot_final_train_test_metrics(
            train_metrics, test_metrics, metric_keys=keys,
        )[0]
        for group, keys in CLASSIFICATION_METRIC_GROUPS.items()
    }


def plot_final_train_test_metrics(
    train_metrics: dict, test_metrics: dict, *, metric_keys=None,
):
    """Create one grouped comparison of train and test classification metrics."""
    labels, train_values, test_values = _classification_metric_plot_data(
        train_metrics, test_metrics, metric_keys,
    )
    positions = np.arange(len(labels), dtype=float)
    figure, axis = plt.subplots(figsize=(max(9, 2.2 * len(labels)), 7))
    train_bars, test_bars = _grouped_metric_bars(
        axis, positions, train_values, test_values
    )
    _configure_grouped_metric_axis(
        axis, positions, labels, train_values, test_values
    )
    _annotate_bars(axis, train_bars, train_values, decimals=4)
    _annotate_bars(axis, test_bars, test_values, decimals=4)
    figure.tight_layout()
    return figure, axis


def _classification_metric_plot_data(
    train_metrics, test_metrics, metric_keys,
):
    """Return labels and paired values selected for one metric figure."""
    keys = _selected_classification_metric_keys(
        train_metrics, test_metrics, metric_keys,
    )
    labels = [metric_display_name(key) for key in keys]
    train_values = [float(train_metrics[key]) for key in keys]
    test_values = [float(test_metrics[key]) for key in keys]
    return labels, train_values, test_values


def _selected_classification_metric_keys(
    train_metrics, test_metrics, metric_keys,
):
    """Return requested metrics that exist in both evaluated datasets."""
    available = _shared_classification_metric_keys(
        train_metrics, test_metrics,
    )
    if metric_keys is None:
        return available
    return [key for key in metric_keys if key in available]


def _grouped_metric_bars(axis, positions, train_values, test_values):
    """Draw paired train and test bars for every classification metric."""
    train_bars = axis.bar(
        positions - 0.19, train_values, 0.38,
        color=TRAIN_COLOR, label=TRAIN_DATA_LABEL,
    )
    test_bars = axis.bar(
        positions + 0.19, test_values, 0.38,
        color=VALIDATION_COLOR, label=TEST_DATA_LABEL,
    )
    return train_bars, test_bars


def _shared_classification_metric_keys(train_metrics, test_metrics):
    """Return ordered classification metrics present in both dictionaries."""
    keys = (
        "accuracy", "precision", "recall",
        "false_positive_rate", "error_rate", "f1", "roc_auc",
    )
    return [
        key for key in keys if key in train_metrics and key in test_metrics
    ]


def _configure_grouped_metric_axis(
    axis, positions, labels, train_values, test_values,
):
    """Configure the single train/test classification-metric figure."""
    axis.set_xlabel("Miara jakości klasyfikacji")
    axis.set_ylabel("Wartość miary")
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    configure_dynamic_axis(
        axis, "y", ([0.0], train_values, test_values),
        floor=0.0, ceiling=1.0,
    )
    axis.yaxis.set_major_formatter(MAX_TWO_DECIMALS)
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=2, frameon=False,
    )


def plot_parameter_train_test_error(
    parameter_values,
    train_errors,
    test_errors,
    *,
    x_label: str,
    logarithmic_x: bool = False,
):
    """Compare classification error on train and untouched test by parameter."""
    figure, axis = plt.subplots(figsize=(9, 6))
    _plot_error_pair(
        axis, parameter_values, train_errors, test_errors, TEST_DATA_LABEL
    )
    _configure_error_axis(axis, x_label, train_errors, test_errors, logarithmic_x)
    figure.tight_layout()
    return figure, axis


def plot_parameter_train_validation_error(
    parameter_values,
    train_errors,
    validation_errors,
    *,
    x_label: str,
    logarithmic_x: bool = False,
):
    """Compare train and held-out validation error without using the test set."""
    figure, axis = plt.subplots(figsize=(9, 6))
    _plot_error_pair(
        axis, parameter_values, train_errors, validation_errors,
        VALIDATION_DATA_LABEL,
    )
    _configure_error_axis(axis, x_label, train_errors, validation_errors, logarithmic_x)
    figure.tight_layout()
    return figure, axis


def _plot_error_pair(axis, x_values, train_values, other_values, other_label):
    """Draw training error beside one validation or test error series."""
    _plot_error_series(
        axis, x_values, train_values, TRAIN_DATA_LABEL, TRAIN_COLOR, "-"
    )
    _plot_error_series(
        axis, x_values, other_values, other_label, VALIDATION_COLOR, "--"
    )


def _plot_error_series(axis, x_values, values, label, color, linestyle):
    """Draw one train/test error curve."""
    axis.plot(
        x_values, values, label=label, color=color, linewidth=2,
        marker="o", linestyle=linestyle, clip_on=False, zorder=3,
    )


def _configure_error_axis(axis, x_label, train_errors, test_errors, logarithmic_x):
    """Configure common labels and bounds for error curves."""
    axis.set(xlabel=x_label, ylabel="Współczynnik błędu klasyfikacji")
    configure_dynamic_axis(
        axis, "y", (train_errors, test_errors), floor=0.0, ceiling=1.0
    )
    _configure_observed_x_axis(axis, x_label, logarithmic_x)
    axis.legend()
    axis.grid(True, alpha=0.3)


def plot_feature_ranking(dataframe, dataset_name: str, *, x_column: str, x_label: str):
    """Create one horizontal feature-ranking figure."""
    figure, axis = plt.subplots(figsize=(9, 7))
    sns.barplot(
        x=x_column,
        y="Cecha",
        data=dataframe,
        color=PRIMARY_COLOR,
        saturation=1.0,
        ax=axis,
    )
    axis.set_xlabel(x_label)
    axis.set_ylabel("Cecha")
    configure_dynamic_axis(
        axis, "x", ([0.0], dataframe[x_column]), floor=0.0
    )
    axis.grid(True, alpha=0.3, axis="x")
    _add_dataset_legend(axis, dataset_name, PRIMARY_COLOR)
    figure.tight_layout()
    return figure, axis


def _add_dataset_legend(axis, dataset_name, color) -> None:
    """Add one explicit dataset entry to a standalone figure."""
    axis.legend(
        handles=[Patch(facecolor=color, label=dataset_name)],
        frameon=False,
    )


def plot_training_time(training_time: float, dataset_name: str):
    """Create a standalone zero-based chart of model-training time."""
    return _plot_scalar_measure(
        "Czas uczenia modelu",
        float(training_time),
        "Czas uczenia [s]",
        dataset_name,
        decimals=2,
    )


def plot_training_time_comparison(training_times: dict):
    """Compare full and optimized model-training time with two bars."""
    labels = list(training_times)
    values = [float(training_times[label]) for label in labels]
    colors = [DATASET_COLORS[dataset_key(label)] for label in labels]
    figure, axis = plt.subplots(figsize=(8, 6))
    bars = axis.bar(labels, values, color=colors, alpha=0.85)
    axis.set_xlabel("Wariant danych")
    axis.set_ylabel("Czas uczenia modelu [s]")
    configure_dynamic_axis(axis, "y", ([0.0], values), floor=0.0)
    axis.grid(True, alpha=0.3, axis="y")
    _annotate_bars(axis, bars, values, decimals=2)
    _add_training_time_legend(axis, colors, labels)
    figure.tight_layout()
    return figure, axis


def _add_training_time_legend(axis, colors, labels) -> None:
    """Add both data variants above the training-time comparison."""
    axis.legend(
        handles=[
            Patch(facecolor=color, label=label)
            for color, label in zip(colors, labels)
        ],
        loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=2, frameon=False,
    )


def training_time_measurement_summary() -> str:
    """Explain the exact scope of the reported model-training time."""
    return (
        "Czas uczenia jest rzeczywistym czasem upływu mierzonym funkcją "
        "time.perf_counter() w izolowanym procesie. Pomiar rozpoczyna się "
        "bezpośrednio przed wywołaniem model.fit(...) i kończy po jego "
        "zakończeniu. Obejmuje operacje wykonywane wewnątrz fit, lecz nie "
        "obejmuje przygotowania danych, doboru cech i parametrów, uruchomienia "
        "procesu, serializacji modelu, ewaluacji ani tworzenia wykresów."
        " Czasy należy porównywać przy sekwencyjnym wykonaniu notebooków w "
        "tym samym środowisku i bez konkurencyjnych obciążeń systemu."
    )


def feature_count_measurement_summary() -> str:
    """Explain which columns are included in the reported feature count."""
    return (
        "Liczba cech oznacza liczbę kolumn macierzy uczącej przekazanej do "
        "model.fit(...) po przygotowaniu pełnego albo zoptymalizowanego "
        "wariantu danych. Zmienna docelowa nie jest liczona jako cecha."
    )


def feature_count_comparison_rows(
    full_count: int, optimized_count: int,
) -> list[dict]:
    """Return a compact comparison of full and optimized feature counts."""
    full = int(full_count)
    optimized = int(optimized_count)
    reduction = full - optimized
    percentage = 100.0 * reduction / full if full else 0.0
    return [
        _feature_count_row("Pełny zbiór", full, 0, 0.0),
        _feature_count_row(
            "Zoptymalizowany zbiór", optimized, reduction, percentage,
        ),
    ]


def _feature_count_row(name, count, reduction, percentage) -> dict:
    """Return one display row for a feature-count comparison."""
    return {
        "Wariant danych": name,
        "Liczba cech": count,
        "Redukcja liczby cech": reduction,
        "Redukcja [%]": round(percentage, 2),
    }


def _plot_scalar_measure(label, value, y_label, dataset_name, *, decimals):
    """Create a single-measure chart without mixing incompatible units."""
    figure, axis = plt.subplots(figsize=(7, 6))
    color = DATASET_COLORS[dataset_key(dataset_name)]
    bars = axis.bar([label], [value], color=color, alpha=0.85)
    axis.set_xlabel("Mierzona charakterystyka modelu")
    axis.set_ylabel(y_label)
    configure_dynamic_axis(axis, "y", ([0.0], [value]), floor=0.0)
    axis.yaxis.set_major_formatter(MAX_TWO_DECIMALS)
    axis.grid(True, alpha=0.3, axis="y")
    _annotate_bars(axis, bars, [value], decimals=decimals)
    axis.legend(
        handles=[Patch(facecolor=color, label=dataset_name)],
        frameon=False,
    )
    figure.tight_layout()
    return figure, axis


def _annotate_bars(axis, bars, values, *, decimals: int) -> None:
    """Place consistently formatted values above vertical bars."""
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            polish_number(value, decimals),
            ha="center",
            va="bottom",
        )


def plot_probability_bars(axis, probabilities, *, sample_label: str) -> None:
    """Draw class probabilities as comparable horizontal percentage bars."""
    values = np.asarray(probabilities, dtype=float)
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("Class probabilities or votes must have a positive sum")
    values = values / total * 100.0
    labels = ["Brak phishingu", "Phishing"]
    colors = [CLASS_COLORS["safe"], CLASS_COLORS["phishing"]]
    bars = axis.barh(labels, values, color=colors, alpha=0.85)
    axis.set_xlim(0.0, 100.0)
    axis.set_xlabel("Przewidywane prawdopodobieństwo klasy [%]")
    axis.set_ylabel("Klasa")
    axis.xaxis.set_major_formatter(MAX_TWO_DECIMALS)
    configure_axis_limit_ticks(axis, "x")
    axis.grid(True, alpha=0.3, axis="x")
    _annotate_probability_bars(axis, bars, values)
    _annotate_sample(axis, sample_label)


def plot_probability_figure(probabilities, *, sample_label: str):
    """Create one standalone class-probability figure for one sample."""
    figure, axis = plt.subplots(figsize=(9, 5))
    plot_probability_bars(axis, probabilities, sample_label=sample_label)
    figure.tight_layout()
    return figure, axis


def plot_sample_probability_prediction(
    phishing_probability, true_label, sample_identifier, *,
    decision_threshold: float = 0.5, language: str = "pl",
):
    """Visualize probabilities and the decision for one labelled test sample."""
    values, actual = _sample_probability_values(
        phishing_probability, true_label,
    )
    text = _sample_probability_text(language)
    predicted = int(values[1] >= float(decision_threshold) * 100.0)
    figure, axis = plt.subplots(figsize=(10, 5.5))
    _draw_sample_probability_bars(axis, values, predicted, text)
    _configure_sample_probability_axis(
        axis, actual, predicted, sample_identifier, text,
    )
    _add_sample_probability_legend(
        axis, predicted, decision_threshold, text, language,
    )
    figure.tight_layout()
    return figure, axis


def _sample_probability_values(phishing_probability, true_label):
    """Validate and return both probabilities and the true binary label."""
    probability = float(phishing_probability)
    actual = int(true_label)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("Phishing probability must be within [0, 1]")
    if actual not in (0, 1):
        raise ValueError("True label must be binary")
    return np.asarray([1.0 - probability, probability]) * 100.0, actual


def _draw_sample_probability_bars(axis, values, predicted, text) -> None:
    """Draw two comparable probability bars and outline the model decision."""
    colors = [CLASS_COLORS["safe"], CLASS_COLORS["phishing"]]
    bars = axis.barh(
        [0, 1], values, color=colors, alpha=0.85,
        edgecolor=[
            NEUTRAL_COLOR if index == predicted else color
            for index, color in enumerate(colors)
        ],
        linewidth=[3 if index == predicted else 1 for index in range(2)],
    )
    for bar, value in zip(bars, values):
        axis.text(
            min(value + 1.5, 98.0), bar.get_y() + bar.get_height() / 2,
            text["number"](value), ha="right" if value > 90 else "left",
            va="center", fontweight="bold",
        )


def _configure_sample_probability_axis(
    axis, actual, predicted, sample_identifier, text,
) -> None:
    """Configure axes and state the true and predicted classes explicitly."""
    labels = list(text["classes"])
    labels[actual] = f"{labels[actual]} — {text['actual_marker']}"
    axis.set_xlim(0.0, 100.0)
    axis.set_yticks([0, 1], labels)
    axis.set_xlabel(text["x_label"])
    axis.set_ylabel(text["y_label"])
    axis.invert_yaxis()
    configure_axis_limit_ticks(axis, "x")
    axis.grid(True, axis="x", alpha=0.3)
    axis.text(
        0.0, 1.08, text["summary"](
            sample_identifier, actual, predicted, actual == predicted,
        ),
        transform=axis.transAxes, ha="left", va="bottom",
    )


def _add_sample_probability_legend(
    axis, predicted, threshold, text, language,
) -> None:
    """Explain colors, decision outline and phishing threshold."""
    threshold_value = _localized_percentage(threshold * 100.0, language)
    axis.axvline(
        threshold * 100.0, color=NEUTRAL_COLOR, linestyle="--", linewidth=1.5,
    )
    axis.legend(
        handles=_sample_probability_legend_handles(
            predicted, threshold_value, text,
        ),
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=2, frameon=False,
    )


def _sample_probability_legend_handles(predicted, threshold_value, text):
    """Return semantic legend handles for a one-sample decision plot."""
    return [
        Patch(facecolor=CLASS_COLORS["safe"],
              label=text["safe_probability"]),
        Patch(facecolor=CLASS_COLORS["phishing"],
              label=text["phishing_probability"]),
        Patch(
            facecolor="none", edgecolor=NEUTRAL_COLOR, linewidth=3,
            label=text["decision"](text["classes"][predicted]),
        ),
        Line2D(
            [0], [0], color=NEUTRAL_COLOR, linestyle="--", linewidth=1.5,
            label=text["threshold"](threshold_value),
        ),
    ]


def _sample_probability_text(language):
    """Return localized labels for one-sample probability figures."""
    if language == "pl":
        return _polish_sample_probability_text()
    if language == "en":
        return _english_sample_probability_text()
    raise ValueError("Language must be 'pl' or 'en'")


def _polish_sample_probability_text():
    """Return Polish one-sample probability labels."""
    classes = ["Brak phishingu", "Phishing"]
    return {
        "classes": classes,
        "number": lambda value: f"{polish_number(value, 2)}%",
        "x_label": "Prawdopodobieństwo przewidywane przez model [%]",
        "y_label": "Klasa",
        "actual_marker": "etykieta rzeczywista",
        "safe_probability": "Prawdopodobieństwo braku phishingu",
        "phishing_probability": "Prawdopodobieństwo phishingu",
        "decision": lambda name: f"Decyzja modelu: {name}",
        "threshold": lambda value: f"Próg decyzji phishing: {value}",
        "summary": lambda identifier, actual, predicted, correct: (
            f"Próbka #{identifier} | Etykieta rzeczywista: {classes[actual]} | "
            f"Decyzja modelu: {classes[predicted]} | "
            f"Wynik: {'poprawny' if correct else 'błędny'}"
        ),
    }


def _english_sample_probability_text():
    """Return English one-sample probability labels."""
    classes = ["Non-phishing", "Phishing"]
    return {
        "classes": classes,
        "number": lambda value: f"{value:.2f}%",
        "x_label": "Probability predicted by the model [%]",
        "y_label": "Class",
        "actual_marker": "actual label",
        "safe_probability": "Non-phishing probability",
        "phishing_probability": "Phishing probability",
        "decision": lambda name: f"Model decision: {name}",
        "threshold": lambda value: f"Phishing decision threshold: {value}",
        "summary": lambda identifier, actual, predicted, correct: (
            f"Sample #{identifier} | Actual label: {classes[actual]} | "
            f"Model decision: {classes[predicted]} | "
            f"Result: {'correct' if correct else 'incorrect'}"
        ),
    }


def _localized_percentage(value, language):
    """Format one percentage according to the requested language."""
    if language == "pl":
        return f"{polish_number(value, 2)}%"
    return f"{value:.2f}%"


def _annotate_sample(axis, sample_label):
    """Add sample identity as metadata rather than an embedded plot title."""
    axis.text(
        0.0, 1.03, sample_label, transform=axis.transAxes,
        ha="left", va="bottom", fontsize=10,
    )


def _annotate_probability_bars(axis, bars, values) -> None:
    """Place percentage labels without overflowing the fixed axis."""
    for bar, value in zip(bars, values):
        axis.text(
            min(value + 1.0, 99.0),
            bar.get_y() + bar.get_height() / 2,
            f"{polish_number(value, 2)}%",
            va="center",
            ha="right" if value > 95.0 else "left",
        )
