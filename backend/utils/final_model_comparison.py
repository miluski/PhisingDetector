"""Build thesis-ready comparisons from final per-model result tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from utils.notebook_plot_style import (
    DATASET_COLORS,
    QUATERNARY_COLOR,
    SECONDARY_COLOR,
    configure_axis_limit_ticks,
    polish_number,
    save_comparison_figure,
)


MODEL_FILES = {
    "Regresja logistyczna": "logistic_regression_comparison.csv",
    "Random Forest": "random_forest_comparison.csv",
    "XGBoost": "xgboost_comparison.csv",
    "SVM": "svm_comparison.csv",
    "MLP": "mlp_comparison.csv",
}
DATASET_ORDER = ("Pełny zbiór", "Zoptymalizowany zbiór")
QUALITY_COLUMNS = ("Dokładność", "Precyzja", "Czułość", "Miara F1", "AUC-ROC")
ERROR_RATE_COLUMN = "Współczynnik błędu klasyfikacji"
FPR_COLUMN = "Odsetek fałszywych alarmów (FPR)"
FINAL_TABLE_COLUMNS = (
    "Algorytm klasyfikacji", "Zbiór danych", *QUALITY_COLUMNS,
    ERROR_RATE_COLUMN, FPR_COLUMN, "Czas uczenia (s)", "Liczba cech",
    "Średnie CPU [%]", "Średni RAM [GB]",
)


def load_final_model_results(input_dir: str | Path) -> pd.DataFrame:
    """Load and combine the final comparison tables for all algorithms."""
    directory = Path(input_dir)
    missing = missing_final_model_results(directory)
    if missing:
        raise FileNotFoundError(_missing_files_message(missing))
    frames = [_read_model_table(directory, model, filename)
              for model, filename in MODEL_FILES.items()]
    results = pd.concat(frames, ignore_index=True)
    _validate_result_table(results)
    return _select_result_columns(_order_results(results))


def missing_final_model_results(input_dir: str | Path) -> list[Path]:
    """Return model result tables required for a complete comparison."""
    return _missing_model_files(Path(input_dir))


def _missing_model_files(directory: Path) -> list[Path]:
    """Return required result files that have not been generated yet."""
    return [directory / filename for filename in MODEL_FILES.values()
            if not (directory / filename).is_file()]


def _missing_files_message(paths: list[Path]) -> str:
    """Explain which model notebooks must finish before comparison."""
    names = ", ".join(path.name for path in paths)
    return f"Brakuje końcowych wyników modeli: {names}"


def _read_model_table(directory: Path, model: str, filename: str) -> pd.DataFrame:
    """Read one algorithm table and attach its stable display name."""
    frame = pd.read_csv(directory / filename)
    frame.insert(0, "Algorytm klasyfikacji", model)
    frame[ERROR_RATE_COLUMN] = 1.0 - frame["Dokładność"]
    if "FPR (odsetek fałszywie dodatnich)" in frame:
        frame[FPR_COLUMN] = frame["FPR (odsetek fałszywie dodatnich)"]
    return frame


def _validate_result_table(results: pd.DataFrame) -> None:
    """Reject incomplete or duplicated final model result tables."""
    required = {"Algorytm klasyfikacji", "Zbiór danych", *QUALITY_COLUMNS,
                "Czas uczenia (s)", "Średnie CPU [%]", "Średni RAM [GB]"}
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError(f"Brak wymaganych kolumn: {', '.join(missing)}")
    duplicates = results.duplicated(["Algorytm klasyfikacji", "Zbiór danych"])
    if duplicates.any():
        raise ValueError("Powielone końcowe wyniki algorytmu i wariantu danych")


def _order_results(results: pd.DataFrame) -> pd.DataFrame:
    """Sort algorithms and dataset variants in one deterministic order."""
    model_order = {name: index for index, name in enumerate(MODEL_FILES)}
    dataset_order = {name: index for index, name in enumerate(DATASET_ORDER)}
    ordered = results.assign(
        _model=results["Algorytm klasyfikacji"].map(model_order),
        _dataset=results["Zbiór danych"].map(dataset_order),
    )
    return ordered.sort_values(["_model", "_dataset"]).drop(
        columns=["_model", "_dataset"]
    ).reset_index(drop=True)


def _select_result_columns(results: pd.DataFrame) -> pd.DataFrame:
    """Keep only comparable columns with a defined cross-model meaning."""
    columns = [
        column for column in FINAL_TABLE_COLUMNS
        if column in results.columns
    ]
    return results.loc[:, columns]


def generate_final_model_comparisons(
    input_dir: str | Path, output_dir: str | Path, *,
    display_in_notebook: bool = False,
) -> tuple[pd.DataFrame, list[Path]]:
    """Save one final table and bilingual inter-model comparison figures."""
    output = Path(output_dir)
    results = load_final_model_results(input_dir)
    existing_figures = set(plt.get_fignums())
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "final_model_results.csv", index=False)
    figures = _comparison_figures(results)
    paths = [_save_named_figure(figure, output, name, display_in_notebook)
             for name, figure in figures]
    _close_generated_figures(existing_figures)
    return results, paths


def _close_generated_figures(existing_figures: set[int]) -> None:
    """Close copied figure managers created by the inline backend."""
    generated = set(plt.get_fignums()).difference(existing_figures)
    for figure_number in generated:
        plt.close(figure_number)


def _comparison_figures(results: pd.DataFrame):
    """Return named final figures without saving or displaying them."""
    figures = []
    for dataset in DATASET_ORDER:
        figures.append((_quality_filename(dataset),
                        plot_quality_comparison(results, dataset)))
        figures.append((_error_filename(dataset),
                        plot_error_comparison(results, dataset)))
    figures.extend(_measure_figures(results))
    return figures


def _measure_figures(results: pd.DataFrame):
    """Return comparisons of training cost and mean resource usage."""
    return [
        ("training_time.png", plot_training_time_comparison(results)),
        ("average_cpu.png", plot_variant_measure(
            results, "Średnie CPU [%]", "Średnie wykorzystanie procesora [%]",
            natural_upper=100.0,
        )),
        ("average_ram.png", plot_variant_measure(
            results, "Średni RAM [GB]", "Średnie wykorzystanie pamięci RAM [GB]",
            natural_upper=None,
        )),
    ]


def plot_training_time_comparison(results: pd.DataFrame):
    """Compare relative training-time changes with absolute-value labels."""
    pivot = _training_time_pivot(results)
    figure, axis = plt.subplots(figsize=(11, 6))
    _plot_training_time_panel(axis, pivot)
    figure.tight_layout(pad=1.2)
    return figure


def _training_time_pivot(results: pd.DataFrame) -> pd.DataFrame:
    """Return validated training times in stable display order."""
    pivot = results.pivot(
        index="Algorytm klasyfikacji", columns="Zbiór danych", values="Czas uczenia (s)"
    ).reindex(index=MODEL_FILES, columns=DATASET_ORDER)
    values = pivot.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("Czasy uczenia muszą być skończone i dodatnie")
    return pivot


def _plot_training_time_panel(axis, pivot: pd.DataFrame) -> None:
    """Draw percentage reduction and retain exact training times."""
    reduction = _training_time_reduction(pivot)
    colors = [DATASET_COLORS["full"] if value >= 0 else SECONDARY_COLOR
              for value in reduction]
    bars = axis.barh(pivot.index, reduction, color=colors)
    _label_training_time_bars(
        axis, bars, _training_time_labels(pivot, reduction), reduction,
    )
    axis.axvline(0.0, color="#25344F", linewidth=0.9)
    axis.set_xlabel("Skrócenie czasu uczenia po redukcji cech [%]")
    axis.set_ylabel("Algorytm klasyfikacji")
    axis.set_xlim(-5.0, 100.0)
    axis.set_xticks(np.arange(0.0, 101.0, 20.0))
    axis.legend(handles=_training_time_legend(), frameon=False)
    axis.grid(axis="x", alpha=0.3)


def _training_time_reduction(pivot: pd.DataFrame) -> np.ndarray:
    """Return percentage reduction from full to optimized features."""
    full = pivot[DATASET_ORDER[0]].to_numpy(dtype=float)
    optimized = pivot[DATASET_ORDER[1]].to_numpy(dtype=float)
    return (full - optimized) / full * 100.0


def _training_time_labels(pivot, reduction) -> list[str]:
    """Combine relative change with full and optimized times."""
    full = pivot[DATASET_ORDER[0]].to_numpy(dtype=float)
    optimized = pivot[DATASET_ORDER[1]].to_numpy(dtype=float)
    return [
        f"{polish_number(change, 2)}% ({polish_number(before, 2)} → "
        f"{polish_number(after, 2)} s)"
        for change, before, after in zip(reduction, full, optimized, strict=True)
    ]


def _label_training_time_bars(axis, bars, labels, reduction) -> None:
    """Place change labels to the right of the zero line."""
    for bar, label, change in zip(bars, labels, reduction, strict=True):
        position = float(change) + 0.8 if change >= 0 else 0.8
        axis.text(position, bar.get_y() + bar.get_height() / 2, label,
                  ha="left", va="center", fontsize=9)


def _training_time_legend() -> list[Patch]:
    """Explain the direction encoded by bar color."""
    return [
        Patch(facecolor=DATASET_COLORS["full"], label="Skrócenie czasu"),
        Patch(facecolor=SECONDARY_COLOR, label="Wydłużenie czasu"),
    ]


def _save_named_figure(
    figure, output: Path, filename: str, display_in_notebook: bool,
) -> Path:
    """Save one bilingual final figure and release its Matplotlib memory."""
    path = save_comparison_figure(
        figure, output / "plots" / filename, dpi=300,
        display_in_notebook=display_in_notebook,
    )
    plt.close(figure)
    return path


def _quality_filename(dataset: str) -> str:
    """Return a stable filename for one dataset quality comparison."""
    return "quality_full.png" if dataset == DATASET_ORDER[0] else "quality_optimized.png"


def _error_filename(dataset: str) -> str:
    """Return a stable filename for one dataset error comparison."""
    return "errors_full.png" if dataset == DATASET_ORDER[0] else "errors_optimized.png"


def plot_quality_comparison(results: pd.DataFrame, dataset: str):
    """Compare final test-set classification metrics between algorithms."""
    matrix = _dataset_rows(results, dataset).set_index(
        "Algorytm klasyfikacji"
    )[list(QUALITY_COLUMNS)]
    matrix = matrix.rename(columns={"AUC-ROC": "ROC-AUC"})
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        matrix, annot=True, fmt=".4f", cmap="Blues", vmin=0.0, vmax=1.0,
        linewidths=0.5, cbar_kws={
            "label": "Wartość miary jakości klasyfikacji",
        }, ax=axis,
    )
    axis.set_xlabel("Miara jakości klasyfikacji")
    axis.set_ylabel("Algorytm klasyfikacji")
    figure.tight_layout(pad=1.2)
    return figure


def _dataset_rows(results: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Select one complete dataset variant in canonical model order."""
    rows = results[results["Zbiór danych"] == dataset]
    if len(rows) != len(MODEL_FILES):
        raise ValueError(f"Niepełne wyniki dla wariantu: {dataset}")
    return rows


def plot_error_comparison(results: pd.DataFrame, dataset: str):
    """Compare final classification error and available false alarms."""
    rows = _dataset_rows(results, dataset)
    columns = [ERROR_RATE_COLUMN]
    if FPR_COLUMN in rows and rows[FPR_COLUMN].notna().all():
        columns.append(FPR_COLUMN)
    values = rows[columns].to_numpy(dtype=float) * 100.0
    figure, axis = plt.subplots(figsize=(11, 6))
    _draw_error_bars(axis, rows["Algorytm klasyfikacji"], columns, values)
    _configure_error_axis(axis, values)
    figure.tight_layout(pad=1.2)
    return figure


def _draw_error_bars(axis, models, columns, values) -> None:
    """Draw grouped error percentages with exact labels above bars."""
    positions = np.arange(len(models), dtype=float)
    width = 0.7 / len(columns)
    colors = (QUATERNARY_COLOR, SECONDARY_COLOR)
    for index, column in enumerate(columns):
        offset = (index - (len(columns) - 1) / 2.0) * width
        bars = axis.bar(positions + offset, values[:, index], width,
                        label=column, color=colors[index])
        axis.bar_label(bars, labels=_percentage_labels(values[:, index]),
                       padding=3, fontsize=9)
    axis.set_xticks(positions, models)


def _percentage_labels(values) -> list[str]:
    """Format compact Polish percentage labels for comparison bars."""
    return [f"{polish_number(float(value), 2)}%" for value in values]


def _configure_error_axis(axis, values) -> None:
    """Apply a zero-based percentage scale and complete scientific labels."""
    axis.set_ylim(0.0, _percentage_axis_upper(values))
    axis.set_xlabel("Algorytm klasyfikacji")
    axis.set_ylabel("Wartość miary błędu [%]")
    axis.legend(title="Miara błędu", frameon=False)
    axis.grid(axis="y", alpha=0.3)
    configure_axis_limit_ticks(axis, "y", floor=0.0, max_decimals=2)


def _percentage_axis_upper(values) -> float:
    """Return a rounded zero-based percentage ceiling with visible reserve."""
    maximum = max(float(np.nanmax(values)), 0.0)
    step = 5.0 if maximum < 40.0 else 10.0
    reserved = maximum * 1.15
    return min(100.0, max(step, float(np.ceil(reserved / step) * step)))


def plot_variant_measure(
    results: pd.DataFrame, column: str, x_label: str, *,
    natural_upper: float | None,
):
    """Compare one final training measurement across models and variants."""
    pivot = results.pivot(
        index="Algorytm klasyfikacji", columns="Zbiór danych", values=column
    ).reindex(MODEL_FILES)
    figure, axis = plt.subplots(figsize=(11, 6))
    _draw_variant_bars(axis, pivot)
    _configure_measure_axis(axis, pivot, x_label, natural_upper)
    figure.tight_layout(pad=1.2)
    return figure


def _draw_variant_bars(axis, pivot: pd.DataFrame) -> None:
    """Draw readable grouped horizontal bars for two dataset variants."""
    positions = np.arange(len(pivot), dtype=float)
    height = 0.34
    for index, dataset in enumerate(DATASET_ORDER):
        values = pivot[dataset].to_numpy(dtype=float)
        offset = (index - 0.5) * height
        color_key = "full" if index == 0 else "optimized"
        bars = axis.barh(
            positions + offset, values, height,
            color=DATASET_COLORS[color_key], label=dataset,
        )
        axis.bar_label(
            bars, labels=_measure_labels(values), padding=4, fontsize=9,
        )
    axis.set_yticks(positions, pivot.index)


def _measure_labels(values) -> list[str]:
    """Format exact values placed beside resource comparison bars."""
    return [polish_number(float(value), 2) for value in values]


def _configure_measure_axis(axis, pivot, x_label, natural_upper) -> None:
    """Configure a zero-based paired measurement plot."""
    values = pivot.to_numpy(dtype=float)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Algorytm klasyfikacji")
    axis.legend(title="Wariant danych", frameon=False)
    axis.grid(axis="x", alpha=0.3)
    _configure_linear_measure_axis(axis, values, natural_upper)


def _configure_linear_measure_axis(axis, values, natural_upper) -> None:
    """Use a nonnegative linear scale with reserve for bar labels."""
    measured_upper = max(float(np.nanmax(values)) * 1.18, 1.0)
    upper = measured_upper if natural_upper is None else natural_upper
    axis.set_xlim(0.0, upper)
    configure_axis_limit_ticks(axis, "x", floor=0.0, max_decimals=2)
