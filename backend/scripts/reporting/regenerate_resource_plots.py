"""Regenerate CPU and RAM figures from cached phase traces without training."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent
os.environ.setdefault("MPLBACKEND", "Agg")
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from utils.notebook_plot_style import (
    plot_resource_usage_separate,
    save_dataset_figure,
)


MODELS = (
    "logistic_regression", "random_forest", "xgboost", "svm", "mlp",
)
DATASETS = {
    "full": "Pełny zbiór",
    "optimized": "Zoptymalizowany zbiór",
}


def parse_args() -> argparse.Namespace:
    """Parse result scope and optional project result directory."""
    parser = argparse.ArgumentParser(
        description="Odtwarza wykresy CPU/RAM z zapisanych trace'ów faz."
    )
    parser.add_argument(
        "--scope", choices=("original", "url_corpus", "all"), default="all",
    )
    parser.add_argument(
        "--results-root", type=Path, default=PROJECT_ROOT / "results",
    )
    return parser.parse_args()


def selected_scopes(scope: str) -> tuple[str, ...]:
    """Return concrete scopes selected by the command-line argument."""
    return ("original", "url_corpus") if scope == "all" else (scope,)


def regenerate_scope(results_root: Path, scope: str) -> list[Path]:
    """Regenerate every available CPU/RAM pair for one result scope."""
    root = results_root / scope
    paths = []
    for model in MODELS:
        for dataset_key, dataset_name in DATASETS.items():
            trace = root / "resource_usage" / f"{model}_{dataset_key}.csv"
            if trace.is_file():
                paths.extend(_regenerate_trace(root, model, dataset_name, trace))
    return paths


def _regenerate_trace(
    root: Path, model: str, dataset_name: str, trace_path: Path,
) -> list[Path]:
    """Render and save one cached trace as separate bilingual figures."""
    frame = pd.read_csv(trace_path)
    monitor = _monitor_from_trace(frame)
    stats = _training_stats(frame)
    figures = plot_resource_usage_separate(
        monitor, stats, dataset_name, include_post_training=True
    )
    paths = _save_resource_figures(root, model, dataset_name, figures)
    _close_figures(figures)
    return paths


def _monitor_from_trace(frame: pd.DataFrame) -> SimpleNamespace:
    """Reconstruct plotting metadata from synchronized cached phase samples."""
    _validate_trace(frame)
    training = frame[frame["faza"] == "training"]["czas_s"]
    return SimpleNamespace(
        timestamps=frame["czas_s"].tolist(),
        phases=frame["faza"].tolist(),
        cpu_percent=frame["cpu_procent"].tolist(),
        ram_gb=frame["ram_gb"].tolist(),
        training_start_time=float(training.iloc[0]),
        training_end_time=float(training.iloc[-1]),
        num_cores=os.cpu_count() or 1,
    )


def _validate_trace(frame: pd.DataFrame) -> None:
    """Reject cached traces without the columns and phases needed for plotting."""
    required = {"czas_s", "faza", "cpu_procent", "ram_gb"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Brak kolumn trace'u: {', '.join(missing)}")
    phases = set(frame["faza"])
    if not {"baseline", "training", "post_training"}.issubset(phases):
        raise ValueError("Trace musi zawierać fazy baseline, training i post_training")


def _training_stats(frame: pd.DataFrame) -> dict:
    """Calculate the raw training-window means required by the plotter."""
    training = frame[frame["faza"] == "training"]
    return {
        "cpu": {"mean": float(training["cpu_procent"].mean())},
        "ram": {"mean": float(training["ram_gb"].mean())},
    }


def _save_resource_figures(
    root: Path, model: str, dataset_name: str, figures,
) -> list[Path]:
    """Save cached CPU and RAM plots under their canonical model paths."""
    output = root / "plots" / model
    names = (
        f"{model}_resource_usage_cpu.png",
        f"{model}_resource_usage_ram.png",
    )
    return [
        save_dataset_figure(figure, output, dataset_name, name)
        for figure, name in zip(figures, names)
    ]


def _close_figures(figures) -> None:
    """Release Matplotlib figures after noninteractive cached rendering."""
    from matplotlib import pyplot as plt

    for figure in figures:
        plt.close(figure)


def main() -> int:
    """Regenerate cached resource figures for every selected result scope."""
    args = parse_args()
    for scope in selected_scopes(args.scope):
        paths = regenerate_scope(args.results_root, scope)
        print(f"{scope}: zapisano {len(paths)} wykresów CPU/RAM")
        for path in paths:
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
