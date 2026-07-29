"""Plotting helpers for :mod:`resource_monitor`.

The monitor owns sampling; this module owns rendering and file output so the
sampling lifecycle cannot affect axis construction.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .resource_monitor import MAX_TWO_DECIMALS, _data_limits


MEAN_COLOR = "#228833"


def plot_monitor(monitor, title: str, save_path: str | None = None):
    """Render a two-panel resource figure and optionally save separate panels."""
    data = prepare_monitor_data(monitor)
    if data is None:
        print("Brak danych do wykresu")
        return None
    figure, axes = plt.subplots(2, 1, figsize=(12, 10))
    draw_axis(axes[0], data[0], data[1], "Użycie CPU (%)", "#2E86AB")
    draw_axis(axes[1], data[0], data[2], "Użycie RAM (GB)", "#A23B72")
    figure.tight_layout()
    if save_path:
        save_panels(data, title, save_path)
    plt.show()
    return figure


def prepare_monitor_data(monitor):
    """Return elapsed time, CPU and RAM arrays for the training window."""
    if not monitor.timestamps:
        return None
    times = np.asarray(monitor.timestamps, dtype=float)
    cpu = np.asarray(monitor.cpu_percent, dtype=float)
    ram = np.asarray(monitor.ram_gb, dtype=float)
    start, end = monitor.training_start_time, monitor.training_end_time
    if start is not None and end is not None:
        mask = (times >= start) & (times <= end)
        times, cpu, ram = times[mask] - start, cpu[mask], ram[mask]
    return times, cpu, ram


def draw_axis(axis, times, values, ylabel, color):
    """Draw one resource series with zero-based time and bounded y-axis."""
    limits = _data_limits(values)
    mean = float(values.mean())
    axis.plot(times, values, color=color, linewidth=1.5, label="Pomiar")
    axis.axhline(
        mean, color=MEAN_COLOR, linestyle="--", linewidth=2.4,
        label="Średnia podczas uczenia", clip_on=False, zorder=8,
    )
    _annotate_mean(axis, mean, ylabel)
    axis.set_xlabel("Czas (sekundy)")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3, linestyle="--")
    axis.set_xlim(0.0, max(float(times[-1]), 0.05))
    axis.set_ylim(0.0, limits[1])
    axis.yaxis.set_major_formatter(MAX_TWO_DECIMALS)
    axis.legend()


def _annotate_mean(axis, mean, ylabel):
    """Place the formatted resource mean directly above its line."""
    unit = "%" if "CPU" in ylabel else " GB"
    axis.annotate(
        f"Średnia: {_mean_label(np.asarray([mean]))}{unit}",
        xy=(0.5, mean), xycoords=("axes fraction", "data"),
        xytext=(0, 6), textcoords="offset points",
        ha="center", va="bottom", color=MEAN_COLOR, fontweight="bold",
        zorder=9, bbox={
            "facecolor": "white", "edgecolor": MEAN_COLOR,
            "linewidth": 0.8, "alpha": 0.95, "pad": 2.0,
        },
    )


def _mean_label(values):
    """Format a resource mean using Polish decimal notation."""
    return f"{float(values.mean()):.2f}".replace(".", ",").rstrip("0").rstrip(",")


def save_panels(data, title, save_path):
    """Save CPU and RAM panels as independent PNG files."""
    times, cpu, ram = data
    target = Path(save_path)
    save_panel(times, cpu, target.with_name(target.stem + "_cpu.png"), "Użycie CPU (%)", "#2E86AB")
    save_panel(times, ram, target.with_name(target.stem + "_ram.png"), "Użycie RAM (GB)", "#A23B72")


def save_panel(times, values, path, ylabel, color):
    """Render and save one resource panel."""
    figure, axis = plt.subplots(figsize=(12, 6))
    draw_axis(axis, times, values, ylabel, color)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"✓ Zapisano wykres: {path}")
