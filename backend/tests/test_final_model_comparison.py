from pathlib import Path

import numpy as np
import pandas as pd

from utils.final_model_comparison import (
    DATASET_ORDER,
    MODEL_FILES,
    generate_final_model_comparisons,
    load_final_model_results,
    missing_final_model_results,
    plot_error_comparison,
    plot_quality_comparison,
    plot_variant_measure,
)


def _write_model_results(directory: Path, *, include_fpr: bool = True) -> None:
    """Write deterministic final result fixtures for every model."""
    directory.mkdir()
    for index, filename in enumerate(MODEL_FILES.values()):
        frame = _model_frame(index, include_fpr)
        frame.to_csv(directory / filename, index=False)


def _model_frame(index: int, include_fpr: bool) -> pd.DataFrame:
    """Return two realistic final-result rows for one model fixture."""
    rows = []
    for variant_index, dataset in enumerate(DATASET_ORDER):
        score = 0.90 + index * 0.01 - variant_index * 0.002
        row = _result_row(dataset, score, index, variant_index)
        if include_fpr:
            row["FPR (odsetek fałszywie dodatnich)"] = 0.04 - index * 0.004
        rows.append(row)
    return pd.DataFrame(rows)


def _result_row(dataset, score, model_index, variant_index) -> dict:
    """Build one complete model-comparison fixture row."""
    return {
        "Zbiór danych": dataset,
        "Dokładność": score,
        "Precyzja": score + 0.01,
        "Czułość": score - 0.01,
        "Miara F1": score,
        "AUC-ROC": min(score + 0.05, 0.999),
        "Czas uczenia (s)": 0.5 * 10 ** model_index + variant_index,
        "Średnie CPU [%]": 10.0 + model_index,
        "Średni RAM [GB]": 0.2 + model_index * 0.1,
    }


def test_load_final_results_combines_all_models_and_variants(tmp_path):
    input_dir = tmp_path / "model_comparisons"
    _write_model_results(input_dir)

    results = load_final_model_results(input_dir)

    assert len(results) == len(MODEL_FILES) * len(DATASET_ORDER)
    assert list(results["Algorytm klasyfikacji"].unique()) == list(MODEL_FILES)
    assert np.allclose(
        results["Współczynnik błędu klasyfikacji"],
        1.0 - results["Dokładność"],
    )


def test_final_table_omits_noncomparable_nan_cv_columns(tmp_path):
    input_dir = tmp_path / "model_comparisons"
    _write_model_results(input_dir)
    mlp_path = input_dir / MODEL_FILES["MLP"]
    mlp = pd.read_csv(mlp_path)
    mlp["Średnia CV AUC"] = np.nan
    mlp["Odchylenie CV AUC"] = np.nan
    mlp.to_csv(mlp_path, index=False)

    results = load_final_model_results(input_dir)

    assert "Średnia CV AUC" not in results
    assert "Odchylenie CV AUC" not in results
    assert not results.isna().any(axis=None)


def test_load_final_results_reports_missing_algorithm(tmp_path):
    import pytest

    input_dir = tmp_path / "model_comparisons"
    _write_model_results(input_dir)
    (input_dir / "mlp_comparison.csv").unlink()

    with pytest.raises(FileNotFoundError, match="mlp_comparison.csv"):
        load_final_model_results(input_dir)


def test_missing_final_results_supports_nonfatal_notebook_preflight(tmp_path):
    """Report incomplete comparison inputs before optional plot generation."""
    input_dir = tmp_path / "model_comparisons"
    _write_model_results(input_dir)
    missing_path = input_dir / "mlp_comparison.csv"
    missing_path.unlink()
    assert missing_final_model_results(input_dir) == [missing_path]


def test_final_comparison_plots_have_complete_scientific_labels(tmp_path):
    from matplotlib import pyplot as plt

    input_dir = tmp_path / "model_comparisons"
    _write_model_results(input_dir)
    results = load_final_model_results(input_dir)
    figures = [
        plot_quality_comparison(results, DATASET_ORDER[0]),
        plot_error_comparison(results, DATASET_ORDER[0]),
        plot_variant_measure(
            results, "Czas uczenia (s)", "Czas uczenia modelu [s]",
            natural_upper=None,
        ),
        plot_variant_measure(
            results, "Średnie CPU [%]", "Średnie wykorzystanie procesora [%]",
            natural_upper=100.0,
        ),
    ]

    assert figures[0].axes[0].get_xlabel() == "Miara jakości klasyfikacji"
    assert figures[0].axes[-1].get_ylabel() == "Wartość miary jakości klasyfikacji"
    assert figures[0].axes[0].collections[0].norm.vmin == 0.0
    assert figures[0].axes[0].collections[0].norm.vmax == 1.0
    assert figures[1].axes[0].get_legend() is not None
    assert figures[2].axes[0].get_legend() is not None
    assert figures[1].axes[0].get_ylim()[0] == 0.0
    assert figures[1].axes[0].get_ylim()[1] < 100.0
    assert figures[2].axes[0].get_xscale() == "linear"
    assert figures[2].axes[0].get_xlim()[0] == 0.0
    assert len(figures[2].axes[0].patches) == 10
    assert len(figures[3].axes[0].patches) == 10
    assert figures[3].axes[0].get_xlim() == (0.0, 100.0)
    assert not figures[3].axes[0].collections
    for figure in figures:
        plt.close(figure)


def test_generation_saves_one_table_and_bilingual_figures(tmp_path):
    from matplotlib import pyplot as plt

    input_dir = tmp_path / "model_comparisons"
    output_dir = tmp_path / "final_model_comparison"
    _write_model_results(input_dir)
    existing = plt.figure()

    results, paths = generate_final_model_comparisons(input_dir, output_dir)

    assert len(results) == 10
    assert len(paths) == 7
    assert (output_dir / "final_model_results.csv").is_file()
    assert all(path.is_file() and path.parent.name == "pl" for path in paths)
    assert all((path.parent.parent / "eng" / path.name).is_file()
               for path in paths)
    assert plt.get_fignums() == [existing.number]
    plt.close(existing)
