"""Tests for reading notebook accuracy artifacts."""

import csv
import pickle

from services.model_accuracy_service import ModelAccuracyService


def _write_comparison(root, filename, accuracy):
    directory = root / "original" / "model_comparisons"
    directory.mkdir(parents=True)
    path = directory / filename
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Zbiór danych", "Dokładność"])
        writer.writeheader()
        writer.writerow({"Zbiór danych": "Pełny zbiór", "Dokładność": accuracy})


def test_accuracy_service_reads_current_values_and_preserves_missing_models(tmp_path):
    _write_comparison(tmp_path, "random_forest_comparison.csv", "0.875")

    values = ModelAccuracyService(tmp_path).get_accuracies()

    assert values["current"]["full"]["random_forest"] == 0.875
    assert values["current"]["optimized"] == {}
    assert values["url_corpus"]["full"] == {}


def test_accuracy_service_reads_missing_value_from_model_artifact(tmp_path):
    artifact = tmp_path / "models" / "mlp"
    artifact.mkdir(parents=True)
    with (artifact / "mlp_model_full_url_corpus.pkl").open("wb") as handle:
        pickle.dump({"metrics": {"accuracy": 0.8597}}, handle)

    values = ModelAccuracyService(tmp_path / "results").get_accuracies()

    assert values["url_corpus"]["full"]["mlp"] == 0.8597
