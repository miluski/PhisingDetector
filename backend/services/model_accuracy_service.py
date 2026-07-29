"""Read current test-set accuracy values from notebook result files."""

from __future__ import annotations

import csv
import logging
import pickle
from pathlib import Path
from typing import Any

from config.project_paths import BACKEND_ROOT, RESULTS_ROOT


LOGGER = logging.getLogger(__name__)


class ModelAccuracyService:
    """Expose the latest saved test accuracy for each dataset and feature mode."""

    MODEL_FILES = {
        "random_forest": "random_forest_comparison.csv",
        "xgboost": "xgboost_comparison.csv",
        "mlp": "mlp_comparison.csv",
        "svm": "svm_comparison.csv",
        "logistic_regression": "logistic_regression_comparison.csv",
    }
    MODEL_ARTIFACT_PREFIXES = {
        "random_forest": "random_forest",
        "xgboost": "xgboost_model",
        "mlp": "mlp_model",
        "svm": "svm_model",
        "logistic_regression": "logistic_regression",
    }
    MODE_LABELS = {"full": "Pełny zbiór", "optimized": "Zoptymalizowany zbiór"}

    def __init__(self, results_root: Path | None = None) -> None:
        """Create a reader using the repository results directory by default."""
        self.results_root = results_root or RESULTS_ROOT
        parent = BACKEND_ROOT if results_root is None else results_root.parent
        self.model_root = parent / "models"

    def get_accuracies(self) -> dict[str, dict[str, dict[str, float]]]:
        """Return available accuracy values without inventing missing results."""
        return {
            "current": self._read_dataset("original"),
            "url_corpus": self._read_dataset("url_corpus"),
        }

    def _read_dataset(self, dataset: str) -> dict[str, dict[str, float]]:
        """Read every model comparison file available for one dataset."""
        return {
            mode: self._read_mode(dataset, mode)
            for mode in self.MODE_LABELS
        }

    def _read_mode(self, dataset: str, mode: str) -> dict[str, float]:
        """Collect model accuracies for one feature mode."""
        values: dict[str, float] = {}
        for model_id, filename in self.MODEL_FILES.items():
            value = self._read_model_value(dataset, model_id, filename, mode)
            if value is not None:
                values[model_id] = value
        return values

    def _read_model_value(
        self, dataset: str, model_id: str, filename: str, mode: str
    ) -> float | None:
        """Read one accuracy cell, returning ``None`` for missing artifacts."""
        path = self.results_root / dataset / "model_comparisons" / filename
        csv_value = self._read_csv_value(path, mode)
        return csv_value if csv_value is not None else self._read_artifact_value(
            dataset, model_id, mode
        )

    def _read_csv_value(self, path: Path, mode: str) -> float | None:
        """Read a CSV value while allowing a model artifact fallback."""
        if not path.exists():
            return None
        try:
            return self._find_accuracy(path, self.MODE_LABELS[mode])
        except (OSError, KeyError, ValueError, csv.Error) as exc:
            LOGGER.warning("Nie można odczytać dokładności z %s: %s", path, exc)
            return None

    def _read_artifact_value(self, dataset: str, model_id: str, mode: str) -> float | None:
        """Read ``metrics.accuracy`` from the deployed model artifact."""
        path = self._artifact_path(dataset, model_id, mode)
        if not path.exists():
            return None
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            if not isinstance(payload, dict):
                return None
            metrics = payload.get("metrics")
            if not isinstance(metrics, dict) or metrics.get("accuracy") is None:
                return None
            return float(metrics["accuracy"])
        except (OSError, EOFError, KeyError, TypeError, ValueError, pickle.UnpicklingError) as exc:
            LOGGER.warning("Nie można odczytać metryki z %s: %s", path, exc)
            return None

    def _artifact_path(self, dataset: str, model_id: str, mode: str) -> Path:
        """Return the model artifact path matching a dataset and feature mode."""
        prefix = self.MODEL_ARTIFACT_PREFIXES[model_id]
        suffix = "" if dataset == "current" else "_url_corpus"
        return self.model_root / model_id / f"{prefix}_{mode}{suffix}.pkl"

    @staticmethod
    def _find_accuracy(path: Path, dataset_label: str) -> float | None:
        """Find the accuracy in the row matching a dataset label."""
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("Zbiór danych") == dataset_label:
                    raw_value = row.get("Dokładność")
                    return float(raw_value) if raw_value else None
        return None


def serialize_accuracy_response(service: ModelAccuracyService) -> dict[str, Any]:
    """Build the stable JSON payload used by the frontend accuracy panel."""
    return {"success": True, "accuracies": service.get_accuracies()}
