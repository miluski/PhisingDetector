"""Model-specific prediction adapters."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd


class ModelAdapter(Protocol):
    """Protocol implemented by all inference adapters."""

    def probabilities(self, model: Any, vector: pd.DataFrame) -> np.ndarray:
        """Return safe and phishing probabilities."""

    def prediction(self, probabilities: np.ndarray, threshold: float) -> int:
        """Convert probabilities into a binary label."""


class SklearnModelAdapter:
    """Adapter for scikit-learn estimators exposing ``predict_proba``."""

    def probabilities(self, model: Any, vector: pd.DataFrame) -> np.ndarray:
        """Call the estimator and normalize the result to a numpy array."""
        values = vector if hasattr(model, "feature_names_in_") else vector.values
        return np.asarray(model.predict_proba(values)[0], dtype=float)

    def prediction(self, probabilities: np.ndarray, threshold: float) -> int:
        """Apply the configured phishing threshold."""
        return int(probabilities[1] >= threshold)


class MlpModelAdapter:
    """Adapter for Keras MLP models with one sigmoid output."""

    def probabilities(self, model: Any, vector: pd.DataFrame) -> np.ndarray:
        """Convert a Keras prediction into safe/phishing probabilities."""
        result = self._raw_prediction(model, vector)
        phishing = float(result[0]) if len(result) == 1 else float(result[1])
        return np.asarray([1.0 - phishing, phishing], dtype=float)

    @staticmethod
    def _raw_prediction(model: Any, vector: pd.DataFrame) -> np.ndarray:
        """Evaluate Keras directly to avoid progress-log scalar conversion."""
        if callable(model):
            output = model(vector.values, training=False)
            output = output.numpy() if hasattr(output, "numpy") else output
        else:
            output = model.predict(vector.values, verbose=0)
        return np.asarray(output, dtype=float).reshape(-1)

    def prediction(self, probabilities: np.ndarray, threshold: float) -> int:
        """Apply the configured phishing threshold."""
        return int(probabilities[1] >= threshold)


def adapter_for(model_name: str) -> ModelAdapter:
    """Return an adapter for a supported model name."""
    return MlpModelAdapter() if model_name == "mlp" else SklearnModelAdapter()
