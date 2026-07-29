"""Unit tests for model prediction adapters."""

import numpy as np
import pandas as pd

from services.model_adapters import MlpModelAdapter, SklearnModelAdapter


class ProbabilityModel:
    """Minimal classifier double exposing ``predict_proba``."""

    def predict_proba(self, values):
        """Return a fixed two-class probability row."""
        return np.array([[0.2, 0.8]])


class MlpModel:
    """Minimal neural model double exposing ``predict``."""

    def predict(self, values, verbose=0):
        """Return a one-column phishing probability."""
        return np.array([[0.8]])


def test_sklearn_adapter_reads_positive_probability():
    """Sklearn adapter should expose the positive-class probability."""
    adapter = SklearnModelAdapter()
    probabilities = adapter.probabilities(ProbabilityModel(), pd.DataFrame([[0.0, 0.0]]))
    assert np.allclose(probabilities, [0.2, 0.8])
    assert adapter.prediction(probabilities, 0.5) == 1


def test_mlp_adapter_reads_one_column_probability():
    """MLP adapter should support the one-output Keras contract."""
    adapter = MlpModelAdapter()
    probabilities = adapter.probabilities(MlpModel(), pd.DataFrame([[0.0, 0.0]]))
    assert np.allclose(probabilities, [0.2, 0.8])
    assert adapter.prediction(probabilities, 0.5) == 1


def test_adapters_include_probability_equal_to_the_frozen_threshold():
    """Inference must use the same inclusive boundary as calibration."""
    probabilities = np.asarray([0.5, 0.5])
    assert SklearnModelAdapter().prediction(probabilities, 0.5) == 1
    assert MlpModelAdapter().prediction(probabilities, 0.5) == 1
