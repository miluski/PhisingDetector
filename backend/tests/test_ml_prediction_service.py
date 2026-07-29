import numpy as np
import pandas as pd

from services.ml_prediction_service import MLPredictionService


class FixedProbabilityModel:
    def predict_proba(self, values):
        return np.array([[0.6, 0.4]])


def test_prediction_uses_model_payload_decision_threshold():
    service = MLPredictionService.__new__(MLPredictionService)
    features = pd.DataFrame([[1.0]], columns=["feature"])

    safe, probabilities = service._get_prediction_and_proba(
        FixedProbabilityModel(), features, "random_forest", decision_threshold=0.5
    )
    phishing, _ = service._get_prediction_and_proba(
        FixedProbabilityModel(), features, "random_forest", decision_threshold=0.3
    )

    assert safe == 0
    assert phishing == 1
    assert probabilities.tolist() == [0.6, 0.4]
