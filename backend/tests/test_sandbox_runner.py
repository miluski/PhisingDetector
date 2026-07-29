from sandbox_runner import SandboxRunner


class _FakeFeatureService:
    def extract(self, url, mode):
        return {'feature': 1}


class _FakePredictionService:
    def __init__(self):
        self.predict_calls = 0
        self.ensemble_calls = 0

    def predict_multiple_models(self, features, models, mode, dataset):
        self.predict_calls += 1
        return ([{
            'model': models[0],
            'prediction': 0,
            'confidence': 0.9,
        }], [])

    def get_ensemble_from_predictions(self, predictions, dataset, similar_samples):
        self.ensemble_calls += 1
        return {
            'ensemble_prediction': 0,
            'models_voting': {'phishing': 0, 'safe': 1},
            'confidence': 1.0,
        }


def test_sandbox_runs_model_prediction_once_per_url():
    runner = SandboxRunner()
    runner.feature_service = _FakeFeatureService()
    runner.prediction_service = _FakePredictionService()

    result = runner._analyze_url(
        'https://example.com', 1, 1, ['random_forest'], 'optimized', 'current'
    )

    assert result['success'] is True
    assert runner.prediction_service.predict_calls == 1
    assert runner.prediction_service.ensemble_calls == 1
