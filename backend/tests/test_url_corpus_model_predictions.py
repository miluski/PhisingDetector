"""Smoke and behavioural tests for trained URL-corpus models.

Tests use lexical/offline feature extraction deliberately. They opt into
deterministic network defaults explicitly, so pytest never contacts external
services. Production inference preserves missing network values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from services.ml_prediction_service import MLPredictionService
from services.url_feature_service import URLFeatureService


BACKEND_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = (
    "random_forest",
    "xgboost",
    "mlp",
    "svm",
    "logistic_regression",
)
MODES = ("optimized", "full")

LABELLED_URLS = (
    pytest.param("safe", "https://gmail.com/", 0, id="safe-gmail"),
    pytest.param("safe", "https://github.com/", 0, id="safe-github"),
    pytest.param(
        "phishing",
        "https://bankofamerica-verify.com/secure/login?id=12345&token=abc",
        1,
        id="phishing-brand-lookalike",
    ),
    pytest.param(
        "phishing",
        "http://192.0.2.10/secure/login.php?session=reset",
        1,
        id="phishing-ip-login",
    ),
)


@pytest.fixture(scope="module")
def feature_service() -> URLFeatureService:
    return URLFeatureService(apply_network_defaults=True)


def test_offline_feature_extraction_honours_missing_network_policy():
    url = "https://example.com/"
    without_defaults = URLFeatureService(apply_network_defaults=False).extract_offline(url, "full")
    with_defaults = URLFeatureService(apply_network_defaults=True).extract_offline(url, "full")

    assert "missing_whois" not in without_defaults
    assert "missing_whois" not in with_defaults


@pytest.fixture(scope="module")
def prediction_service() -> MLPredictionService:
    """Create prediction service without loading the large similar-sample CSVs."""
    service = MLPredictionService.__new__(MLPredictionService)
    service.models = {}
    service.scalers = {}
    service.models_dir = BACKEND_ROOT / "models"
    return service


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("sample_name,url,expected", LABELLED_URLS)
def test_url_corpus_models_predict_labelled_urls(
    feature_service: URLFeatureService,
    prediction_service: MLPredictionService,
    mode: str,
    sample_name: str,
    url: str,
    expected: int,
):
    """All five URL-corpus artifacts load and ensemble vote matches label."""
    features = feature_service.extract_offline(url, mode)
    predictions = []

    for model_name in MODEL_NAMES:
        result = prediction_service.predict_url(
            features,
            model_name,
            mode,
            dataset="url_corpus",
            include_similar=False,
        )

        assert result["dataset"] == "url_corpus"
        assert result["mode"] == mode
        assert result["prediction"] in (0, 1), (sample_name, model_name, result)
        assert result["features_used"] > 0
        probabilities = result["probabilities"]
        assert np.isfinite(probabilities["safe"])
        assert np.isfinite(probabilities["phishing"])
        assert probabilities["safe"] + probabilities["phishing"] == pytest.approx(1.0)
        predictions.append(result["prediction"])

    phishing_votes = sum(predictions)
    if expected == 1:
        assert phishing_votes >= 3, (sample_name, mode, predictions)
    else:
        assert phishing_votes <= 2, (sample_name, mode, predictions)


@pytest.mark.parametrize("mode", MODES)
def test_url_corpus_model_artifacts_exist(mode: str):
    """Every selectable URL-corpus model has both feature-space variants."""
    expected_files = {
        "random_forest": f"random_forest_{mode}_url_corpus.pkl",
        "xgboost": f"xgboost_model_{mode}_url_corpus.pkl",
        "mlp": f"mlp_model_{mode}_url_corpus.pkl",
        "svm": f"svm_model_{mode}_url_corpus.pkl",
        "logistic_regression": f"logistic_regression_{mode}_url_corpus.pkl",
    }
    missing = [
        str(BACKEND_ROOT / "models" / model / filename)
        for model, filename in expected_files.items()
        if not (BACKEND_ROOT / "models" / model / filename).exists()
    ]
    assert not missing, f"Missing URL-corpus model artifacts: {missing}"
