"""Run labelled URL-corpus predictions and save a JSON smoke-test report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.ml_prediction_service import MLPredictionService
from services.url_feature_service import URLFeatureService
from config.feature_modes import CANONICAL_MODES
from config.project_paths import RESULTS_ROOT
MODEL_NAMES = (
    "random_forest",
    "xgboost",
    "mlp",
    "svm",
    "logistic_regression",
)
SAMPLES = (
    {"name": "safe_gmail", "url": "https://gmail.com/", "expected": 0},
    {"name": "safe_github", "url": "https://github.com/", "expected": 0},
    {
        "name": "phishing_brand_lookalike",
        "url": "https://bankofamerica-verify.com/secure/login?id=12345&token=abc",
        "expected": 1,
    },
    {
        "name": "phishing_ip_login",
        "url": "http://192.0.2.10/secure/login.php?session=reset",
        "expected": 1,
    },
)


def build_prediction_service() -> MLPredictionService:
    """Build a lazy prediction facade for smoke-test model loading."""
    service = MLPredictionService.__new__(MLPredictionService)
    service.models = {}
    service.scalers = {}
    service.models_dir = BACKEND_ROOT / "models"
    return service


def run(output_path: Path) -> dict:
    """Run labelled offline predictions for every configured mode and model."""
    feature_service = URLFeatureService(apply_network_defaults=False)
    prediction_service = build_prediction_service()
    report = {"dataset": "url_corpus", "modes": {
        mode: _mode_results(mode, feature_service, prediction_service)
        for mode in CANONICAL_MODES
    }}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _mode_results(mode, feature_service, prediction_service):
    """Evaluate all labelled samples for one feature mode."""
    return [_sample_result(sample, mode, feature_service, prediction_service) for sample in SAMPLES]


def _sample_result(sample, mode, feature_service, prediction_service):
    """Evaluate one sample across all model adapters."""
    features = feature_service.extract_offline(sample["url"], mode)
    predictions = {name: _model_result(features, name, mode, prediction_service) for name in MODEL_NAMES}
    votes = sum(item["prediction"] for item in predictions.values())
    ensemble = int(votes >= 3)
    return {**sample, "models": predictions, "phishing_votes": votes,
            "ensemble_prediction": ensemble, "passed": ensemble == sample["expected"]}


def _model_result(features, model_name, mode, prediction_service):
    """Return compact prediction details for one model."""
    result = prediction_service.predict_url(features, model_name, mode, dataset="url_corpus", include_similar=False)
    return {"prediction": result["prediction"],
            "phishing_probability": result["probabilities"]["phishing"],
            "features_used": result["features_used"]}


def main() -> None:
    """Run the smoke test and fail when ensemble labels do not match."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_ROOT / "url_corpus" / "model_smoke_test.json",
    )
    args = parser.parse_args()
    report = run(args.output)
    total = sum(len(results) for results in report["modes"].values())
    passed = sum(
        item["passed"]
        for results in report["modes"].values()
        for item in results
    )
    print(f"URL corpus model smoke test: {passed}/{total} passed")
    print(f"Report: {args.output}")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
