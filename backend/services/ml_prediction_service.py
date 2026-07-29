"""Facade for model loading, inference, ensembles and explanations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.feature_policy import FEATURE_POLICY_VERSION
from config.feature_modes import normalize_mode
from services.feature_vector_builder import FeatureVectorBuilder
from services.model_adapters import adapter_for
from services.model_explainer import ModelExplainer
from services.model_repository import ModelRepository
from services.similar_sample_finder import SimilarSampleFinder

logger = logging.getLogger("MLPredictionService")


class MLPredictionService:
    """Keep the public prediction API while delegating each responsibility."""

    MODELS_REQUIRING_SCALER = {"mlp", "svm", "logistic_regression"}
    MODELS_WITH_MODEL_SUFFIX = ModelRepository.MODEL_SUFFIXES
    VALID_DATASETS = ModelRepository.VALID_DATASETS

    def __init__(self, repository: ModelRepository | None = None, similar_finder: Any = None):
        """Initialize inference collaborators."""
        directory = Path(__file__).parent.parent / "models"
        self.repository = repository or ModelRepository(directory)
        self.models_dir = directory
        self.models = self.repository.models
        self.scalers = self.repository.scalers
        self.vector_builder = FeatureVectorBuilder()
        self.explainer = ModelExplainer()
        self.similar_sample_finder = similar_finder or SimilarSampleFinder()

    def _get_repository(self) -> ModelRepository:
        """Return a repository, including for legacy test fixtures using ``__new__``."""
        repository = getattr(self, "repository", None)
        if repository is None:
            repository = ModelRepository(getattr(self, "models_dir"))
            repository.models = getattr(self, "models", {})
            repository.scalers = getattr(self, "scalers", {})
            self.repository = repository
        return repository

    def _normalize_dataset(self, dataset: str | None) -> str:
        """Normalize a dataset name through the repository."""
        return self._get_repository().normalize_dataset(dataset)

    def load_model(self, model_name: str, mode: str = "optimized", dataset: str = "current") -> dict[str, Any]:
        """Load one model payload using the repository."""
        mode = normalize_mode(mode)
        payload = self._get_repository().load(model_name, mode, dataset)
        self._validate_policy(payload)
        return payload

    @staticmethod
    def _validate_policy(payload: dict[str, Any]) -> None:
        """Validate explicitly tagged payloads without rejecting legacy artifacts."""
        policy = payload.get("feature_policy")
        if policy and policy != FEATURE_POLICY_VERSION:
            raise ValueError(f"Model feature policy mismatch: {policy!r}")

    def _build_vector(self, features: dict[str, float], payload: dict[str, Any], model_name: str) -> pd.DataFrame:
        """Build and optionally scale a vector according to model metadata."""
        if not hasattr(self, "vector_builder"):
            self.vector_builder = FeatureVectorBuilder()
        names = payload["feature_names"]
        vector = self.vector_builder.build(features, names)
        scaler = payload.get("scaler") if model_name in self.MODELS_REQUIRING_SCALER else None
        if scaler is not None:
            vector = self.vector_builder.scale(vector, scaler, names)
        return vector

    def _predict_payload(
        self, payload: dict[str, Any], model_name: str, vector: pd.DataFrame,
    ) -> tuple[int, np.ndarray]:
        """Run an adapter and apply the payload threshold."""
        adapter = adapter_for(model_name)
        probabilities = adapter.probabilities(payload["model"], vector)
        threshold = float(payload.get("decision_threshold", 0.5))
        return adapter.prediction(probabilities, threshold), probabilities

    def predict_url(
        self, features: dict[str, float], model_name: str, mode: str = "optimized",
        dataset: str = "current", include_similar: bool = True,
    ) -> dict[str, Any]:
        """Predict one URL and optionally attach similar training samples."""
        mode = normalize_mode(mode)
        dataset = self._normalize_dataset(dataset)
        payload = self.load_model(model_name, mode, dataset)
        vector = self._build_vector(features, payload, model_name)
        prediction, probabilities = self._predict_payload(payload, model_name, vector)
        result = self._result(model_name, mode, dataset, prediction, probabilities, payload)
        result.update(self._explain(model_name, payload, vector, features))
        return self._with_similar(result, features, mode, dataset, include_similar)

    def _result(self, model_name: str, mode: str, dataset: str, prediction: int,
                probabilities: np.ndarray, payload: dict[str, Any]) -> dict[str, Any]:
        """Build the stable public prediction response."""
        return {
            "model": model_name, "mode": mode, "dataset": dataset,
            "prediction": int(prediction),
            "prediction_label": "Phishing" if prediction else "Safe",
            "probabilities": {"safe": float(probabilities[0]), "phishing": float(probabilities[1])},
            "confidence": float(max(probabilities)),
            "features_used": len(payload["feature_names"]),
        }

    def _explain(self, model_name: str, payload: dict[str, Any], vector: pd.DataFrame,
                 features: dict[str, float]) -> dict[str, Any]:
        """Return optional model-specific explanation data."""
        if model_name != "random_forest":
            return {}
        if not hasattr(self, "explainer"):
            self.explainer = ModelExplainer()
        return self.explainer.random_forest(payload["model"], vector, features)

    def _with_similar(self, result: dict[str, Any], features: dict[str, float], mode: str,
                      dataset: str, include_similar: bool) -> dict[str, Any]:
        """Attach similar samples when requested."""
        if include_similar:
            result["similar_samples"] = self.similar_sample_finder.find_similar_samples(
                features, mode, dataset, top_k=3
            )
        return result

    def _get_prediction_and_proba(self, model: Any, vector: pd.DataFrame, model_name: str,
                                  decision_threshold: float = 0.5) -> tuple[int, np.ndarray]:
        """Backward-compatible adapter entry point used by existing tests."""
        adapter = adapter_for(model_name)
        probabilities = adapter.probabilities(model, vector)
        return adapter.prediction(probabilities, decision_threshold), probabilities

    def predict_multiple_models(self, features: dict[str, float], model_names: list[str],
                                mode: str = "optimized", dataset: str = "current"):
        """Predict all requested models and retain similar samples once."""
        mode = normalize_mode(mode)
        dataset = self._normalize_dataset(dataset)
        results, similar = [], None
        for index, name in enumerate(model_names):
            result, similar = self._predict_one(features, name, mode, dataset, index == 0, similar)
            results.append(result)
        return results, similar

    def _predict_one(self, features: dict[str, float], name: str, mode: str, dataset: str,
                     include_similar: bool, similar: Any):
        """Execute one model while converting failures to response records."""
        try:
            result = self.predict_url(features, name, mode, dataset, include_similar)
            return result, result.get("similar_samples", similar)
        except Exception as exc:
            return {"model": name, "mode": mode, "dataset": dataset, "error": str(exc), "success": False}, similar

    def get_ensemble_prediction(self, features: dict[str, float], model_names: list[str],
                                mode: str = "optimized", dataset: str = "current") -> dict[str, Any]:
        """Predict models and aggregate their successful results."""
        predictions, similar = self.predict_multiple_models(features, model_names, mode, dataset)
        return self.get_ensemble_from_predictions(predictions, dataset, similar)

    def get_ensemble_from_predictions(self, predictions: list[dict[str, Any]], dataset: str = "current",
                                      similar_samples: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Build an ensemble response without running inference again."""
        successful = [item for item in predictions if "error" not in item]
        if not successful:
            return {"success": False, "error": "All models failed to predict"}
        votes = self._ensemble_votes(successful)
        result = {"ensemble_prediction": int(votes["phishing"] > votes["safe"]),
                  "ensemble_label": "Phishing" if votes["phishing"] > votes["safe"] else "Safe",
                  "dataset": dataset, "models_voting": votes,
                  "average_phishing_probability": float(np.mean([p["probabilities"]["phishing"] for p in successful])),
                  "confidence": float(max(votes["phishing"], votes["safe"]) / votes["total"]),
                  "individual_predictions": successful}
        if similar_samples:
            result["similar_samples"] = similar_samples
        return result

    @staticmethod
    def _ensemble_votes(predictions: list[dict[str, Any]]) -> dict[str, int]:
        """Count phishing and safe model votes."""
        phishing = sum(item["prediction"] == 1 for item in predictions)
        return {"phishing": phishing, "safe": len(predictions) - phishing, "total": len(predictions)}
