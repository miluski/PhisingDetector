"""Loading and caching trained model artifacts."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

from config.feature_names_config import FeatureConfig
from config.feature_policy import FEATURE_POLICY_VERSION
from config.feature_modes import normalize_mode

logger = logging.getLogger("ModelRepository")


class ModelRepository:
    """Load model payloads and optional scalers from the model directory."""

    MODEL_SUFFIXES = {"xgboost", "mlp", "svm"}
    VALID_DATASETS = {"current", "url_corpus"}

    def __init__(self, models_dir: Path):
        """Create a repository rooted at ``models_dir``."""
        self.models_dir = Path(models_dir)
        self.models: dict[str, dict[str, Any]] = {}
        self.scalers: dict[str, Any] = {}

    def normalize_dataset(self, dataset: str | None) -> str:
        """Return a supported dataset name, defaulting to ``current``."""
        value = (dataset or "current").strip().lower()
        return value if value in self.VALID_DATASETS else "current"

    def load(self, model_name: str, mode: str, dataset: str) -> dict[str, Any]:
        """Load and cache a model payload with its schema metadata."""
        mode = normalize_mode(mode)
        dataset = self.normalize_dataset(dataset)
        key = self._cache_key(model_name, mode, dataset)
        if key not in self.models:
            self.models[key] = self._read_payload(model_name, mode, dataset)
        return self.models[key]

    def _cache_key(self, model_name: str, mode: str, dataset: str) -> str:
        """Build a stable cache key."""
        return f"{model_name}_{mode}_{dataset}"

    def _read_payload(self, model_name: str, mode: str, dataset: str) -> dict[str, Any]:
        """Read a model file and normalize legacy payloads."""
        path = self._model_path(model_name, mode, dataset)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        with path.open("rb") as file:
            loaded = pickle.load(file)
        payload = loaded if isinstance(loaded, dict) and "model" in loaded else None
        return payload or self._legacy_payload(loaded, model_name, mode, dataset)

    def _legacy_payload(self, model: Any, model_name: str, mode: str, dataset: str) -> dict[str, Any]:
        """Wrap a legacy bare model and recover its schema and scaler."""
        scaler = self._load_scaler(model_name, mode, dataset)
        names = self._feature_names(scaler, model, mode)
        return {"model": model, "scaler": scaler, "feature_names": names}

    def _model_path(self, model_name: str, mode: str, dataset: str) -> Path:
        """Return the artifact path for a model variant."""
        suffix = "" if dataset == "current" else "_url_corpus"
        stem = f"{model_name}_model" if model_name in self.MODEL_SUFFIXES else model_name
        return self.models_dir / model_name / f"{stem}_{mode}{suffix}.pkl"

    def _scaler_path(self, model_name: str, mode: str, dataset: str) -> Path:
        """Return the scaler path for a model variant."""
        suffix = "" if dataset == "current" else "_url_corpus"
        name = f"{model_name}_scaler_{mode}{suffix}.pkl"
        return self.models_dir / model_name / name

    def _load_scaler(self, model_name: str, mode: str, dataset: str) -> Any:
        """Load a scaler when the legacy artifact requires one."""
        key = self._cache_key(model_name + "_scaler", mode, dataset)
        if key in self.scalers:
            return self.scalers[key]
        path = self._scaler_path(model_name, mode, dataset)
        if not path.exists():
            return None
        with path.open("rb") as file:
            self.scalers[key] = pickle.load(file)
        return self.scalers[key]

    def _feature_names(self, scaler: Any, model: Any, mode: str) -> list[str]:
        """Recover feature names from an artifact or the current schema."""
        for candidate in (scaler, model):
            names = getattr(candidate, "feature_names_in_", None)
            if names is not None:
                return list(names)
        return FeatureConfig.get_feature_names(mode)

    @staticmethod
    def validate_policy(payload: dict[str, Any]) -> None:
        """Reject artifacts explicitly tagged with another feature policy."""
        policy = payload.get("feature_policy")
        if policy and policy != FEATURE_POLICY_VERSION:
            raise ValueError(f"Model feature policy mismatch: {policy!r}")
