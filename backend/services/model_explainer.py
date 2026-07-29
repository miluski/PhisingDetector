"""Optional explanations for model predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class ModelExplainer:
    """Generate model-specific details without changing predictions."""

    def random_forest(self, model: Any, vector: pd.DataFrame, raw_features: dict[str, float]) -> dict[str, Any]:
        """Return tree votes and the ten most important feature values."""
        votes = self._votes(model, vector)
        names = vector.columns.tolist()
        importance = np.asarray(model.feature_importances_)
        order = np.argsort(importance)[::-1][:10]
        top = [self._feature_record(names, importance, raw_features, index) for index in order]
        return {"voting": self._vote_result(votes), "top_features": top}

    @staticmethod
    def _votes(model: Any, vector: pd.DataFrame) -> np.ndarray:
        """Collect one prediction from every tree."""
        return np.asarray([tree.predict(vector.values)[0] for tree in model.estimators_])

    @staticmethod
    def _feature_record(names: list[str], importance: np.ndarray, raw: dict[str, float], index: int) -> dict[str, Any]:
        """Build one feature explanation record."""
        name = names[index]
        return {"name": name, "importance": float(importance[index]), "value": float(raw.get(name, 0))}

    @staticmethod
    def _vote_result(votes: np.ndarray) -> dict[str, int | float]:
        """Summarize tree votes using the response schema."""
        total = len(votes)
        safe = int(np.sum(votes == 0))
        phishing = int(np.sum(votes == 1))
        return {"safe": safe, "phishing": phishing, "total_trees": total,
                "safe_percentage": safe / total * 100, "phishing_percentage": phishing / total * 100}
