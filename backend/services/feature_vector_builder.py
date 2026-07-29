"""Feature ordering and preprocessing for model inference."""

from __future__ import annotations

from typing import Any

import pandas as pd

from config.feature_policy import MISSING_VALUE, add_missing_indicators


class FeatureVectorBuilder:
    """Build a model-ready one-row dataframe from extracted URL features."""

    def build(self, features: dict[str, float], names: list[str]) -> pd.DataFrame:
        """Order features, add missing indicators and fill unavailable values."""
        enriched = add_missing_indicators(features)
        values = {name: enriched.get(name, MISSING_VALUE) for name in names}
        return pd.DataFrame([values], columns=names).fillna(MISSING_VALUE)

    def scale(self, vector: pd.DataFrame, scaler: Any, names: list[str]) -> pd.DataFrame:
        """Apply a fitted scaler while preserving feature names."""
        if scaler is None:
            return vector
        return pd.DataFrame(scaler.transform(vector), columns=names)
