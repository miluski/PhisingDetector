"""Regression tests for robust retraining metric masks."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.scripts.modeling.retrain_url_corpus_all_models import metrics


def test_metrics_accepts_malformed_unicode_host() -> None:
    """Malformed host syntax must not abort metric calculation."""
    urls = pd.Series([
        "https://sample.𝕧𝕚𝕣𝕡𝕞𝕠𝕛.cn/path",
        "https://example.com/",
    ])
    result = metrics(pd.Series([0, 1]), np.array([0.2, 0.8]), 0.5, urls)
    assert result["accuracy"] == 1.0
    assert result["benign_root_samples"] == 0
