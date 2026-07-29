"""Tests for process-isolated model training."""

import numpy as np
import psutil
from sklearn.linear_model import LogisticRegression

from utils.isolated_training import fit_model_isolated


def test_isolated_training_restores_model_after_worker_releases_resources():
    features = np.array(
        [[0.0], [0.1], [0.9], [1.0], [0.2], [0.8]], dtype=float
    )
    target = np.array([0, 0, 1, 1, 0, 1])

    result = fit_model_isolated(
        LogisticRegression(), features, target,
        interval=0.02, baseline_seconds=0.05,
    )

    assert result.model.predict([[0.05], [0.95]]).tolist() == [0, 1]
    assert result.training_time > 0.0
    assert result.stats["samples"] >= 1
    assert not psutil.pid_exists(result.monitor.process.pid)
    post_indices = [
        index for index, phase in enumerate(result.monitor.phases)
        if phase == "post_training"
    ]
    assert post_indices
    assert result.monitor.cpu_percent[post_indices[-1]] == 0.0
    assert result.monitor.ram_gb[post_indices[-1]] == 0.0
