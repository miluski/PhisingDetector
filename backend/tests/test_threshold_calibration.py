"""Tests for validation-only decision-threshold calibration."""

import pytest

from utils.threshold_calibration import calibrate_threshold


def test_calibration_prefers_the_distribution_shift_safety_margin():
    """A separable validation set should use the lower target FPR."""
    result = calibrate_threshold(
        [0, 0, 0, 1, 1, 1],
        [0.10, 0.20, 0.30, 0.70, 0.80, 0.90],
    )

    assert result["validation_fpr"] == 0.0
    assert result["validation_recall"] == 1.0
    assert result["safety_margin_satisfied"]


def test_calibration_may_relax_only_to_the_hard_fpr_limit():
    """Recall may require the hard limit when the safety target is infeasible."""
    result = calibrate_threshold(
        [0, 0, 1, 1],
        [0.10, 0.40, 0.30, 0.80],
        target_false_positive_rate=0.0,
        max_false_positive_rate=0.5,
        min_recall=1.0,
    )

    assert result["validation_fpr"] == 0.5
    assert result["validation_recall"] == 1.0
    assert not result["safety_margin_satisfied"]


def test_relaxed_calibration_uses_the_lowest_fpr_that_retains_recall():
    """A relaxed threshold must remain conservative instead of maximising F1."""
    result = calibrate_threshold(
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0.10, 0.20, 0.35, 0.45, 0.80, 0.30, 0.40, 0.50, 0.60, 0.70],
        target_false_positive_rate=0.2,
        max_false_positive_rate=0.6,
        min_recall=0.8,
    )

    assert result["validation_fpr"] == 0.4
    assert result["validation_recall"] == 0.8
    assert not result["safety_margin_satisfied"]


def test_calibration_rejects_an_impossible_operating_point():
    """No model artifact should be saved with violated hard constraints."""
    with pytest.raises(ValueError, match="No validation threshold"):
        calibrate_threshold(
            [0, 0, 1, 1],
            [0.80, 0.90, 0.10, 0.20],
            target_false_positive_rate=0.0,
            max_false_positive_rate=0.0,
            min_recall=1.0,
        )


@pytest.mark.parametrize(
    ("labels", "scores"),
    [([], []), ([0, 1], [0.2]), ([0, 0], [0.1, 0.2])],
)
def test_calibration_rejects_invalid_validation_data(labels, scores):
    """Calibration requires aligned labels and both binary classes."""
    with pytest.raises(ValueError):
        calibrate_threshold(labels, scores)
