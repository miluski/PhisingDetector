import pytest

from validators.model_validator import ModelValidator


def test_model_validator_accepts_all_selectable_models():
    ModelValidator().validate_or_raise([
        'random_forest', 'xgboost', 'mlp', 'svm', 'logistic_regression'
    ])


def test_model_validator_rejects_unknown_models():
    with pytest.raises(ValueError, match='Unsupported model'):
        ModelValidator().validate_or_raise(['random_forest', 'unknown'])
