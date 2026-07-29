"""Validation of model names accepted by API resources."""


class ModelValidator:
    """Reject unsupported model identifiers before inference."""
    VALID_MODELS = {
        'random_forest',
        'xgboost',
        'mlp',
        'svm',
        'logistic_regression',
    }

    def validate_or_raise(self, models):
        """Validate a model collection or raise a descriptive ``ValueError``."""
        invalid = sorted(set(models or ()) - self.VALID_MODELS)
        if invalid:
            raise ValueError(
                'Unsupported model(s): ' + ', '.join(invalid)
            )
