"""Validation of supported dataset selectors."""


class DatasetValidator:
    """Validate current and URL-corpus dataset names."""

    VALID_DATASETS = {'current', 'url_corpus'}

    def validate(self, dataset: str) -> bool:
        """Return whether a dataset name is supported."""
        return dataset in self.VALID_DATASETS

    def validate_or_raise(self, dataset: str):
        """Validate a dataset or raise a client-facing ``ValueError``."""
        if not self.validate(dataset):
            raise ValueError('dataset must be either "current" or "url_corpus"')
