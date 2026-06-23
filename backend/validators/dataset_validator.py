class DatasetValidator:

    VALID_DATASETS = {'current', 'url_corpus'}

    def validate(self, dataset: str) -> bool:
        return dataset in self.VALID_DATASETS

    def validate_or_raise(self, dataset: str):
        if not self.validate(dataset):
            raise ValueError('dataset must be either "current" or "url_corpus"')
