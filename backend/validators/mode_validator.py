class ModeValidator:

    VALID_MODES = {'full', 'optimized'}

    def validate(self, mode: str) -> bool:
        return mode in self.VALID_MODES

    def validate_or_raise(self, mode: str):
        if not self.validate(mode):
            raise ValueError('mode must be either "full" or "optimized"')
