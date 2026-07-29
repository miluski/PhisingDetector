"""Validation of feature extraction modes."""

from config.feature_modes import SUPPORTED_MODES


class ModeValidator:
    """Validate canonical feature modes and the legacy selection alias."""

    VALID_MODES = set(SUPPORTED_MODES)

    def validate(self, mode: str) -> bool:
        """Return whether a mode name is supported."""
        return mode in self.VALID_MODES

    def validate_or_raise(self, mode: str):
        """Validate a mode or raise a client-facing ``ValueError``."""
        if not self.validate(mode):
            raise ValueError('mode must be "full", "optimized" or "selected"')
