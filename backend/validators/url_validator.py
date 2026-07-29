"""Validation of URLs accepted by extraction endpoints."""


class URLValidator:
    """Require HTTP or HTTPS URL schemes."""

    VALID_SCHEMES = ('http://', 'https://')

    def validate(self, url: str) -> bool:
        """Return whether a URL has a supported non-empty scheme."""
        if not url:
            return False
        return url.startswith(self.VALID_SCHEMES)

    def validate_or_raise(self, url: str):
        """Validate a URL or raise a client-facing ``ValueError``."""
        if not self.validate(url):
            raise ValueError('URL must start with http:// or https://')
