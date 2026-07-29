"""Detection of known URL-shortening services."""


class URLShortenerDetector:
    """Detect domains belonging to configured shortening providers."""

    SHORTENER_DOMAINS = [
        'bit.ly', 'tinyurl.com', 'goo.gl', 'ow.ly', 't.co',
        'is.gd', 'cli.gs', 'pic.gd', 'DwarfURL.com'
    ]

    def detect(self, domain: str) -> int:
        """Return one when a known shortener domain occurs."""
        domain_lower = domain.lower()
        return 1 if any(s in domain_lower for s in self.SHORTENER_DOMAINS) else 0
