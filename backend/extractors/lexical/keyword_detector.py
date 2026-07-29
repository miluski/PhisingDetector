"""Detection of administrative and hosting keywords in domains."""


class KeywordDetector:
    """Detect configured server or administration keywords."""

    KEYWORDS = ['server', 'client', 'host', 'admin', 'user']

    def detect(self, text: str) -> int:
        """Return one when any configured keyword occurs in the text."""
        text_lower = text.lower()
        return 1 if any(kw in text_lower for kw in self.KEYWORDS) else 0
