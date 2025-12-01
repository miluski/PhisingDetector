class KeywordDetector:

    KEYWORDS = ['server', 'client', 'host', 'admin', 'user']

    def detect(self, text: str) -> int:
        text_lower = text.lower()
        return 1 if any(kw in text_lower for kw in self.KEYWORDS) else 0
