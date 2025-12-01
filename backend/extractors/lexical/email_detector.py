import re


class EmailDetector:

    EMAIL_PATTERN = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

    def detect(self, url: str) -> int:
        return 1 if re.search(self.EMAIL_PATTERN, url) else 0
