"""Detection of email addresses embedded in URL text."""

import re


class EmailDetector:
    """Detect syntactically recognizable email addresses."""

    EMAIL_PATTERN = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

    def detect(self, url: str) -> int:
        """Return one when the input contains an email address."""
        return 1 if re.search(self.EMAIL_PATTERN, url) else 0
