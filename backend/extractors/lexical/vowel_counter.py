"""Vowel counting primitive for domain lexical features."""


class VowelCounter:
    """Count ASCII vowels in normalized domain text."""

    VOWELS = 'aeiou'

    def count(self, text: str) -> int:
        """Return the number of configured vowels in text."""
        return sum(1 for c in text.lower() if c in self.VOWELS)
