class VowelCounter:

    VOWELS = 'aeiou'

    def count(self, text: str) -> int:
        return sum(1 for c in text.lower() if c in self.VOWELS)
