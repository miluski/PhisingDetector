"""Extraction of URL and domain lexical features."""

from typing import Dict
from .character_counter import CharacterCounter
from .vowel_counter import VowelCounter
from .tld_counter import TLDCounter


class URLPartExtractor:
    """Extract punctuation, vowel, TLD and length features."""

    def __init__(self):
        """Initialize lexical counters used by URL parts."""
        self.char_counter = CharacterCounter()
        self.vowel_counter = VowelCounter()
        self.tld_counter = TLDCounter()

    def extract_url_features(self, url: str, char_names: list) -> Dict[str, int]:
        """Return canonical lexical features for the complete URL."""
        features = self.char_counter.count_all(url, char_names)
        result = {f'qty_{k}_url': v for k, v in features.items()}
        result['qty_tld_url'] = self.tld_counter.count(url)
        result['length_url'] = len(url)
        return result

    def extract_domain_features(self, domain: str, char_names: list) -> Dict[str, int]:
        """Return canonical lexical features for the registrable domain."""
        features = self.char_counter.count_all(domain, char_names)
        result = {f'qty_{k}_domain': v for k, v in features.items()}
        result['qty_vowels_domain'] = self.vowel_counter.count(domain)
        result['domain_length'] = len(domain)
        return result
