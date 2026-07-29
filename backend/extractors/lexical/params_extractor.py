"""Extraction of query-parameter lexical features."""

from typing import Dict
from urllib.parse import parse_qs
from .character_counter import CharacterCounter
from .tld_counter import TLDCounter


class ParamsExtractor:
    """Extract punctuation, parameter count and optional TLD features."""

    def __init__(self):
        """Initialize character and TLD counters."""
        self.char_counter = CharacterCounter()
        self.tld_counter = TLDCounter()

    def extract_features(self, params: str, char_names: list, include_tld: bool = False) -> Dict[str, int]:
        """Return canonical query-parameter feature names and values."""
        features = self.char_counter.count_all(params, char_names)
        features['params_length'] = len(params)
        features['qty_params'] = len(parse_qs(params)) if params else 0

        if include_tld:
            tld_count = self.tld_counter.count(params)
            features['tld_present_params'] = 1 if tld_count > 0 else 0

        result = {}
        for k, v in features.items():
            if k in ['params_length', 'qty_params', 'tld_present_params']:
                result[k] = v
            else:
                result[f'qty_{k}_params'] = v
        return result
