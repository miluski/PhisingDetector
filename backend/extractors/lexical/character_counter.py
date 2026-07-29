"""Character-counting primitives for lexical URL features."""

from typing import Dict


class CharacterCounter:
    """Count configured punctuation characters in URL components."""

    CHARACTERS = {
        'dot': '.', 'hyphen': '-', 'underline': '_', 'slash': '/',
        'questionmark': '?', 'equal': '=', 'at': '@', 'and': '&',
        'exclamation': '!', 'space': ' ', 'tilde': '~', 'comma': ',',
        'plus': '+', 'asterisk': '*', 'hashtag': '#', 'dollar': '$',
        'percent': '%'
    }

    def count(self, text: str, character: str) -> int:
        """Return the number of occurrences of one character."""
        return text.count(character)

    def count_all(self, text: str, char_names: list) -> Dict[str, int]:
        """Return counts for all named characters."""
        return {
            name: self.count(text, self.CHARACTERS[name])
            for name in char_names
        }
