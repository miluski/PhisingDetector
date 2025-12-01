from typing import Dict


class CharacterCounter:

    CHARACTERS = {
        'dot': '.', 'hyphen': '-', 'underline': '_', 'slash': '/',
        'questionmark': '?', 'equal': '=', 'at': '@', 'and': '&',
        'exclamation': '!', 'space': ' ', 'tilde': '~', 'comma': ',',
        'plus': '+', 'asterisk': '*', 'hashtag': '#', 'dollar': '$',
        'percent': '%'
    }

    def count(self, text: str, character: str) -> int:
        return text.count(character)

    def count_all(self, text: str, char_names: list) -> Dict[str, int]:
        return {
            name: self.count(text, self.CHARACTERS[name])
            for name in char_names
        }
