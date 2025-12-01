from typing import Dict
from .character_counter import CharacterCounter


class PathPartExtractor:

    def __init__(self):
        self.char_counter = CharacterCounter()

    def extract_directory_features(self, directory: str) -> Dict[str, int]:
        features = self.char_counter.count_all(
            directory,
            list(CharacterCounter.CHARACTERS.keys())
        )
        result = {f'qty_{k}_directory': v for k, v in features.items()}
        result['directory_length'] = len(directory)
        return result

    def extract_file_features(self, file_part: str, exclude: list = None) -> Dict[str, int]:
        char_names = list(CharacterCounter.CHARACTERS.keys())
        if exclude:
            char_names = [c for c in char_names if c not in exclude]

        features = self.char_counter.count_all(file_part, char_names)
        result = {f'qty_{k}_file': v for k, v in features.items()}
        result['file_length'] = len(file_part)
        return result
