from bs4 import BeautifulSoup


class PopupDetector:

    POPUP_PATTERNS = ['window.open', 'prompt(', 'alert(', 'confirm(']

    def detect(self, soup: BeautifulSoup) -> int:
        scripts = soup.find_all('script')

        for script in scripts:
            if script.string and self._has_popup_pattern(script.string):
                return -1

        return 1

    def _has_popup_pattern(self, script_text: str) -> bool:
        return any(pattern in script_text for pattern in self.POPUP_PATTERNS)
