from bs4 import BeautifulSoup


class FormHandler:

    def analyze(self, soup: BeautifulSoup, base_domain: str) -> int:
        forms = soup.find_all('form')
        if not forms:
            return 1

        for form in forms:
            action = form.get('action', '')
            classification = self._classify_action(action, base_domain)
            if classification < 1:
                return classification

        return 1

    def _classify_action(self, action: str, base_domain: str) -> int:
        if not action or action.strip() == '' or action == 'about:blank':
            return -1

        if action.startswith('http') and base_domain not in action:
            return 0

        return 1
