from bs4 import BeautifulSoup


class ExternalResourceAnalyzer:

    RESOURCE_TAGS = {
        'img': 'src', 'script': 'src', 'iframe': 'src',
        'link': 'href', 'embed': 'src', 'audio': 'src', 'video': 'src'
    }

    def analyze(self, soup: BeautifulSoup, base_domain: str) -> int:
        total, external = self._count_resources(soup, base_domain)
        if total == 0:
            return 1

        percentage = (external / total) * 100
        return self._classify_percentage(percentage, 22, 61)

    def _count_resources(self, soup: BeautifulSoup, base_domain: str) -> tuple:
        total = 0
        external = 0

        for tag_name, attr in self.RESOURCE_TAGS.items():
            for tag in soup.find_all(tag_name):
                src = tag.get(attr)
                if src:
                    total += 1
                    if src.startswith('http') and base_domain not in src:
                        external += 1

        return total, external

    def _classify_percentage(self, percentage: float, low: int, high: int) -> int:
        if percentage < low:
            return 1
        elif percentage < high:
            return 0
        return -1
