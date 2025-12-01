from bs4 import BeautifulSoup


class AnchorAnalyzer:

    SUSPICIOUS_HREFS = ['#', 'javascript:void(0)', 'javascript:;', '']

    def analyze(self, soup: BeautifulSoup, base_domain: str) -> int:
        total, suspicious = self._count_anchors(soup, base_domain)
        if total == 0:
            return 1

        percentage = (suspicious / total) * 100
        return self._classify_percentage(percentage, 31, 67)

    def _count_anchors(self, soup: BeautifulSoup, base_domain: str) -> tuple:
        total = 0
        suspicious = 0

        for anchor in soup.find_all('a'):
            href = anchor.get('href')
            if href:
                total += 1
                if self._is_suspicious(href, base_domain):
                    suspicious += 1

        return total, suspicious

    def _is_suspicious(self, href: str, base_domain: str) -> bool:
        if href in self.SUSPICIOUS_HREFS or href.startswith('javascript:'):
            return True
        return href.startswith('http') and base_domain not in href

    def _classify_percentage(self, percentage: float, low: int, high: int) -> int:
        if percentage < low:
            return 1
        elif percentage < high:
            return 0
        return -1
