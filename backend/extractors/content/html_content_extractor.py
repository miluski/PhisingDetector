from typing import Dict
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from .external_resource_analyzer import ExternalResourceAnalyzer
from .anchor_analyzer import AnchorAnalyzer
from .form_handler import FormHandler
from .popup_detector import PopupDetector


class HTMLContentExtractor:

    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self.session = self._create_session()
        self.resource_analyzer = ExternalResourceAnalyzer()
        self.anchor_analyzer = AnchorAnalyzer()
        self.form_handler = FormHandler()
        self.popup_detector = PopupDetector()

    def _create_session(self):
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        return session

    def extract(self, url: str) -> Dict[str, int]:
        try:
            response = self.session.get(url, timeout=self.timeout)
            soup = BeautifulSoup(response.content, 'html.parser')
            base_domain = urlparse(url).netloc

            return {
                'request_url': self.resource_analyzer.analyze(soup, base_domain),
                'url_of_anchor': self.anchor_analyzer.analyze(soup, base_domain),
                'sfh_handler': self.form_handler.analyze(soup, base_domain),
                'popup_window': self.popup_detector.detect(soup),
                'web_traffic': -1
            }
        except Exception:
            return self._default_features()

    def _default_features(self) -> Dict[str, int]:
        return {
            'request_url': -1, 'url_of_anchor': -1,
            'sfh_handler': -1, 'popup_window': -1, 'web_traffic': -1
        }
