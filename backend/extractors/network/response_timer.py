"""HTTP response-time and redirect-count extraction."""

import time
import requests
import logging
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logger = logging.getLogger("ResponseTimer")


class ResponseTimer:
    """Measure HTTP response time using one reusable requests session."""

    def __init__(self, timeout: int = 10):
        """Initialize a session with the configured request timeout."""
        self.timeout = timeout
        self.session = self._create_session()

    def _create_session(self):
        session = requests.Session()
        session.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            )
        })
        return session

    def measure(self, url: str) -> tuple:
        """Return elapsed seconds and redirect count, or ``(-1, -1)``."""
        try:
            start = time.time()
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True, verify=False)
            elapsed = time.time() - start
            redirects = len(response.history)
            logger.debug(f"Response for {url}: {elapsed:.3f}s, {redirects} redirects")
            return elapsed, redirects
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout for {url}")
            return -1, -1
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error for {url}: {e}")
            return -1, -1
        except Exception as e:
            logger.warning(f"Error measuring {url}: {e}")
            return -1, -1
