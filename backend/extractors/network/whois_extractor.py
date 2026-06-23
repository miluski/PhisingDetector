import whois
from datetime import datetime
from typing import Dict


class WHOISExtractor:
    def _first_date(self, value):
        if isinstance(value, list):
            return value[0] if value else None
        return value

    def extract_all(self, domain: str) -> Dict[str, int]:
        try:
            w = whois.whois(domain)
            creation = self._first_date(w.creation_date)
            expiration = self._first_date(w.expiration_date)
            now = datetime.now()
            return {
                'time_domain_activation': (now - creation).days if creation else -1,
                'time_domain_expiration': (expiration - now).days if expiration else -1
            }
        except Exception:
            return {
                'time_domain_activation': -1,
                'time_domain_expiration': -1
            }
