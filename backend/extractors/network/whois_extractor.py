import whois
from datetime import datetime
from typing import Dict


class WHOISExtractor:

    def extract_creation_date(self, domain: str) -> int:
        try:
            w = whois.whois(domain)
            if not w.creation_date:
                return -1

            creation = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
            if creation:
                return (datetime.now() - creation).days
            return -1
        except Exception:
            return -1

    def extract_expiration_date(self, domain: str) -> int:
        try:
            w = whois.whois(domain)
            if not w.expiration_date:
                return -1

            expiration = w.expiration_date[0] if isinstance(w.expiration_date, list) else w.expiration_date
            if expiration:
                return (expiration - datetime.now()).days
            return -1
        except Exception:
            return -1

    def extract_all(self, domain: str) -> Dict[str, int]:
        return {
            'time_domain_activation': self.extract_creation_date(domain),
            'time_domain_expiration': self.extract_expiration_date(domain)
        }
