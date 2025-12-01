import re


class IPAddressDetector:

    IP_PATTERN = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'

    def detect(self, domain: str) -> int:
        return 1 if re.match(self.IP_PATTERN, domain) else 0
