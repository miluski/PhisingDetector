"""Detection of IPv4 literals used as URL domains."""

import re


class IPAddressDetector:
    """Detect simple dotted IPv4 domain representations."""

    IP_PATTERN = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'

    def detect(self, domain: str) -> int:
        """Return one when the domain matches the IPv4 literal pattern."""
        return 1 if re.match(self.IP_PATTERN, domain) else 0
