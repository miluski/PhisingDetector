import dns.resolver
import logging
from typing import Dict

logger = logging.getLogger("DNSResolver")


class DNSResolver:

    def __init__(self):
        self.resolver = dns.resolver.Resolver()
        self.resolver.timeout = 5
        self.resolver.lifetime = 10

    def resolve_a_records(self, domain: str) -> tuple:
        try:
            answers = self.resolver.resolve(domain, 'A')
            result = len(answers), answers.rrset.ttl if answers else -1
            logger.debug(f"A records for {domain}: {result}")
            return result
        except dns.resolver.NXDOMAIN:
            logger.warning(f"Domain {domain} does not exist")
            return -1, -1
        except dns.resolver.NoAnswer:
            logger.warning(f"No A records for {domain}")
            return 0, -1
        except Exception as e:
            logger.warning(f"DNS A lookup failed for {domain}: {e}")
            return -1, -1

    def resolve_ns_records(self, domain: str) -> int:
        try:
            answers = self.resolver.resolve(domain, 'NS')
            return len(answers)
        except Exception as e:
            logger.debug(f"NS lookup failed for {domain}: {e}")
            return 0

    def resolve_mx_records(self, domain: str) -> int:
        try:
            answers = self.resolver.resolve(domain, 'MX')
            return len(answers)
        except Exception as e:
            logger.debug(f"MX lookup failed for {domain}: {e}")
            return 0

    def check_spf_record(self, domain: str) -> int:
        try:
            answers = self.resolver.resolve(domain, 'TXT')
            for rdata in answers:
                if 'spf' in str(rdata).lower():
                    return 1
            return 0
        except Exception as e:
            logger.debug(f"SPF lookup failed for {domain}: {e}")
            return 0

    def extract_all(self, domain: str) -> Dict[str, int]:
        qty_ip, ttl = self.resolve_a_records(domain)
        return {
            'qty_ip_resolved': qty_ip,
            'ttl_hostname': ttl,
            'qty_nameservers': self.resolve_ns_records(domain),
            'qty_mx_servers': self.resolve_mx_records(domain),
            'domain_spf': self.check_spf_record(domain)
        }
