"""DNS feature extraction with bounded resolver failures."""

import dns.resolver
import dns.exception
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

logger = logging.getLogger("DNSResolver")


class DNSResolver:
    """Resolve A, NS, MX and SPF records for one domain."""

    def __init__(self, timeout: int = 5, lifetime: int | None = None):
        """Initialize a resolver with bounded timeout values."""
        self.resolver = dns.resolver.Resolver()
        self.resolver.timeout = timeout
        self.resolver.lifetime = lifetime or timeout * 2

    def resolve_a_records(self, domain: str) -> tuple:
        """Return A-record count and hostname TTL."""
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
        """Return the number of authoritative name servers."""
        try:
            answers = self.resolver.resolve(domain, 'NS')
            return len(answers)
        except dns.resolver.NoAnswer:
            return 0
        except dns.resolver.NXDOMAIN:
            return -1
        except dns.exception.Timeout:
            logger.debug(f"NS lookup timed out for {domain}")
            return -1
        except Exception as e:
            logger.debug(f"NS lookup failed for {domain}: {e}")
            return -1

    def resolve_mx_records(self, domain: str) -> int:
        """Return the number of mail-exchange records."""
        try:
            answers = self.resolver.resolve(domain, 'MX')
            return len(answers)
        except dns.resolver.NoAnswer:
            return 0
        except dns.resolver.NXDOMAIN:
            return -1
        except dns.exception.Timeout:
            logger.debug(f"MX lookup timed out for {domain}")
            return -1
        except Exception as e:
            logger.debug(f"MX lookup failed for {domain}: {e}")
            return -1

    def check_spf_record(self, domain: str) -> int:
        """Return whether a TXT record contains an SPF declaration."""
        try:
            answers = self.resolver.resolve(domain, 'TXT')
            for rdata in answers:
                if 'spf' in str(rdata).lower():
                    return 1
            return 0
        except dns.resolver.NoAnswer:
            return 0
        except dns.resolver.NXDOMAIN:
            return -1
        except dns.exception.Timeout:
            logger.debug(f"SPF lookup timed out for {domain}")
            return -1
        except Exception as e:
            logger.debug(f"SPF lookup failed for {domain}: {e}")
            return -1

    def extract_all(self, domain: str) -> Dict[str, int]:
        """Resolve all DNS features concurrently."""
        with ThreadPoolExecutor(max_workers=4) as pool:
            a_future = pool.submit(self.resolve_a_records, domain)
            ns_future = pool.submit(self.resolve_ns_records, domain)
            mx_future = pool.submit(self.resolve_mx_records, domain)
            spf_future = pool.submit(self.check_spf_record, domain)
            qty_ip, ttl = a_future.result()
        return {
            'qty_ip_resolved': qty_ip,
            'ttl_hostname': ttl,
            'qty_nameservers': ns_future.result(),
            'qty_mx_servers': mx_future.result(),
            'domain_spf': spf_future.result()
        }
