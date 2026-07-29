"""TLS certificate validation feature extraction."""

import ssl
import socket


class SSLCertificateValidator:
    """Validate whether a domain presents a readable HTTPS certificate."""

    def __init__(self, timeout: int = 5):
        """Initialize certificate socket timeout."""
        self.timeout = timeout

    def validate(self, domain: str) -> int:
        """Return one for a certificate, zero for invalid TLS, -1 for errors."""
        try:
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=self.timeout) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
                    return 1 if cert else 0
            return 0
        except ssl.SSLError:
            return 0
        except Exception:
            return -1
