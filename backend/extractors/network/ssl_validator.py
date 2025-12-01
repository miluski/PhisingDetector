import ssl
import socket


class SSLCertificateValidator:

    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    def validate(self, domain: str) -> int:
        try:
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=self.timeout) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
                    return 1 if cert else 0
            return 0
        except Exception:
            return 0
