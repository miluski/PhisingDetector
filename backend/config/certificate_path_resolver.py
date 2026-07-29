"""Resolve certificate paths used by the HTTPS server."""

import os


class CertificatePathResolver:
    """Resolve explicit environment certificates or backend defaults."""

    def __init__(self):
        """Initialize the default certificate directory."""
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.cert_dir = os.path.join(backend_dir, 'certs')

    def resolve(self):
        """Return certificate and key paths selected by environment settings."""
        cert_file = os.getenv('SSL_CERT_FILE')
        key_file = os.getenv('SSL_KEY_FILE')

        if cert_file and key_file:
            return cert_file, key_file

        return os.path.join(self.cert_dir, 'cert.pem'), os.path.join(self.cert_dir, 'key.pem')

    def validate(self, cert_file, key_file):
        """Return whether both certificate files exist."""
        return os.path.exists(cert_file) and os.path.exists(key_file)
