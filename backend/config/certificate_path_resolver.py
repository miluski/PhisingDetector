import os


class CertificatePathResolver:

    def __init__(self):
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.cert_dir = os.path.join(backend_dir, 'certs')

    def resolve(self):
        cert_file = os.getenv('SSL_CERT_FILE')
        key_file = os.getenv('SSL_KEY_FILE')

        if cert_file and key_file:
            return cert_file, key_file

        return os.path.join(self.cert_dir, 'cert.pem'), os.path.join(self.cert_dir, 'key.pem')

    def validate(self, cert_file, key_file):
        return os.path.exists(cert_file) and os.path.exists(key_file)

