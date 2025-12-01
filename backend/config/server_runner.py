from config.certificate_path_resolver import CertificatePathResolver


class ServerRunner:

    def __init__(self, app, host='0.0.0.0', port=5001):
        self.app = app
        self.host = host
        self.port = port
        self.resolver = CertificatePathResolver()

    def run(self):
        cert_file, key_file = self.resolver.resolve()

        if self.resolver.validate(cert_file, key_file):
            self._run_https(cert_file, key_file)
        else:
            self._run_http(cert_file, key_file)

    def _run_https(self, cert_file, key_file):
        print("")
        print(f"🔒 API available at: https://localhost:{self.port}")
        print("⚠️  Self-signed certificate - browser warnings expected")
        print(f"Using cert: {cert_file}")
        print(f"Using key:  {key_file}")
        print("")
        self.app.run(debug=False, host=self.host, port=self.port,
                    ssl_context=(cert_file, key_file))

    def _run_http(self, cert_file, key_file):
        print("⚠️  SSL certificates not found!")
        print(f"Checked: {cert_file}")
        print(f"Checked: {key_file}")
        print("")
        print("Run './generate_certs.sh' to create certificates")
        print("")
        print(f"Starting without HTTPS at: http://localhost:{self.port}")
        print("")
        self.app.run(debug=False, host=self.host, port=self.port)
