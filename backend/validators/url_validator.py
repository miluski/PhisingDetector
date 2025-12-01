class URLValidator:

    VALID_SCHEMES = ('http://', 'https://')

    def validate(self, url: str) -> bool:
        if not url:
            return False
        return url.startswith(self.VALID_SCHEMES)

    def validate_or_raise(self, url: str):
        if not self.validate(url):
            raise ValueError('URL must start with http:// or https://')
