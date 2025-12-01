class TLDCounter:

    COMMON_TLDS = {
        '.com', '.org', '.net', '.edu', '.gov', '.mil', '.int',
        '.co', '.io', '.info', '.biz', '.me', '.tv', '.cc',
        '.uk', '.de', '.fr', '.jp', '.cn', '.ru', '.br', '.au',
        '.in', '.nl', '.pl', '.es', '.it', '.ca', '.kr', '.se'
    }

    def count(self, text: str) -> int:
        text_lower = text.lower()
        return sum(text_lower.count(tld) for tld in self.COMMON_TLDS)
