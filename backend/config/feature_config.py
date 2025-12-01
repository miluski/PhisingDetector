class FeatureSetConfig:

    URL_CHARS_OPTIMIZED = [
        'dot', 'hyphen', 'underline', 'slash', 'questionmark',
        'equal', 'at', 'and'
    ]

    URL_CHARS_FULL = [
        'dot', 'hyphen', 'underline', 'slash', 'questionmark',
        'equal', 'at', 'and', 'exclamation', 'space', 'tilde',
        'comma', 'plus', 'asterisk', 'hashtag', 'dollar', 'percent'
    ]

    DOMAIN_CHARS_OPTIMIZED = ['dot', 'hyphen']
    DOMAIN_CHARS_FULL = URL_CHARS_FULL

    PARAMS_CHARS_OPTIMIZED = [
        'dot', 'hyphen', 'underline', 'slash', 'questionmark',
        'at', 'and', 'space', 'asterisk', 'percent'
    ]

    PARAMS_CHARS_FULL = URL_CHARS_FULL

    FILE_EXCLUDE_OPTIMIZED = ['dollar']

    @staticmethod
    def get_url_chars(mode: str) -> list:
        return (FeatureSetConfig.URL_CHARS_OPTIMIZED if mode == 'optimized'
                else FeatureSetConfig.URL_CHARS_FULL)

    @staticmethod
    def get_domain_chars(mode: str) -> list:
        return (FeatureSetConfig.DOMAIN_CHARS_OPTIMIZED if mode == 'optimized'
                else FeatureSetConfig.DOMAIN_CHARS_FULL)

    @staticmethod
    def get_params_chars(mode: str) -> list:
        return (FeatureSetConfig.PARAMS_CHARS_OPTIMIZED if mode == 'optimized'
                else FeatureSetConfig.PARAMS_CHARS_FULL)

    @staticmethod
    def get_file_exclude(mode: str) -> list:
        return FeatureSetConfig.FILE_EXCLUDE_OPTIMIZED if mode == 'optimized' else []
