class FeatureConfig:
    OPTIMIZED_FEATURES = [
        'qty_dot_url', 'qty_hyphen_url', 'qty_underline_url', 'qty_slash_url',
        'qty_questionmark_url', 'qty_equal_url', 'qty_at_url', 'qty_and_url',
        'qty_tld_url', 'length_url', 'qty_dot_domain', 'qty_hyphen_domain',
        'qty_vowels_domain', 'domain_length', 'qty_dot_directory', 'qty_hyphen_directory',
        'qty_underline_directory', 'qty_slash_directory', 'qty_questionmark_directory',
        'qty_equal_directory', 'qty_at_directory', 'qty_and_directory',
        'qty_exclamation_directory', 'qty_space_directory', 'qty_tilde_directory',
        'qty_comma_directory', 'qty_plus_directory', 'qty_asterisk_directory',
        'qty_hashtag_directory', 'qty_dollar_directory', 'qty_percent_directory',
        'directory_length', 'qty_dot_file', 'qty_hyphen_file', 'qty_underline_file',
        'qty_slash_file', 'qty_questionmark_file', 'qty_equal_file', 'qty_at_file',
        'qty_and_file', 'qty_exclamation_file', 'qty_space_file', 'qty_tilde_file',
        'qty_comma_file', 'qty_plus_file', 'qty_asterisk_file', 'qty_hashtag_file',
        'qty_percent_file', 'file_length', 'qty_dot_params', 'qty_hyphen_params',
        'qty_underline_params', 'qty_slash_params', 'qty_questionmark_params',
        'qty_at_params', 'qty_and_params', 'qty_space_params', 'qty_asterisk_params',
        'qty_percent_params', 'params_length', 'qty_params', 'email_in_url',
        'time_response', 'domain_spf', 'asn_ip', 'time_domain_activation',
        'time_domain_expiration', 'qty_ip_resolved', 'qty_nameservers',
        'qty_mx_servers', 'ttl_hostname', 'tls_ssl_certificate', 'qty_redirects',
        'url_shortened'
    ]

    FULL_FEATURES = OPTIMIZED_FEATURES + [
        'qty_exclamation_url', 'qty_space_url', 'qty_tilde_url', 'qty_comma_url',
        'qty_plus_url', 'qty_asterisk_url', 'qty_hashtag_url', 'qty_dollar_url',
        'qty_percent_url', 'qty_underline_domain', 'qty_slash_domain',
        'qty_questionmark_domain', 'qty_equal_domain', 'qty_at_domain',
        'qty_and_domain', 'qty_exclamation_domain', 'qty_space_domain',
        'qty_tilde_domain', 'qty_comma_domain', 'qty_plus_domain',
        'qty_asterisk_domain', 'qty_hashtag_domain', 'qty_dollar_domain',
        'qty_percent_domain', 'domain_in_ip', 'server_client_domain',
        'qty_dollar_file', 'tld_present_params', 'url_google_index',
        'domain_google_index'
    ]

    @staticmethod
    def get_feature_names(mode: str) -> list:
        if mode == 'optimized':
            return FeatureConfig.OPTIMIZED_FEATURES
        return FeatureConfig.FULL_FEATURES
