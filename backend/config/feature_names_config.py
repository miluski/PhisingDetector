import json
from pathlib import Path

from config.feature_policy import MISSING_INDICATORS, WHOIS_ENABLED, WHOIS_FEATURES
from config.feature_modes import normalize_mode


# The v4 corpus selection is the current audited schema used by retraining and inference.
SELECTED_MANIFEST = (
    Path(__file__).resolve().parent
    / "feature_manifests" / "url_corpus" / "quality_v4" / "selected.json"
)


def _load_selected_features() -> list[str]:
    """Load the immutable feature list selected from the training split."""
    if not SELECTED_MANIFEST.exists():
        raise ValueError(
            "selected manifest is missing. Run select_url_corpus_features.py first: "
            f"{SELECTED_MANIFEST}"
        )
    payload = json.loads(SELECTED_MANIFEST.read_text(encoding="utf-8"))
    features = payload.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError(f"Invalid selected manifest: {SELECTED_MANIFEST}")
    return [str(feature) for feature in features]


class FeatureConfig:
    """Canonical ordered feature schemas used by trained models."""
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
        'time_response', 'domain_spf', 'time_domain_activation',
        'time_domain_expiration', 'qty_ip_resolved', 'qty_nameservers',
        'qty_mx_servers', 'ttl_hostname', 'tls_ssl_certificate', 'qty_redirects',
        'url_shortened', 'domain_in_ip', *MISSING_INDICATORS
    ]

    if not WHOIS_ENABLED:
        OPTIMIZED_FEATURES = [
            feature for feature in OPTIMIZED_FEATURES if feature not in WHOIS_FEATURES
        ]

    FULL_FEATURES = OPTIMIZED_FEATURES + [
        'qty_exclamation_url', 'qty_space_url', 'qty_tilde_url', 'qty_comma_url',
        'qty_plus_url', 'qty_asterisk_url', 'qty_hashtag_url', 'qty_dollar_url',
        'qty_percent_url', 'qty_underline_domain', 'qty_slash_domain',
        'qty_questionmark_domain', 'qty_equal_domain', 'qty_at_domain',
        'qty_and_domain', 'qty_exclamation_domain', 'qty_space_domain',
        'qty_tilde_domain', 'qty_comma_domain', 'qty_plus_domain',
        'qty_asterisk_domain', 'qty_hashtag_domain', 'qty_dollar_domain',
        'qty_percent_domain', 'server_client_domain',
        'qty_dollar_file', 'tld_present_params', 'qty_equal_params', 'qty_exclamation_params',
        'qty_tilde_params', 'qty_comma_params', 'qty_plus_params',
        'qty_hashtag_params', 'qty_dollar_params'
    ]

    @staticmethod
    def get_feature_names(mode: str) -> list:
        """Return the canonical feature order for a model mode."""
        mode = normalize_mode(mode)
        if mode == 'selected':
            return _load_selected_features()
        if mode == 'optimized':
            return FeatureConfig.OPTIMIZED_FEATURES
        return FeatureConfig.FULL_FEATURES
