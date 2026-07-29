from config.feature_names_config import FeatureConfig
from config.feature_policy import (
    AUDIT_INDICATORS,
    MISSING_INDICATORS,
    add_missing_indicators,
    validate_feature_policy,
)
from services.url_feature_service import URLFeatureService


def test_current_schema_has_no_constant_asn_and_has_missing_indicators():
    for mode in ("optimized", "full"):
        names = FeatureConfig.get_feature_names(mode)
        assert "asn_ip" not in names
        assert set(MISSING_INDICATORS).issubset(names)
        assert set(AUDIT_INDICATORS).isdisjoint(names)


def test_full_schema_keeps_original_parameter_features_without_placeholders():
    names = set(FeatureConfig.get_feature_names("full"))
    assert {
        "qty_comma_params",
        "qty_dollar_params",
        "qty_equal_params",
        "qty_exclamation_params",
        "qty_hashtag_params",
        "qty_plus_params",
        "qty_tilde_params",
    }.issubset(names)
    assert {"url_google_index", "domain_google_index"}.isdisjoint(names)
    assert len(names) == 110


def test_offline_extraction_marks_network_groups_missing():
    values = URLFeatureService().extract_offline("https://example.com/login", "full")
    assert "asn_ip" not in values
    assert all(values[name] == 1 for name in MISSING_INDICATORS)
    assert "missing_whois" not in values


def test_extractor_returns_exact_central_schema_for_both_modes():
    service = URLFeatureService()
    for mode in ("optimized", "selected", "full"):
        values = service.extract_offline("https://example.com/login?a=1", mode)
        assert list(values) == FeatureConfig.get_feature_names(mode)


def test_domain_ip_feature_is_computed_for_every_mode():
    service = URLFeatureService()
    for mode in ("optimized", "selected", "full"):
        hostname = service.extract_offline("https://example.com/login", mode)
        ip_literal = service.extract_offline("http://192.0.2.10/login", mode)
        assert hostname["domain_in_ip"] == 0
        assert ip_literal["domain_in_ip"] == 1


def test_missing_indicators_preserve_observed_values():
    values = add_missing_indicators({
        "qty_ip_resolved": 1,
        "qty_nameservers": 2,
        "qty_mx_servers": 0,
        "domain_spf": 0,
        "time_domain_activation": -1,
        "time_domain_expiration": 10,
        "tls_ssl_certificate": 1,
        "ttl_hostname": 120,
    })
    assert values["missing_dns"] == 0
    assert values["missing_whois"] == 1
    assert values["missing_ssl"] == 0
    assert values["missing_ttl"] == 0


def test_legacy_model_payload_is_rejected():
    try:
        validate_feature_policy({"feature_policy": "legacy", "feature_names": ["asn_ip"]})
    except ValueError as error:
        assert "current feature policy" in str(error)
    else:
        raise AssertionError("legacy payload must be rejected")
