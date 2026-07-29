"""Shared feature-extraction contract for training and production inference.

Training, external evaluation and application inference must use the same
network lookup mode and representation of unavailable values.
"""

from __future__ import annotations

import os


def _env_flag(name: str, default: bool = False) -> bool:
    """Read an explicit boolean environment flag without truthiness traps."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

WHOIS_ENABLED = _env_flag("URL_WHOIS_ENABLED", default=False)
FEATURE_POLICY_VERSION = (
    "url-v4-111-live-rdap-whois-missing-indicators"
    if WHOIS_ENABLED
    else "url-v3-111-live-no-whois-missing-indicators"
)
NETWORK_MODE = "live"
MISSING_POLICY = "keep"
MISSING_VALUE = -1

WHOIS_FEATURES = (
    "time_domain_activation",
    "time_domain_expiration",
)

NETWORK_FEATURES = (
    "time_response",
    "qty_redirects",
    "qty_ip_resolved",
    "ttl_hostname",
    "qty_nameservers",
    "qty_mx_servers",
    "domain_spf",
    "time_domain_activation",
    "time_domain_expiration",
    "tls_ssl_certificate",
)

_MODEL_MISSING_INDICATORS = (
    "missing_dns",
    "missing_ssl",
    "missing_ttl",
    "missing_http",
)

MISSING_INDICATORS = _MODEL_MISSING_INDICATORS + (
    ("missing_whois",) if WHOIS_ENABLED else ()
)

AUDIT_INDICATORS = () if WHOIS_ENABLED else ("missing_whois",)

MISSING_GROUPS = {
    "missing_dns": (
        "qty_ip_resolved",
        "qty_nameservers",
        "qty_mx_servers",
        "domain_spf",
    ),
    "missing_whois": (
        "time_domain_activation",
        "time_domain_expiration",
    ),
    "missing_ssl": ("tls_ssl_certificate",),
    "missing_ttl": ("ttl_hostname",),
    "missing_http": ("time_response", "qty_redirects"),
}


def missing_indicator(values: dict[str, object], feature_names: tuple[str, ...]) -> int:
    """Return 1 when at least one source feature is unavailable."""
    return int(any(values.get(name, MISSING_VALUE) == MISSING_VALUE for name in feature_names))


def add_missing_indicators(values: dict[str, object]) -> dict[str, object]:
    """Add missingness flags without replacing raw values."""
    enriched = dict(values)
    for indicator, source_features in MISSING_GROUPS.items():
        enriched[indicator] = missing_indicator(enriched, source_features)
    return enriched


def validate_feature_policy(payload: dict[str, object]) -> None:
    """Reject artifacts trained with legacy/default feature semantics."""
    if payload.get("feature_policy") != FEATURE_POLICY_VERSION:
        raise ValueError(
            "Model was not trained with current feature policy: "
            f"expected={FEATURE_POLICY_VERSION!r}, got={payload.get('feature_policy')!r}"
        )
    raw_names = payload.get("feature_names", [])
    feature_names = set(raw_names if isinstance(raw_names, (list, tuple, set)) else [])
    if "asn_ip" in feature_names:
        raise ValueError("Legacy asn_ip feature is forbidden in current model schema")
    _validate_whois_features(feature_names)
    _validate_missing_indicators(feature_names)


def _validate_whois_features(feature_names: set[object]) -> None:
    """Validate WHOIS fields according to the active network policy."""
    whois_present = set(WHOIS_FEATURES) & feature_names
    if WHOIS_ENABLED:
        missing_whois = set(WHOIS_FEATURES) - feature_names
        if missing_whois:
            raise ValueError(
                "WHOIS-enabled policy requires WHOIS date features: "
                f"{sorted(missing_whois)}"
            )
    elif whois_present:
        raise ValueError(
            "WHOIS date features are not valid in the default v3 policy: "
            f"{sorted(whois_present)}"
        )


def _validate_missing_indicators(feature_names: set[object]) -> None:
    """Require explicit missingness indicators in model schemas."""
    missing = set(MISSING_INDICATORS) - feature_names
    if missing:
        raise ValueError(f"Model schema lacks missing indicators: {sorted(missing)}")
