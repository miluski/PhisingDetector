import os
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any
from urllib.parse import urlparse
import tldextract
from config.feature_config import FeatureSetConfig
from config.feature_names_config import FeatureConfig
from config.feature_modes import normalize_mode
from config.feature_policy import MISSING_VALUE, NETWORK_FEATURES, add_missing_indicators
from config.feature_policy import WHOIS_ENABLED, WHOIS_FEATURES
from extractors.lexical.url_part_extractor import URLPartExtractor
from extractors.lexical.path_part_extractor import PathPartExtractor
from extractors.lexical.params_extractor import ParamsExtractor
from extractors.lexical.pattern_detector import (
    EmailDetector, IPAddressDetector, KeywordDetector, URLShortenerDetector
)
from extractors.network.response_timer import ResponseTimer
from extractors.network.dns_resolver import DNSResolver
from extractors.network.whois_extractor import WHOISExtractor
from extractors.network.ssl_validator import SSLCertificateValidator

logger = logging.getLogger("URLFeatureService")


class URLFeatureService:
    """Extract the canonical lexical and network feature schema for URLs."""

    NETWORK_FEATURE_DEFAULTS = {
        'time_response': 0.50,
        'qty_redirects': 0,
        'qty_ip_resolved': 1,
        'ttl_hostname': 3440,
        'qty_nameservers': 2,
        'qty_mx_servers': 1,
        'domain_spf': 0,
        'time_domain_activation': 5189,
        'time_domain_expiration': 281,
        'tls_ssl_certificate': 1,
    }

    def __init__(self, timeout: int = 5, apply_network_defaults: bool = False):
        self.timeout = timeout
        self.apply_network_defaults = apply_network_defaults
        self.tld_extractor = self._create_tld_extractor()
        self._init_extractors()
        self._init_detectors()

    def _init_extractors(self):
        self.url_extractor = URLPartExtractor()
        self.path_extractor = PathPartExtractor()
        self.params_extractor = ParamsExtractor()
        self.response_timer = ResponseTimer(self.timeout)
        self.dns_resolver = DNSResolver(self.timeout)
        self.whois_extractor = WHOISExtractor(self.timeout)
        self.ssl_validator = SSLCertificateValidator(self.timeout)

    def _init_detectors(self):
        self.email_detector = EmailDetector()
        self.ip_detector = IPAddressDetector()
        self.keyword_detector = KeywordDetector()
        self.shortener_detector = URLShortenerDetector()

    def extract(self, url: str, mode: str) -> Dict[str, Any]:
        """Extract live features while isolating failures per extractor group."""
        mode = normalize_mode(mode)
        url = self._normalize_url(url)
        logger.info(f"   🔍 Extracting features for: {url}")
        parts = self._url_parts(url)
        features = self._extract_groups(url, parts, mode)
        features = self._complete_schema(features, mode)
        logger.info(f"   ✅ Extracted {len(features)} features")
        return features

    def _url_parts(self, url: str):
        """Return parsed URL, domain, directory, file and query components."""
        parsed, extracted = self._safe_parse(url)
        domain = self._build_domain(extracted)
        path = parsed.path.rsplit('/', 1)
        return parsed, domain, path[0] if len(path) > 1 else '', path[-1], parsed.query

    def _extract_groups(self, url: str, parts, mode: str) -> Dict[str, Any]:
        """Extract lexical, network and full-mode feature groups safely."""
        _, domain, directory, file_part, params = parts
        lexical = lambda: self._extract_lexical(url, domain, directory, file_part, params, mode)
        features = self._safe_extract("lexical", lexical, mode)
        features.update(self._safe_extract("network", lambda: self._extract_network(url, domain, mode), mode))
        features.update(
            self._safe_extract(
                "domain identity", lambda: self._extract_domain_identity(domain), mode
            )
        )
        if mode == 'full':
            features.update(self._safe_extract(
                "full special", lambda: self._extract_special_full(domain), mode
            ))
        return features

    def extract_offline(self, url: str, mode: str) -> Dict[str, Any]:
        """Extract deterministic lexical features without network lookups."""
        mode = normalize_mode(mode)
        url = self._normalize_url(url)
        parsed, extracted = self._safe_parse(url)

        domain = self._build_domain(extracted)
        directory = parsed.path.rsplit('/', 1)[0] if '/' in parsed.path else ''
        file_part = parsed.path.rsplit('/', 1)[1] if '/' in parsed.path else ''
        params = parsed.query
        features = {}
        features.update(self._extract_lexical(url, domain, directory, file_part, params, mode))
        features.update(self._offline_network_features(domain))
        features.update(self._extract_domain_identity(domain))
        if mode == 'full':
            features.update(self._extract_special_full(domain))
        return self._complete_schema(features, mode)

    def _offline_network_features(self, domain: str) -> Dict[str, Any]:
        """Return network placeholders for deterministic offline extraction."""
        features = {key: MISSING_VALUE for key in NETWORK_FEATURES}
        if self.apply_network_defaults:
            self._apply_defaults(features)
        features['url_shortened'] = self.shortener_detector.detect(domain)
        return features

    def _create_tld_extractor(self):
        cache_dir = self._resolve_cache_dir()
        return tldextract.TLDExtract(
            cache_dir=str(cache_dir),
            fallback_to_snapshot=True,
            suffix_list_urls=(),
        )

    def _resolve_cache_dir(self) -> Path:
        cache_path = Path(os.environ.get('TLD_CACHE_DIR', '/tmp/tldextract'))
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

    def _build_domain(self, extracted) -> str:
        if extracted.suffix:
            return f"{extracted.domain}.{extracted.suffix}"
        return extracted.domain

    def _normalize_url(self, url: str) -> str:
        value = str(url or "").strip()
        if not value:
            return "http://"
        if "://" not in value:
            return f"http://{value}"
        return value

    def _safe_parse(self, url: str):
        try:
            parsed = urlparse(url)
            extracted = self.tld_extractor(url)
            return parsed, extracted
        except Exception as exc:
            logger.warning("URL parse failed for %s: %s", url, exc)
            parsed = urlparse("http://")
            extracted = self.tld_extractor("http://")
            return parsed, extracted

    def _safe_extract(self, name: str, extractor, mode: str) -> Dict[str, Any]:
        try:
            return extractor()
        except Exception as exc:
            logger.warning("%s feature extraction failed: %s", name, exc)
            return {}

    def _complete_schema(self, features: Dict[str, Any], mode: str) -> Dict[str, Any]:
        features = add_missing_indicators(features)
        completed = {
            feature: features.get(feature, MISSING_VALUE)
            for feature in FeatureConfig.get_feature_names(mode)
        }
        return completed

    def _extract_lexical(self, url: str, domain: str, directory: str,
                        file_part: str, params: str, mode: str) -> Dict[str, int]:
        features = {}

        url_chars = FeatureSetConfig.get_url_chars(mode)
        domain_chars = FeatureSetConfig.get_domain_chars(mode)
        params_chars = FeatureSetConfig.get_params_chars(mode)
        file_exclude = FeatureSetConfig.get_file_exclude(mode)

        features.update(self.url_extractor.extract_url_features(url, url_chars))
        features.update(self.url_extractor.extract_domain_features(domain, domain_chars))
        features.update(self.path_extractor.extract_directory_features(directory))
        features.update(self.path_extractor.extract_file_features(file_part, file_exclude))
        features.update(self.params_extractor.extract_features(params, params_chars, mode == 'full'))
        features['email_in_url'] = self.email_detector.detect(url)

        return features

    def _extract_network(self, url: str, domain: str, mode: str) -> Dict[str, Any]:
        if not domain:
            features = {key: MISSING_VALUE for key in NETWORK_FEATURES}
            if self.apply_network_defaults:
                self._apply_defaults(features)
            return features

        domain_features = self._extract_domain_network(domain, mode)
        features = dict(domain_features)

        time_resp, redirects = self._measure_response(url, domain_features)
        features['time_response'] = time_resp
        features['qty_redirects'] = redirects
        if self.apply_network_defaults:
            self._apply_defaults(features)

        return features

    def _measure_response(self, url: str, domain_features: Dict[str, Any]):
        """Measure HTTP response only when DNS resolved an address."""
        if domain_features.get('qty_ip_resolved', -1) > 0:
            return self.response_timer.measure(url)
        return -1, -1

    @lru_cache(maxsize=200000)
    def _extract_domain_network(self, domain: str, mode: str) -> Dict[str, Any]:
        with ThreadPoolExecutor(max_workers=2) as pool:
            dns_future = pool.submit(self.dns_resolver.extract_all, domain)
            features = dict(dns_future.result())
            if WHOIS_ENABLED:
                whois_future = pool.submit(self.whois_extractor.extract_all, domain)
                features.update(whois_future.result())
            else:
                features.update({
                    "time_domain_activation": MISSING_VALUE,
                    "time_domain_expiration": MISSING_VALUE,
                })
        if features.get('qty_ip_resolved', -1) > 0:
            features['tls_ssl_certificate'] = self.ssl_validator.validate(domain)
        else:
            features['tls_ssl_certificate'] = -1
        features['url_shortened'] = self.shortener_detector.detect(domain)

        return features

    def _apply_defaults(self, features: Dict[str, Any]):
        for key, default in self.NETWORK_FEATURE_DEFAULTS.items():
            if key in WHOIS_FEATURES and not WHOIS_ENABLED:
                continue
            if key in features and features[key] == -1:
                features[key] = default

    def _extract_special_full(self, domain: str) -> Dict[str, int]:
        return {
            'server_client_domain': self.keyword_detector.detect(domain)
        }

    def _extract_domain_identity(self, domain: str) -> Dict[str, int]:
        """Extract deterministic domain identity features for every mode."""
        return {
            'domain_in_ip': self.ip_detector.detect(domain)
        }
