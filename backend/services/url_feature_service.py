import os
import logging
from pathlib import Path
from typing import Dict, Any
from urllib.parse import urlparse
import tldextract
from config.feature_config import FeatureSetConfig
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
from extractors.content.html_content_extractor import HTMLContentExtractor

logger = logging.getLogger("URLFeatureService")


class URLFeatureService:

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
        'asn_ip': 22612,
    }

    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self.tld_extractor = self._create_tld_extractor()
        self._init_extractors()
        self._init_detectors()

    def _init_extractors(self):
        self.url_extractor = URLPartExtractor()
        self.path_extractor = PathPartExtractor()
        self.params_extractor = ParamsExtractor()
        self.response_timer = ResponseTimer(self.timeout)
        self.dns_resolver = DNSResolver()
        self.whois_extractor = WHOISExtractor()
        self.ssl_validator = SSLCertificateValidator(self.timeout)
        self.html_extractor = HTMLContentExtractor(self.timeout)

    def _init_detectors(self):
        self.email_detector = EmailDetector()
        self.ip_detector = IPAddressDetector()
        self.keyword_detector = KeywordDetector()
        self.shortener_detector = URLShortenerDetector()

    def extract(self, url: str, mode: str) -> Dict[str, Any]:
        logger.info(f"   🔍 Extracting features for: {url}")
        logger.debug(f"      Mode: {mode}")

        parsed = urlparse(url)
        extracted = self.tld_extractor(url)

        domain = self._build_domain(extracted)
        directory = parsed.path.rsplit('/', 1)[0] if '/' in parsed.path else ''
        file_part = parsed.path.rsplit('/', 1)[1] if '/' in parsed.path else ''
        params = parsed.query

        logger.debug(f"      Domain: {domain}")
        logger.debug(f"      Directory: {directory}")
        logger.debug(f"      File: {file_part}")
        logger.debug(f"      Params: {params[:50]}..." if len(params) > 50 else f"      Params: {params}")

        features = {}

        logger.debug("      Extracting lexical features...")
        features.update(self._extract_lexical(url, domain, directory, file_part, params, mode))

        logger.debug("      Extracting network features...")
        features.update(self._extract_network(url, domain, mode))

        if mode == 'full':
            logger.debug("      Extracting full mode features (special + HTML)...")
            features.update(self._extract_special_full(domain))
            features.update(self.html_extractor.extract(url))

        logger.info(f"   ✅ Extracted {len(features)} features")
        return features

    def _create_tld_extractor(self):
        cache_dir = self._resolve_cache_dir()
        return tldextract.TLDExtract(cache_dir=str(cache_dir), fallback_to_snapshot=True)

    def _resolve_cache_dir(self) -> Path:
        cache_path = Path(os.environ.get('TLD_CACHE_DIR', '/tmp/tldextract'))
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

    def _build_domain(self, extracted) -> str:
        if extracted.suffix:
            return f"{extracted.domain}.{extracted.suffix}"
        return extracted.domain

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
        features = {}

        time_resp, redirects = self.response_timer.measure(url)
        features['time_response'] = time_resp
        features['qty_redirects'] = redirects

        features.update(self.dns_resolver.extract_all(domain))
        features.update(self.whois_extractor.extract_all(domain))
        features['tls_ssl_certificate'] = self.ssl_validator.validate(domain)
        features['url_shortened'] = self.shortener_detector.detect(domain)
        features['asn_ip'] = self.NETWORK_FEATURE_DEFAULTS['asn_ip']

        if mode == 'full':
            features['url_google_index'] = 0
            features['domain_google_index'] = 0

        self._apply_defaults(features)

        return features

    def _apply_defaults(self, features: Dict[str, Any]):
        for key, default in self.NETWORK_FEATURE_DEFAULTS.items():
            if key in features and features[key] == -1:
                features[key] = default

    def _extract_special_full(self, domain: str) -> Dict[str, int]:
        return {
            'domain_in_ip': self.ip_detector.detect(domain),
            'server_client_domain': self.keyword_detector.detect(domain)
        }
