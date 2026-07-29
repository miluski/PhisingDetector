"""Offline tests for RDAP date parsing and legacy fallback semantics."""

from datetime import datetime, timezone

from extractors.network.whois_extractor import WHOISExtractor


class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "events": [
                {"eventAction": "registration", "eventDate": "2020-01-01T00:00:00Z"}
            ]
        }


class _FakeSession:
    def __init__(self):
        self.headers = {}

    def get(self, url, timeout):
        assert url.endswith("example.com")
        assert timeout == 3
        return _FakeResponse()


def test_parse_rdap_dates_converts_events_to_days():
    reference = datetime(2026, 7, 17, tzinfo=timezone.utc)
    payload = {
        "events": [
            {"eventAction": "registration", "eventDate": "2020-01-01T00:00:00Z"},
            {"eventAction": "expiration", "eventDate": "2027-01-01T00:00:00Z"},
        ]
    }

    values = WHOISExtractor.parse_rdap_dates(payload, now=reference)

    assert values["time_domain_activation"] == 2389
    assert values["time_domain_expiration"] == 168


def test_parse_rdap_dates_keeps_missing_values_explicit():
    values = WHOISExtractor.parse_rdap_dates({"events": []})

    assert values == {
        "time_domain_activation": -1,
        "time_domain_expiration": -1,
    }


def test_extract_all_uses_rdap_before_legacy_whois():
    extractor = WHOISExtractor(timeout=3, session=_FakeSession())

    values = extractor.extract_all("example.com")

    assert values["time_domain_activation"] > 0
    assert values["time_domain_expiration"] == -1
