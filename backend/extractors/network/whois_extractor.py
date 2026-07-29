"""Domain registration dates via RDAP with legacy WHOIS fallback."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict

import requests
import whois


logger = logging.getLogger("WHOISExtractor")


class WHOISExtractor:
    """Read registration dates without depending exclusively on port 43.

    RDAP uses HTTPS and structured JSON. Legacy WHOIS remains fallback for
    registries not covered by RDAP or transient RDAP failures.
    """

    RDAP_URL = "https://rdap.org/domain/"

    def __init__(self, timeout: int = 5, session: requests.Session | None = None):
        """Initialize RDAP/WHOIS clients with a bounded timeout."""
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update({"Accept": "application/rdap+json, application/json"})

    @staticmethod
    def _parse_date(value: Any) -> datetime | None:
        if not value:
            return None
        candidate = value[0] if isinstance(value, list) else value
        if isinstance(candidate, datetime):
            parsed = candidate
        else:
            text = str(candidate).strip()
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                parsed = datetime.fromisoformat(text)
            except ValueError:
                return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @classmethod
    def parse_rdap_dates(
        cls, payload: dict[str, Any], now: datetime | None = None
    ) -> Dict[str, int]:
        """Convert RDAP registration events into project day features."""
        if not isinstance(payload, dict):
            return cls._empty_dates()
        dates = cls._event_dates(payload.get("events") or [])
        reference = now or datetime.now(timezone.utc)
        activation, expiration = dates.get("activation"), dates.get("expiration")
        return {"time_domain_activation": (reference - activation).days if activation else -1,
                "time_domain_expiration": (expiration - reference).days if expiration else -1}

    @staticmethod
    def _empty_dates() -> Dict[str, int]:
        """Return the missing-value registration feature pair."""
        return {"time_domain_activation": -1, "time_domain_expiration": -1}

    @classmethod
    def _event_dates(cls, events):
        """Extract first registration and expiration dates from RDAP events."""
        dates: dict[str, datetime] = {}
        for event in events:
            if not isinstance(event, dict):
                continue
            action = str(event.get("eventAction") or "").lower()
            date = cls._parse_date(event.get("eventDate"))
            if date is None:
                continue
            if action in {"registration", "registered", "creation"}:
                dates.setdefault("activation", date)
            elif action in {"expiration", "expiry", "expiration date"}:
                dates.setdefault("expiration", date)
        return dates

    def _extract_rdap(self, domain: str) -> Dict[str, int] | None:
        try:
            response = self.session.get(
                f"{self.RDAP_URL}{domain}", timeout=self.timeout
            )
            response.raise_for_status()
            values = self.parse_rdap_dates(response.json())
            if any(value != -1 for value in values.values()):
                return values
        except (requests.RequestException, AttributeError, ValueError, TypeError) as exc:
            logger.debug("RDAP lookup failed for %s: %s", domain, exc)
        return None

    @staticmethod
    def _first_date(value: Any) -> Any:
        return value[0] if isinstance(value, list) and value else value

    def _extract_legacy_whois(self, domain: str) -> Dict[str, int]:
        try:
            record = whois.whois(domain)
            creation = self._parse_date(self._first_date(record.creation_date))
            expiration = self._parse_date(self._first_date(record.expiration_date))
            now = datetime.now(timezone.utc)
            return {
                "time_domain_activation": (now - creation).days if creation else -1,
                "time_domain_expiration": (expiration - now).days if expiration else -1,
            }
        except Exception as exc:
            logger.debug("Legacy WHOIS lookup failed for %s: %s", domain, exc)
            return {
                "time_domain_activation": -1,
                "time_domain_expiration": -1,
            }

    def extract_all(self, domain: str) -> Dict[str, int]:
        """Return registration-day features using RDAP then WHOIS fallback."""
        rdap_values = self._extract_rdap(domain)
        return rdap_values if rdap_values is not None else self._extract_legacy_whois(domain)
