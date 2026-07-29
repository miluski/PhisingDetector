"""Read reusable e-mail and SMS examples from the test data pool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MessageExampleService:
    """Provide message examples stored outside application source code."""

    def __init__(self, source_path: Path | None = None) -> None:
        """Create a reader for the shared sample message JSON file."""
        self.source_path = source_path or (
            Path(__file__).resolve().parents[1] / "data" / "test" / "sample_messages.json"
        )

    def get_examples(self) -> dict[str, list[dict[str, Any]]]:
        """Return normalized e-mail and SMS records from the configured pool."""
        payload = self._read_payload()
        emails = self._combine(payload, "phishing_emails", "legitimate_emails")
        sms = self._combine(payload, "phishing_sms", "legitimate_sms")
        return {
            "emails": [self._email_record(item) for item in emails],
            "sms": [self._sms_record(item) for item in sms],
        }

    def _read_payload(self) -> dict[str, Any]:
        """Load and decode the JSON pool, raising a controlled configuration error."""
        try:
            with self.source_path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Nie można wczytać puli przykładowych wiadomości") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Pula przykładowych wiadomości ma nieprawidłowy format")
        return payload

    @staticmethod
    def _combine(payload: dict[str, Any], *groups: str) -> list[dict[str, Any]]:
        """Combine phishing and safe groups while preserving their source records."""
        examples: list[dict[str, Any]] = []
        for group in groups:
            records = payload.get(group, [])
            if isinstance(records, list):
                examples.extend(item for item in records if isinstance(item, dict))
        return examples

    @staticmethod
    def _email_record(record: dict[str, Any]) -> dict[str, Any]:
        """Expose only fields required to populate the e-mail form."""
        return {
            **{key: record.get(key, "") for key in ("sender", "recipient", "subject", "body")},
            "contains_phishing_url": MessageExampleService._is_phishing(record),
        }

    @staticmethod
    def _sms_record(record: dict[str, Any]) -> dict[str, Any]:
        """Expose only the message body required to populate the SMS form."""
        return {
            "body": record.get("body", ""),
            "contains_phishing_url": MessageExampleService._is_phishing(record),
        }

    @staticmethod
    def _is_phishing(record: dict[str, Any]) -> bool:
        """Translate the source label into a safe boolean for the test form."""
        return record.get("expected_result") in {"phishing", "suspicious"}


def serialize_message_examples(service: MessageExampleService) -> dict[str, Any]:
    """Build the stable JSON response consumed by the frontend."""
    return {"success": True, **service.get_examples()}
