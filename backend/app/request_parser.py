"""Shared normalization of email and SMS API requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from config.feature_modes import normalize_mode


@dataclass(frozen=True)
class AnalysisRequest:
    """Normalized request passed from an API resource to the service layer."""

    content: str
    models: list[str]
    mode: str
    dataset: str
    sender: str = ""
    receiver: str = ""
    attachments: list[dict[str, Any]] | None = None


class AnalysisRequestParser:
    """Parse shared fields while preserving current endpoint defaults."""

    DEFAULT_MODEL = "random_forest"
    DEFAULT_MODE = "optimized"
    DEFAULT_DATASET = "current"

    def parse_sms(self, args: Any) -> AnalysisRequest:
        """Build a normalized SMS request."""
        content = self.clean(args.get("content"))
        if not content:
            raise ValueError("Missing required field: content")
        return AnalysisRequest(content, self.models(args.get("models")),
                               self.mode(args.get("mode")), self.dataset(args.get("dataset")))

    def parse_email(self, args: Any, files: Iterable[Any] | None) -> AnalysisRequest:
        """Build a normalized email request including uploaded attachments."""
        content = self.clean(args.get("content"))
        if not content:
            raise ValueError("Missing request body")
        return AnalysisRequest(content, self.models(args.get("models")),
                               self.mode(args.get("mode")), self.dataset(args.get("dataset")),
                               self.clean(args.get("sender")), self.clean(args.get("receiver")),
                               self.attachments(files))

    @staticmethod
    def clean(value: Any) -> str:
        """Convert an optional form value to trimmed text."""
        return str(value or "").strip()

    def models(self, raw: Any) -> list[str]:
        """Normalize unique model names and apply the default model."""
        items = raw if isinstance(raw, list) else [raw]
        values = list(dict.fromkeys(self.clean(item) for item in items if self.clean(item)))
        return values or [self.DEFAULT_MODEL]

    def mode(self, raw: Any) -> str:
        """Normalize a mode and apply the optimized default."""
        return normalize_mode(self.clean(raw) or self.DEFAULT_MODE)

    def dataset(self, raw: Any) -> str:
        """Normalize a dataset and apply the current-data default."""
        return self.clean(raw).lower() or self.DEFAULT_DATASET

    @staticmethod
    def attachments(files: Iterable[Any] | None) -> list[dict[str, Any]] | None:
        """Read uploaded files into the service attachment contract."""
        if not files:
            return None
        result = [{"filename": item.filename, "content": item.read()} for item in files if item]
        return result or None
