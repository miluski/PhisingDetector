"""Shared URL categories used for benign coverage and threshold calibration."""

from __future__ import annotations

from collections import Counter
from urllib.parse import urlsplit


BENIGN_CATEGORY_PATHS = (
    "about", "contact", "docs", "help", "playlist", "pricing", "status",
    "watch", "products", "login", "account", "secure", "support", "payment",
    "verify",
)
CALIBRATION_CATEGORIES = frozenset(BENIGN_CATEGORY_PATHS)


def classify_benign_category(url: str) -> str | None:
    """Return the first path category used by a URL, or ``None``."""
    parsed = _parse_url(url)
    if parsed is None or parsed.path in {"", "/"}:
        return None
    first_segment = next((part.lower() for part in parsed.path.split("/") if part), "")
    return next(
        (category for category in BENIGN_CATEGORY_PATHS
         if first_segment == category or first_segment.startswith(f"{category}-")),
        None,
    )


def is_calibration_url(url: str) -> bool:
    """Return whether a URL belongs to a calibrated benign path category."""
    return classify_benign_category(url) in CALIBRATION_CATEGORIES


def count_benign_categories(rows: list[dict[str, str]]) -> dict[str, int]:
    """Count categories for rows labelled as benign."""
    counts = Counter()
    for row in rows:
        if str(row.get("label", row.get("phishing", ""))) != "0":
            continue
        category = classify_benign_category(row.get("url", ""))
        if category:
            counts[category] += 1
    return {category: counts.get(category, 0) for category in sorted(BENIGN_CATEGORY_PATHS)}


def _parse_url(url: str):
    """Parse a URL while tolerating malformed source records."""
    candidate = str(url or "")
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    try:
        return urlsplit(candidate)
    except ValueError:
        return None
