"""Canonical feature-mode names and backward-compatible aliases."""

from __future__ import annotations


CANONICAL_MODES = ("full", "optimized", "selected")
LEGACY_MODE_ALIASES = {"optimized_v2": "selected"}
SUPPORTED_MODES = (*CANONICAL_MODES, *LEGACY_MODE_ALIASES)


def normalize_mode(mode: str | None) -> str:
    """Return a canonical mode name while accepting the legacy alias."""
    value = (mode or "full").strip().lower()
    return LEGACY_MODE_ALIASES.get(value, value)


def is_supported_mode(mode: str | None) -> bool:
    """Return whether a mode is canonical or a supported legacy alias."""
    return normalize_mode(mode) in CANONICAL_MODES
