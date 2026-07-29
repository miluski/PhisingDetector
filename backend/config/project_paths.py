"""Canonical filesystem locations shared by backend tools and services."""

from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
