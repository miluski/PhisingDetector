"""Make backend modules importable when pytest starts at repository root."""

from __future__ import annotations

import os
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/phishing-detector-test-cache")
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
