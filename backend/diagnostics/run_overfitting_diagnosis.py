#!/usr/bin/env python3
"""
Legacy wrapper for quick overfitting diagnosis
Use: python -m diagnostics --quick
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diagnostics.__main__ import quick_diagnosis

if __name__ == "__main__":
    print("⚠️  Note: Use 'python -m diagnostics --quick' instead of this script")
    print("=" * 60)
    quick_diagnosis()
