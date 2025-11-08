#!/usr/bin/env python3
"""
Legacy wrapper for diagnostics package
Use: python -m diagnostics [options]
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diagnostics.__main__ import main

if __name__ == "__main__":
    print("⚠️  Note: Use 'python -m diagnostics' instead of this script")
    print("=" * 60)
    main()
