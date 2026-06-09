"""Allow running the test suite as a module: python -m test."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

if __name__ == "__main__":
    test_dir = Path(__file__).resolve().parent
    sys.exit(pytest.main([str(test_dir), *sys.argv[1:]]))
