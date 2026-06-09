"""Shared fixtures and hooks for the UMAF test suite.

This conftest is loaded automatically by pytest before test collection.
It ensures tools_config.json is loaded exactly once and provides shared
fixtures used across all test modules.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tools import ToolRegistry

# Load tools_config.json at module-import time (idempotent - overwriting is cheap).
# This covers both pytest runs (conftest is imported before test modules) and
# direct script runs (python test/test_*.py) when the test file imports
# conftest at module level.
_config_path = Path(__file__).resolve().parent.parent / "tools_config.json"
if _config_path.exists():
    with open(_config_path) as f:
        ToolRegistry.set_tool_config(json.load(f))


@pytest.fixture
def tmpdir() -> str:
    """Temporary directory cleaned up after each test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


def make_agent_result(
    messages: list[dict[str, str]], success: bool = True
) -> MagicMock:
    """Build a mock AgentResult whose .messages are MagicMock AIMessages."""
    mock_msgs: list[MagicMock] = []
    for m in messages:
        mm = MagicMock()
        mm.content = m.get("content", "")
        type(mm).__name__ = m.get("type", "AIMessage")
        mock_msgs.append(mm)
    result = MagicMock()
    result.messages = mock_msgs
    result.success = success
    return result
