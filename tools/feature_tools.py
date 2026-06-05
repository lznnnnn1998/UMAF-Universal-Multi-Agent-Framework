"""Feature Pipeline tool methods for ToolRegistry.

Defines 5 feature_*_tools() classmethods that mirror the existing pattern
of topology_*_tools() and skill_*_tools(). Patched onto ToolRegistry at
import time by tools/__init__.py.

Role → Tool summary
===================
================================ =============================================
Role                              Default tools
================================ =============================================
``feature_scanner_tools``         Read + Write + RunCommand
``feature_planner_tools``         Read + Write
``feature_coder_tools``           Read + Write + WriteLines + RunCommand
``feature_reviewer_tools``        Read + RunCommand
``feature_writer_tools``          Write (write-only)
================================ =============================================
"""

from __future__ import annotations

from typing import Any


def apply_to_tool_registry(tool_registry: type) -> type:
    """Add feature_*_tools() classmethods to a ToolRegistry class.

    After calling, the following methods are available::

        ToolRegistry.feature_scanner_tools()
        ToolRegistry.feature_planner_tools()
        ToolRegistry.feature_coder_tools()
        ToolRegistry.feature_reviewer_tools()
        ToolRegistry.feature_writer_tools()

    Args:
        tool_registry: The ToolRegistry class to patch.

    Returns:
        The patched class (same object as *tool_registry*).
    """
    cls = tool_registry

    @classmethod
    def feature_scanner_tools(cls) -> list:
        defaults = [cls.READ_FILE, cls.WRITE_FILE, cls.RUN_COMMAND]
        return cls._apply_override("feature", "scanner", defaults)

    @classmethod
    def feature_planner_tools(cls) -> list:
        defaults = [cls.READ_FILE, cls.WRITE_FILE]
        return cls._apply_override("feature", "planner", defaults)

    @classmethod
    def feature_coder_tools(cls) -> list:
        defaults = [cls.READ_FILE, cls.WRITE_FILE, cls.WRITE_LINES, cls.RUN_COMMAND]
        return cls._apply_override("feature", "coder", defaults)

    @classmethod
    def feature_reviewer_tools(cls) -> list:
        defaults = [cls.READ_FILE, cls.RUN_COMMAND]
        return cls._apply_override("feature", "reviewer", defaults)

    @classmethod
    def feature_writer_tools(cls) -> list:
        defaults = [cls.WRITE_FILE]
        return cls._apply_override("feature", "writer", defaults)

    cls.feature_scanner_tools = feature_scanner_tools
    cls.feature_planner_tools = feature_planner_tools
    cls.feature_coder_tools = feature_coder_tools
    cls.feature_reviewer_tools = feature_reviewer_tools
    cls.feature_writer_tools = feature_writer_tools

    return cls
