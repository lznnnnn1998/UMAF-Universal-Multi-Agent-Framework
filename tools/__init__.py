"""UMAF Tools — tool specifications, implementations, and role-specific tool sets."""

from .registry import ToolSpec, ToolRegistry
from .functions import (
    TOOL_MAP,
    read_file,
    write_file,
    write_lines,
    run_command,
    call_claude,
    web_search,
    web_fetch,
    download_file,
)
from .feature_tools import apply_to_tool_registry

# Auto-apply feature tool methods to ToolRegistry on first import
apply_to_tool_registry(ToolRegistry)
