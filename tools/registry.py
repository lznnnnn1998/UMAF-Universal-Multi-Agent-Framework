"""ToolSpec dataclass and ToolRegistry class — tool specifications and role methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolSpec:
    """Structured metadata for a tool available to agents."""
    name: str
    description: str
    parameters: dict[str, str] = field(default_factory=dict)


class ToolRegistry:
    """Centralized registry of all tool specifications and implementations.

    Provides role-specific tool sets and the TOOL_MAP for execution.
    Supports JSON-based tool overrides via set_tool_config() so users can
    restrict or customize which tools each agent role receives per pipeline.
    """

    TOOL_MAP: dict[str, Callable] = {}

    _tool_overrides: dict[str, dict[str, list[str]]] = {}

    _tool_timeouts: dict[str, int] = {}

    _TOOL_NAME_MAP: dict[str, str] = {
        "read_file": "READ_FILE",
        "write_file": "WRITE_FILE",
        "write_lines": "WRITE_LINES",
        "run_command": "RUN_COMMAND",
        "call_claude": "CALL_CLAUDE",
        "web_search": "WEB_SEARCH",
        "web_fetch": "WEB_FETCH",
        "download_file": "DOWNLOAD_FILE",
    }

    @classmethod
    def set_tool_config(cls, config: dict[str, dict[str, list[str]]]) -> None:
        timeout_config = config.pop("__timeouts__", None)
        if isinstance(timeout_config, dict):
            for tool_name, seconds in timeout_config.items():
                if isinstance(seconds, (int, float)) and seconds > 0:
                    cls._tool_timeouts[tool_name] = int(seconds)

        def _is_meta(k: str) -> bool:
            return k.startswith("_") and k != "__global__"

        cls._tool_overrides = {
            pk: {rk: rv for rk, rv in pv.items() if not _is_meta(rk)}
            for pk, pv in config.items() if not _is_meta(pk)
        }

    @classmethod
    def _apply_override(cls, pipeline: str, role: str, defaults: list) -> list:
        overrides = cls._tool_overrides
        if not overrides:
            return defaults

        for key in (pipeline, "__global__"):
            role_map = overrides.get(key, {})
            if not role_map:
                continue
            for rname, tool_names in role_map.items():
                if rname.lower() in role.lower():
                    specs = []
                    for tn in tool_names:
                        attr = cls._TOOL_NAME_MAP.get(tn)
                        if attr:
                            specs.append(getattr(cls, attr))
                    return specs

        return defaults

    READ_FILE = ToolSpec(
        name="read_file",
        description="Read contents of a file at the given path.",
        parameters={"path": "str"},
    )
    WRITE_FILE = ToolSpec(
        name="write_file",
        description="Write content to a file at the given path. Creates parent directories as needed.",
        parameters={"path": "str", "content": "str"},
    )
    WRITE_LINES = ToolSpec(
        name="write_lines",
        description="Write a list of lines to a file. Preferred over write_file for large code files — each line is a separate string in a JSON array, which avoids escaping issues with multi-line strings.",
        parameters={"path": "str", "lines": "list[str]"},
    )
    RUN_COMMAND = ToolSpec(
        name="run_command",
        description="Run a shell command and return its output (stdout + stderr). Timeout: 30s.",
        parameters={"command": "str"},
    )
    CALL_CLAUDE = ToolSpec(
        name="call_claude",
        description="Call the Claude Code CLI to perform a complex reasoning subtask.",
        parameters={"prompt": "str"},
    )
    WEB_SEARCH = ToolSpec(
        name="web_search",
        description="Search the web using DuckDuckGo and return results with titles, URLs, and snippets.",
        parameters={"query": "str", "max_results": "int (optional, default 10, max 20)"},
    )
    WEB_FETCH = ToolSpec(
        name="web_fetch",
        description="Fetch content from a URL and return as plain text. Use for arxiv.org, articles, documentation.",
        parameters={"url": "str", "max_chars": "int (optional, default 12000, max 20000)"},
    )
    DOWNLOAD_FILE = ToolSpec(
        name="download_file",
        description="Download a URL to a local file via urllib. Bypasses Claude Code network sandbox.",
        parameters={"url": "str", "output_path": "str"},
    )

    @classmethod
    def coder_tools(cls) -> list[ToolSpec]:
        return cls._apply_override("coder", "coder", [])

    @classmethod
    def reviewer_tools(cls) -> list[ToolSpec]:
        return cls._apply_override("coder", "reviewer", [])

    @classmethod
    def research_decomposer_tools(cls, backend: str = "deepseek") -> list[ToolSpec]:
        return cls._apply_override("research", "decomposer", [])

    @classmethod
    def research_worker_tools(cls) -> list[ToolSpec]:
        return cls._apply_override("research", "worker", [])

    @classmethod
    def research_reviewer_tools(cls) -> list[ToolSpec]:
        return cls._apply_override("research", "reviewer", [])

    @classmethod
    def writer_tools(cls) -> list[ToolSpec]:
        return cls._apply_override("research", "writer", [])

    @classmethod
    def coderpp_decomposer_tools(cls, backend: str = "deepseek") -> list[ToolSpec]:
        return cls._apply_override("coderpp", "decomposer", [])

    @classmethod
    def coderpp_worker_tools(cls) -> list[ToolSpec]:
        return cls._apply_override("coderpp", "worker", [])

    @classmethod
    def coderpp_reviewer_tools(cls) -> list[ToolSpec]:
        return cls._apply_override("coderpp", "reviewer", [])

    @classmethod
    def organizer_tools(cls) -> list[ToolSpec]:
        return cls._apply_override("coderpp", "organizer", [])

    @classmethod
    def topology_analyzer_tools(cls) -> list[ToolSpec]:
        return cls._apply_override("topology", "analyzer", [])

    @classmethod
    def topology_designer_tools(cls) -> list[ToolSpec]:
        return cls._apply_override("topology", "designer", [])

    @classmethod
    def topology_evaluator_tools(cls) -> list[ToolSpec]:
        return cls._apply_override("topology", "evaluator", [])

    @classmethod
    def topology_writer_tools(cls) -> list[ToolSpec]:
        return cls._apply_override("topology", "writer", [])

    @classmethod
    def skill_scanner_tools(cls) -> list[ToolSpec]:
        return cls._apply_override("skill", "scanner", [])

    @classmethod
    def skill_detector_tools(cls) -> list[ToolSpec]:
        return cls._apply_override("skill", "detector", [])

    @classmethod
    def skill_aggregator_tools(cls) -> list[ToolSpec]:
        return cls._apply_override("skill", "aggregator", [])

    @classmethod
    def skill_writer_tools(cls) -> list[ToolSpec]:
        return cls._apply_override("skill", "writer", [])

    @classmethod
    def self_evolution_analyzer_tools(cls) -> list[ToolSpec]:
        return cls._apply_override("self_evolution", "analyzer", [])

    @classmethod
    def self_evolution_planner_tools(cls) -> list[ToolSpec]:
        return cls._apply_override("self_evolution", "planner", [])

    @classmethod
    def self_evolution_coder_tools(cls) -> list[ToolSpec]:
        return cls._apply_override("self_evolution", "coder", [])

    @classmethod
    def self_evolution_reviewer_tools(cls) -> list[ToolSpec]:
        return cls._apply_override("self_evolution", "reviewer", [])

    @classmethod
    def self_evolution_writer_tools(cls) -> list[ToolSpec]:
        return cls._apply_override("self_evolution", "writer", [])

    @classmethod
    def to_dicts(cls, specs: list[ToolSpec]) -> list[dict[str, Any]]:
        return [{"name": s.name, "description": s.description, "parameters": dict(s.parameters)} for s in specs]

    @classmethod
    def get_map(cls) -> dict[str, Callable]:
        return dict(cls.TOOL_MAP)
