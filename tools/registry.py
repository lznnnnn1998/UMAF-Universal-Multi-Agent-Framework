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

    # --- Tool implementations (same functions, accessible as class attribute) ---
    TOOL_MAP: dict[str, Callable] = {}  # populated after function definitions

    # Module-level override: set by set_tool_config() before pipeline runs.
    # Nested dict: {pipeline_name: {role_name: [tool_name_str, ...]}}
    _tool_overrides: dict[str, dict[str, list[str]]] = {}

    # Per-tool timeout overrides in seconds. Keys are tool function names.
    # Populated from the __timeouts__ key in the tools config JSON::
    #     {"__timeouts__": {"call_claude": 300, "web_fetch": 30}}
    _tool_timeouts: dict[str, int] = {}

    # Mapping from human-readable JSON names to ToolSpec class attributes.
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
        """Apply a tool configuration loaded from a JSON file.

        The config dict maps pipeline names to role→tool-list dicts.
        An optional ``__timeouts__`` key sets per-tool timeouts in seconds::

            {
                "__timeouts__": {"call_claude": 300, "web_fetch": 30},
                "research": {
                    "decomposer": ["read_file", "run_command"],
                    "worker": ["read_file", "write_file", "call_claude"],
                    "reviewer": ["read_file", "write_file"],
                    "writer": ["read_file", "write_file", "write_lines"]
                }
            }

        Role names are matched case-insensitively against the classmethod
        suffixes (e.g. "worker" matches ``research_worker_tools``,
        ``coderpp_worker_tools``, and ``writer`` matches ``writer_tools``).
        """
        timeout_config = config.pop("__timeouts__", None)
        if isinstance(timeout_config, dict):
            for tool_name, seconds in timeout_config.items():
                if isinstance(seconds, (int, float)) and seconds > 0:
                    cls._tool_timeouts[tool_name] = int(seconds)
        cls._tool_overrides = dict(config)

    @classmethod
    def _apply_override(cls, pipeline: str, role: str, defaults: list) -> list:
        """Return the override tool list for *role* in *pipeline*, or *defaults*."""
        overrides = cls._tool_overrides
        if not overrides:
            return defaults

        # Check pipeline-specific override first, then global
        for key in (pipeline, "__global__"):
            role_map = overrides.get(key, {})
            if not role_map:
                continue
            # Case-insensitive role match
            for rname, tool_names in role_map.items():
                if rname.lower() in role.lower():
                    specs = []
                    for tn in tool_names:
                        attr = cls._TOOL_NAME_MAP.get(tn)
                        if attr:
                            specs.append(getattr(cls, attr))
                    return specs

        return defaults

    # --- Individual tool specs ---
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

    # --- Role-specific tool lists ---

    @classmethod
    def coder_tools(cls) -> list[ToolSpec]:
        defaults = [cls.READ_FILE, cls.WRITE_FILE, cls.RUN_COMMAND, cls.CALL_CLAUDE, cls.WEB_SEARCH, cls.WEB_FETCH]
        return cls._apply_override("coder", "coder", defaults)

    @classmethod
    def reviewer_tools(cls) -> list[ToolSpec]:
        defaults = [cls.READ_FILE, cls.RUN_COMMAND, cls.CALL_CLAUDE, cls.WEB_SEARCH, cls.WEB_FETCH]
        return cls._apply_override("coder", "reviewer", defaults)

    @classmethod
    def research_decomposer_tools(cls, backend: str = "deepseek") -> list[ToolSpec]:
        defaults: list[ToolSpec]
        if backend == "claude_cli":
            defaults = [cls.READ_FILE]
        else:
            defaults = [cls.RUN_COMMAND, cls.CALL_CLAUDE, cls.WEB_SEARCH]
        return cls._apply_override("research", "decomposer", defaults)

    @classmethod
    def research_worker_tools(cls) -> list[ToolSpec]:
        defaults = [cls.READ_FILE, cls.WRITE_FILE, cls.RUN_COMMAND, cls.CALL_CLAUDE, cls.WEB_SEARCH, cls.WEB_FETCH, cls.DOWNLOAD_FILE]
        return cls._apply_override("research", "worker", defaults)

    @classmethod
    def research_reviewer_tools(cls) -> list[ToolSpec]:
        defaults = [cls.READ_FILE, cls.WRITE_FILE, cls.WEB_SEARCH, cls.WEB_FETCH]
        return cls._apply_override("research", "reviewer", defaults)

    @classmethod
    def writer_tools(cls) -> list[ToolSpec]:
        defaults = [cls.READ_FILE, cls.WRITE_FILE, cls.WRITE_LINES]
        return cls._apply_override("research", "writer", defaults)

    @classmethod
    def coderpp_decomposer_tools(cls, backend: str = "deepseek") -> list[ToolSpec]:
        defaults = [cls.READ_FILE, cls.WRITE_FILE, cls.RUN_COMMAND]
        return cls._apply_override("coderpp", "decomposer", defaults)

    @classmethod
    def coderpp_worker_tools(cls) -> list[ToolSpec]:
        defaults = [cls.READ_FILE, cls.WRITE_FILE, cls.WRITE_LINES, cls.RUN_COMMAND]
        return cls._apply_override("coderpp", "worker", defaults)

    @classmethod
    def coderpp_reviewer_tools(cls) -> list[ToolSpec]:
        defaults = [cls.READ_FILE, cls.WRITE_FILE, cls.WRITE_LINES, cls.RUN_COMMAND]
        return cls._apply_override("coderpp", "reviewer", defaults)

    @classmethod
    def organizer_tools(cls) -> list[ToolSpec]:
        defaults = [cls.READ_FILE, cls.WRITE_FILE, cls.WRITE_LINES, cls.RUN_COMMAND]
        return cls._apply_override("coderpp", "organizer", defaults)

    @classmethod
    def topology_analyzer_tools(cls) -> list[ToolSpec]:
        """Tools for TopologyAnalyzerRole: reads input spec, writes analysis."""
        defaults = [cls.READ_FILE, cls.WRITE_FILE]
        return cls._apply_override("topology", "analyzer", defaults)

    @classmethod
    def topology_designer_tools(cls) -> list[ToolSpec]:
        """Tools for TopologyDesignerRole: reads complexity factors, writes candidates."""
        defaults = [cls.READ_FILE, cls.WRITE_FILE]
        return cls._apply_override("topology", "designer", defaults)

    @classmethod
    def topology_evaluator_tools(cls) -> list[ToolSpec]:
        """Tools for TopologyEvaluatorRole: reads candidate topologies, writes scores."""
        defaults = [cls.READ_FILE, cls.WRITE_FILE]
        return cls._apply_override("topology", "evaluator", defaults)

    @classmethod
    def topology_writer_tools(cls) -> list[ToolSpec]:
        """Tools for TopologyWriterRole: writes final spec and report files."""
        defaults = [cls.WRITE_FILE]
        return cls._apply_override("topology", "writer", defaults)

    @classmethod
    def skill_scanner_tools(cls) -> list[ToolSpec]:
        """Tools for SkillScannerRole: scans project directory."""
        defaults = [cls.READ_FILE, cls.WRITE_FILE, cls.RUN_COMMAND]
        return cls._apply_override("skill", "scanner", defaults)

    @classmethod
    def skill_detector_tools(cls) -> list[ToolSpec]:
        """Tools for domain detector roles: reads project_scan.json, writes report."""
        defaults = [cls.READ_FILE, cls.WRITE_FILE, cls.RUN_COMMAND]
        return cls._apply_override("skill", "detector", defaults)

    @classmethod
    def skill_aggregator_tools(cls) -> list[ToolSpec]:
        """Tools for SkillAggregatorRole: reads domain reports, deduplicates, writes inventory."""
        defaults = [cls.READ_FILE, cls.WRITE_FILE]
        return cls._apply_override("skill", "aggregator", defaults)

    @classmethod
    def skill_writer_tools(cls) -> list[ToolSpec]:
        """Tools for SkillReportWriterRole: writes skills.json and skills_report.md."""
        defaults = [cls.READ_FILE, cls.WRITE_FILE]
        return cls._apply_override("skill", "writer", defaults)

    @classmethod
    def to_dicts(cls, specs: list[ToolSpec]) -> list[dict[str, Any]]:
        """Convert ToolSpec list to the dict format expected by BaseAgent."""
        return [{"name": s.name, "description": s.description, "parameters": dict(s.parameters)} for s in specs]

    @classmethod
    def get_map(cls) -> dict[str, Callable]:
        return dict(cls.TOOL_MAP)
