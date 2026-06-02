import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from claude_config import merge_claude_env


@dataclass
class ToolSpec:
    """Structured metadata for a tool available to agents."""
    name: str
    description: str
    parameters: dict[str, str] = field(default_factory=dict)


class ToolRegistry:
    """Centralized registry of all tool specifications and implementations.

    Provides role-specific tool sets and the TOOL_MAP for execution.
    """

    # --- Tool implementations (same functions, accessible as class attribute) ---
    TOOL_MAP: dict[str, Callable] = {}  # populated after function definitions

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
        return [cls.READ_FILE, cls.WRITE_FILE, cls.RUN_COMMAND, cls.CALL_CLAUDE, cls.WEB_SEARCH, cls.WEB_FETCH]

    @classmethod
    def reviewer_tools(cls) -> list[ToolSpec]:
        return [cls.READ_FILE, cls.RUN_COMMAND, cls.CALL_CLAUDE, cls.WEB_SEARCH, cls.WEB_FETCH]

    @classmethod
    def research_decomposer_tools(cls, backend: str = "deepseek") -> list[ToolSpec]:
        if backend == "claude_cli":
            return [cls.READ_FILE]
        return [cls.RUN_COMMAND, cls.CALL_CLAUDE, cls.WEB_SEARCH]

    @classmethod
    def research_worker_tools(cls) -> list[ToolSpec]:
        return [cls.READ_FILE, cls.WRITE_FILE, cls.RUN_COMMAND, cls.CALL_CLAUDE, cls.WEB_SEARCH, cls.WEB_FETCH, cls.DOWNLOAD_FILE]

    @classmethod
    def research_reviewer_tools(cls) -> list[ToolSpec]:
        return [cls.READ_FILE, cls.WRITE_FILE, cls.WEB_SEARCH, cls.WEB_FETCH]

    @classmethod
    def writer_tools(cls) -> list[ToolSpec]:
        return [cls.READ_FILE, cls.WRITE_FILE]

    @classmethod
    def coderpp_decomposer_tools(cls, backend: str = "deepseek") -> list[ToolSpec]:
        return [cls.READ_FILE, cls.WRITE_FILE, cls.RUN_COMMAND]

    @classmethod
    def coderpp_worker_tools(cls) -> list[ToolSpec]:
        return [cls.READ_FILE, cls.WRITE_FILE, cls.WRITE_LINES, cls.RUN_COMMAND]

    @classmethod
    def coderpp_reviewer_tools(cls) -> list[ToolSpec]:
        return [cls.READ_FILE, cls.WRITE_FILE, cls.WRITE_LINES, cls.RUN_COMMAND]

    @classmethod
    def organizer_tools(cls) -> list[ToolSpec]:
        return [cls.READ_FILE, cls.WRITE_FILE, cls.WRITE_LINES, cls.RUN_COMMAND]

    @classmethod
    def topology_analyzer_tools(cls) -> list[ToolSpec]:
        """Tools for TopologyAnalyzerRole: reads input spec, writes analysis."""
        return [cls.READ_FILE, cls.WRITE_FILE]

    @classmethod
    def topology_designer_tools(cls) -> list[ToolSpec]:
        """Tools for TopologyDesignerRole: reads complexity factors, writes candidates."""
        return [cls.READ_FILE, cls.WRITE_FILE]

    @classmethod
    def topology_evaluator_tools(cls) -> list[ToolSpec]:
        """Tools for TopologyEvaluatorRole: reads candidate topologies, writes scores."""
        return [cls.READ_FILE, cls.WRITE_FILE]

    @classmethod
    def topology_writer_tools(cls) -> list[ToolSpec]:
        """Tools for TopologyWriterRole: writes final spec and report files."""
        return [cls.WRITE_FILE]

    @classmethod
    def to_dicts(cls, specs: list[ToolSpec]) -> list[dict[str, Any]]:
        """Convert ToolSpec list to the dict format expected by BaseAgent."""
        return [{"name": s.name, "description": s.description, "parameters": dict(s.parameters)} for s in specs]

    @classmethod
    def get_map(cls) -> dict[str, Callable]:
        return dict(cls.TOOL_MAP)


def _resolve_path(path: str, working_dir: str) -> Path:
    """Resolve path relative to working_dir, stripping accidental prepend."""
    wd_name = Path(working_dir).name
    parts = Path(path).parts
    if parts and parts[0] == wd_name:
        path = str(Path(*parts[1:]))
    return Path(working_dir) / path


def read_file(path: str, working_dir: str = ".") -> str:
    """Read contents of a file at path, resolved relative to working_dir."""
    full_path = _resolve_path(path, working_dir)
    try:
        return full_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"Error: file not found: {full_path}"
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(path: str, content: str, working_dir: str = ".") -> str:
    """Write content to a file at path, resolved relative to working_dir."""
    full_path = _resolve_path(path, working_dir)
    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        return f"Successfully wrote to {full_path}"
    except Exception as e:
        return f"Error writing file: {e}"


def write_lines(path: str, lines: list[str], working_dir: str = ".") -> str:
    """Write a list of lines to a file, resolved relative to working_dir.

    Preferred over write_file for large code — JSON arrays of strings are
    easier for LLMs to emit correctly than multi-line strings with escaping.
    """
    full_path = _resolve_path(path, working_dir)
    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text("\n".join(lines), encoding="utf-8")
        return f"Successfully wrote {len(lines)} lines to {full_path}"
    except Exception as e:
        return f"Error writing file: {e}"


def run_command(command: str, working_dir: str = ".") -> str:
    """Run a shell command and return its output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=working_dir,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]:\n{result.stderr}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30 seconds"
    except Exception as e:
        return f"Error running command: {e}"


def call_claude(prompt: str, working_dir: str = ".") -> str:
    """Call the Claude Code CLI to perform a subtask. Returns Claude's response.

    Environment variables from claude_env_sample.json are injected so the
    CLI routes through the configured backend (e.g. DeepSeek proxy).
    """
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "text",
             "--permission-mode", "bypassPermissions"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=working_dir,
            env=merge_claude_env(),
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]:\n{result.stderr}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: claude CLI timed out after 120 seconds"
    except FileNotFoundError:
        return "Error: claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
    except Exception as e:
        return f"Error calling claude CLI: {e}"


def web_search(query: str, max_results: int = 10, working_dir: str = ".") -> str:
    """Search the web using DuckDuckGo and return results with titles, URLs, and snippets.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return (default 10, max 20).
    """
    max_results = min(max_results, 20)
    encoded = urllib.parse.quote(query)
    url = f"https://lite.duckduckgo.com/lite?q={encoded}"

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; UniversalMultiAgent/1.0)",
                "Accept": "text/html",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return f"Error searching web: {e}"

    # Parse DuckDuckGo Lite results
    # Each result is a <tr> with class "result-snippet" containing link + snippet
    results = []
    # Find result rows: link in <a> with class "result-link", snippet in <td> with class "result-snippet"
    link_pattern = re.compile(
        r'<a[^>]*class="result-link"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>',
        re.IGNORECASE,
    )
    snippet_pattern = re.compile(
        r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>',
        re.DOTALL | re.IGNORECASE,
    )

    links = link_pattern.findall(html)
    snippets = snippet_pattern.findall(html)

    for i, (href, title) in enumerate(links):
        if i >= max_results:
            break
        snippet = ""
        if i < len(snippets):
            snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()
        results.append(f"{i+1}. **{title.strip()}**\n   URL: {href}\n   {snippet}")

    if not results:
        # Try alternative parsing for simpler layout
        alt_pattern = re.compile(
            r'<a[^>]*href="(https?://[^"]+)"[^>]*>([^<]+)</a>',
            re.IGNORECASE,
        )
        alt_links = alt_pattern.findall(html)
        for i, (href, title) in enumerate(alt_links):
            if i >= max_results or "duckduckgo.com" in href:
                continue
            title_clean = re.sub(r"<[^>]+>", "", title).strip()
            if title_clean:
                results.append(f"{len(results)+1}. **{title_clean}**\n   URL: {href}")

    if not results:
        return f"No results found for: {query}"

    return f"Web search results for '{query}':\n\n" + "\n\n".join(results)


def web_fetch(url: str, max_chars: int = 12000, working_dir: str = ".") -> str:
    """Fetch content from a URL and return as plain text.

    Bypasses Claude Code's permission system entirely — uses Python urllib
    directly. Use this for accessing arxiv.org, academic papers, and other
    trusted research sources.

    Args:
        url: Full URL to fetch.
        max_chars: Maximum characters to return (default 12000, max 20000).
    """
    max_chars = min(max_chars, 20000)
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; UMAF-research/1.0; +https://github.com/anthropics)",
                "Accept": "text/html,application/xhtml+xml,text/plain",
            },
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            content_type = resp.headers.get("Content-Type", "")
            data = resp.read()

            if "text/html" in content_type:
                text = data.decode("utf-8", errors="replace")
                # Strip script/style tags, then all HTML tags
                text = re.sub(
                    r"<(script|style|nav|footer|header)[^>]*>.*?</\1>",
                    "", text, flags=re.DOTALL | re.IGNORECASE,
                )
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
            else:
                text = data.decode("utf-8", errors="replace")

            return text[:max_chars]
    except urllib.error.HTTPError as e:
        return f"Error: HTTP {e.code} fetching URL: {url}"
    except urllib.error.URLError as e:
        return f"Error: could not reach {url} ({e.reason})"
    except Exception as e:
        return f"Error fetching URL: {e}"


def download_file(url: str, output_path: str, working_dir: str = ".") -> str:
    """Download content from a URL and save to a local file.

    Uses Python urllib directly — runs at the framework level, outside Claude Code's
    network sandbox. This is the primary way to fetch arxiv.org and other academic
    sites whose domains can't be verified by Claude Code's cc-switch layer.

    Args:
        url: Full URL to download.
        output_path: Path to save the downloaded content (relative to working_dir).
    """
    full_path = _resolve_path(output_path, working_dir)
    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; UMAF-research/1.0; +https://github.com/anthropics)",
                "Accept": "text/html,application/xhtml+xml,text/plain,application/pdf",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            full_path.write_bytes(data)
        size_kb = len(data) / 1024
        return f"Downloaded {size_kb:.1f}KB from {url} to {full_path}"
    except urllib.error.HTTPError as e:
        return f"Error: HTTP {e.code} downloading {url}"
    except urllib.error.URLError as e:
        return f"Error: could not reach {url} ({e.reason})"
    except Exception as e:
        return f"Error downloading {url}: {e}"


TOOL_MAP = {
    "read_file": read_file,
    "write_file": write_file,
    "write_lines": write_lines,
    "run_command": run_command,
    "call_claude": call_claude,
    "web_search": web_search,
    "web_fetch": web_fetch,
    "download_file": download_file,
}

# Populate ToolRegistry's TOOL_MAP with the same functions
ToolRegistry.TOOL_MAP = TOOL_MAP
