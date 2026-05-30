import os
import re
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

from claude_config import merge_claude_env


def read_file(path: str, working_dir: str = ".") -> str:
    """Read contents of a file at path, resolved relative to working_dir."""
    full_path = Path(working_dir) / path
    try:
        return full_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"Error: file not found: {full_path}"
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(path: str, content: str, working_dir: str = ".") -> str:
    """Write content to a file at path, resolved relative to working_dir."""
    full_path = Path(working_dir) / path
    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        return f"Successfully wrote to {full_path}"
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
            ["claude", "-p", prompt, "--output-format", "text"],
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


TOOL_MAP = {
    "read_file": read_file,
    "write_file": write_file,
    "run_command": run_command,
    "call_claude": call_claude,
    "web_search": web_search,
}
