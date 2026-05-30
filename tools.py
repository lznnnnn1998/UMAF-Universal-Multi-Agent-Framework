import os
import re
import subprocess
import urllib.error
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
    full_path = Path(working_dir) / output_path
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
    "run_command": run_command,
    "call_claude": call_claude,
    "web_search": web_search,
    "web_fetch": web_fetch,
    "download_file": download_file,
}
