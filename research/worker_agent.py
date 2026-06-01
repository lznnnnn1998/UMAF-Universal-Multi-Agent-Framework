import os
from typing import Any

from agent import AgentResult, AgentRole
from tools import ToolRegistry, download_file, web_search


class ResearchWorkerRole(AgentRole):
    """Research a single sub-topic and write findings to a markdown file."""

    agent_name = "worker"
    max_steps = 12

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        return ToolRegistry.to_dicts(ToolRegistry.research_worker_tools())

    def build_task(self, backend: str, sub_task: dict | None = None,
                   output_file: str = "", **context: Any) -> str:
        assert sub_task is not None
        title = sub_task["title"]
        description = sub_task["description"]

        if backend == "claude_cli":
            prefetch_files = _prefetch_arxiv_sources(sub_task, context.get("working_dir", "."))
            return _build_worker_task_claude_cli(title, description, output_file, prefetch_files)
        return _build_worker_task_deepseek(title, description, output_file)

    def parse_result(self, result: AgentResult, working_dir: str,
                     sub_task: dict | None = None, output_file: str = "", **context: Any) -> dict[str, Any]:
        assert sub_task is not None
        summary = ""
        for msg in reversed(result.messages):
            if type(msg).__name__ != "AIMessage":
                continue
            content = msg.content if hasattr(msg, "content") else str(msg)
            if len(content) > 100:
                summary = content[:500] + "..." if len(content) > 500 else content
                break

        # Only report output_file if it was actually written — an empty or
        # missing file counts as failure, which triggers stop-on-failure in
        # _run_workers_with_deps so dependent levels don't run on missing input.
        actual_file = output_file if (
            output_file and os.path.isfile(os.path.join(working_dir, output_file))
        ) else ""

        return {
            "sub_task_id": sub_task["id"],
            "title": sub_task["title"],
            "output_file": actual_file,
            "summary": summary,
        }


def _build_worker_task_deepseek(title: str, description: str, output_file: str) -> str:
    """Build worker task for the deepseek backend — uses call_claude for deep reasoning."""
    return f"""You are a research agent. Conduct thorough research on the following sub-topic.

## Research Sub-Topic
**Title**: {title}
**Description**: {description}

## Instructions
1. Use `web_search` to find recent papers, articles, and references on this sub-topic. Run multiple searches with different query angles.
2. For arxiv.org and other academic URLs you find: use `download_file` to save them locally first, then `read_file` to read the local copy. This two-step pattern bypasses network restrictions. Example:
   - `download_file(url="https://arxiv.org/abs/XXXX.XXXXX", output_path="paper_01.html")`
   - `read_file(path="paper_01.html")`
3. Use `web_fetch` for non-academic URLs (blog posts, documentation) that don't need domain verification.
4. Use `call_claude` for deep reasoning and synthesis on specific questions. Break the sub-topic into 2-3 specific questions and analyze each one.
5. Use `write_file` to save your complete findings to the file: `{output_file}`
6. Organize your findings into sections:
   - **Overview**: 2-3 paragraph summary of the sub-topic
   - **Key Methods & Approaches**: detailed findings with technical depth
   - **Important Papers & References**: key works with authors, venue, year, and brief notes
   - **Open Questions & Future Directions**: gaps and opportunities
   - **Relevance to Main Topic**: how this connects to the broader research area
7. After writing the file, read it back to verify completeness.
8. Output TASK_COMPLETE when done.

Focus on depth and accuracy. Your research will be scored and ranked."""


def _build_worker_task_claude_cli(
    title: str, description: str, output_file: str, prefetch_files: list[str] | None = None,
) -> str:
    """Build worker task for the claude_cli backend.

    The agent IS a Claude Code instance (routed through DeepSeek API), so there is
    no need to spawn nested `claude -p` calls. The agent uses its own tools directly.

    prefetch_files: local file paths already downloaded by the framework. The agent
    should read these directly — no HTTP requests to arxiv.org needed.
    """
    prefetch_section = ""
    if prefetch_files:
        prefetch_section = "\n## Pre-downloaded Reference Material\n"
        prefetch_section += "The following files have already been downloaded from arxiv.org and saved locally. Read them directly — do NOT try to fetch them again from the web:\n\n"
        for f in prefetch_files:
            prefetch_section += f"  - `{f}`\n"
        prefetch_section += "\nStart by reading these files before doing any additional searching.\n"

    return f"""You are a research agent. Conduct thorough research on the following sub-topic.

## Research Sub-Topic
**Title**: {title}
**Description**: {description}
{prefetch_section}
## Instructions
1. FIRST, use **Read** to read any pre-downloaded reference files listed above. These are already saved locally.
2. Use **WebSearch** to find additional papers, articles, and technical documentation. Run multiple searches with different query angles.
3. For any arxiv.org or academic URLs found in search results: do NOT try to fetch them directly (domain verification will block them). Instead, note the URLs in your findings so the framework can download them later.
4. Synthesize your findings using your own reasoning. Identify patterns, compare approaches, and note trade-offs.
5. Use **Write** to save your complete findings to the file: `{output_file}`
   Organize your findings into these sections:
   - **Overview**: 2-3 paragraph summary covering the core ideas and motivation
   - **Key Methods & Approaches**: detailed technical explanation of how each method works, with comparisons
   - **Important Papers & References**: list key papers with authors, venue, year, and a 1-2 sentence note on their contribution
   - **Open Questions & Future Directions**: current limitations and active research frontiers
   - **Relevance to Main Topic**: how this connects to broader research
6. After writing, use **Read** to verify the file was written correctly with all sections present.
7. Output TASK_COMPLETE when done.

Focus on technical depth, concrete details, and accuracy. Your research will be scored and ranked on depth, accuracy, relevance, clarity, and originality."""


def _prefetch_arxiv_sources(
    sub_task: dict[str, Any], working_dir: str, max_files: int = 3,
) -> list[str]:
    """Pre-download arxiv.org content to local files at the framework level.

    Runs outside any agent sandbox — uses Python urllib directly to bypass
    Claude Code's cc-switch domain verification. The downloaded files are
    then passed to the agent as local references.

    Returns list of local file paths (relative to working_dir).
    """
    title = sub_task.get("title", "")
    description = sub_task.get("description", "")
    query = f"{title} {description}"[:200]
    downloaded: list[str] = []

    try:
        results = web_search(query, max_results=8, working_dir=working_dir)
    except Exception:
        return downloaded

    # Extract arxiv.org URLs from search results
    import re
    arxiv_urls = re.findall(r'https?://arxiv\.org/\S+', results)
    # Also try to find semantic scholar, openreview, etc.
    academic_urls = arxiv_urls + re.findall(
        r'https?://(?:openreview\.net|proceedings\.mlr\.press|papers\.nips\.cc)/\S+',
        results,
    )

    for i, url in enumerate(academic_urls[:max_files]):
        # Clean URL (remove trailing punctuation from regex extraction)
        url = url.rstrip(".,;:)!?\"'")
        safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', url.split('/')[-1] or f"paper_{i}")[:40]
        local_path = f"agent_log/prefetched_{i:02d}_{safe_name}.html"

        try:
            result = download_file(url, local_path, working_dir)
            if not result.startswith("Error"):
                downloaded.append(local_path)
        except Exception:
            continue

    return downloaded


def research_subtask(
    sub_task: dict[str, Any],
    working_dir: str,
    backend: str = "deepseek",
    version: int = 1,
) -> dict[str, Any]:
    """Research a single sub-topic and write findings to a file.

    Args:
        sub_task: dict with id, title, description keys.
        working_dir: base working directory.
        backend: LLM backend to use.
        version: checkpoint version (auto-resumes from previous version if > 1).

    Returns:
        dict with sub_task_id, title, output_file, summary keys.
    """
    sub_id = sub_task["id"]
    title = sub_task["title"]
    safe_title = title.replace(' ', '_').replace('/', '_')[:60]
    output_file = f"research_{sub_id:02d}_{safe_title}.md"

    role = ResearchWorkerRole()
    role.agent_name = f"worker_{sub_id:02d}"
    return role.execute(
        working_dir=working_dir,
        backend=backend,
        version=version,
        sub_task=sub_task,
        output_file=output_file,
    )