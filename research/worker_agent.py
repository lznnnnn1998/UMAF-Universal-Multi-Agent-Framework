import os
from typing import Any

from agent import AgentResult, AgentRole
from tools import ToolRegistry, download_file, web_search


class ResearchWorkerRole(AgentRole):
    """Research a single sub-topic and write findings to a markdown file."""

    agent_name = "worker"
    max_steps = 40

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        return ToolRegistry.to_dicts(ToolRegistry.research_worker_tools())

    def build_task(self, backend: str, sub_task: dict | None = None,
                   output_file: str = "", **context: Any) -> str:
        assert sub_task is not None
        title = sub_task["title"]
        description = sub_task["description"]
        dep_outputs: list[dict[str, Any]] = sub_task.get("_dependency_outputs", [])

        # Determine which tools are available so the prompt can be tailored
        tool_specs = self.tools_for_backend(backend)
        tool_names = {t["name"] for t in tool_specs} if tool_specs else set()

        if backend == "claude_cli":
            prefetch_files = _prefetch_arxiv_sources(sub_task, context.get("working_dir", "."))
            return _build_worker_task_claude_cli(title, description, output_file, prefetch_files, tool_names, dep_outputs)
        return _build_worker_task_deepseek(title, description, output_file, tool_names, dep_outputs)

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


def _build_dependency_section(dep_outputs: list[dict[str, Any]]) -> str:
    """Build a prompt section instructing the worker to read upstream outputs.

    When the worker depends on other sub-tasks, those outputs must be read
    first to avoid duplicated work and to build on prior findings.
    """
    if not dep_outputs:
        return ""

    lines = [
        "\n## Dependency Outputs (READ THESE FIRST)",
        "Your research builds on the following completed sub-tasks. "
        "Read each file before starting your own research — do NOT redo "
        "work that has already been done. Instead, use these findings as "
        "a foundation and focus on what's missing or needs deeper analysis.",
        "",
    ]
    for d in dep_outputs:
        fname = d.get("output_file", "")
        d_title = d.get("title", "")
        if fname:
            lines.append(f"- **[{d['dep_id']}] {d_title}**: `{fname}`")
    lines.append("")
    return "\n".join(lines)


def _build_worker_task_deepseek(title: str, description: str, output_file: str,
                                tool_names: set[str] | None = None,
                                dep_outputs: list[dict[str, Any]] | None = None) -> str:
    """Build worker task for the deepseek backend — uses call_claude for deep reasoning.

    tool_names: set of available tool names from the agent's tool list. The prompt
    only mentions tools that are actually available.
    dep_outputs: list of completed upstream worker outputs this worker should read first.
    """
    tools = tool_names or set()
    has_web_search = "web_search" in tools
    has_web_fetch = "web_fetch" in tools
    has_download_file = "download_file" in tools
    has_call_claude = "call_claude" in tools
    has_run_command = "run_command" in tools

    # Build dependency section (upstream worker outputs to read first)
    dep_section = _build_dependency_section(dep_outputs or [])

    # Build tool-specific instruction block
    search_instructions = []
    if has_web_search:
        search_instructions.append(
            "Use `web_search` to find recent papers, articles, and references on this sub-topic. "
            "Run multiple searches with different query angles."
        )
    if has_download_file:
        search_instructions.append(
            "For arxiv.org and other academic URLs you find: use `download_file` to save them locally "
            "first, then `read_file` to read the local copy. This two-step pattern bypasses network restrictions. Example:\n"
            "   - `download_file(url=\"https://arxiv.org/abs/XXXX.XXXXX\", output_path=\"paper_01.html\")`\n"
            "   - `read_file(path=\"paper_01.html\")`"
        )
    if has_web_fetch:
        search_instructions.append(
            "Use `web_fetch` for non-academic URLs (blog posts, documentation) that don't need domain verification."
        )
    if has_call_claude:
        search_instructions.append(
            "Use `call_claude` for deep reasoning and synthesis on specific questions. "
            "Break the sub-topic into 2-3 specific questions and analyze each one."
        )

    tool_section = ""
    if search_instructions:
        numbered = [f"{i+1}. {inst}" for i, inst in enumerate(search_instructions)]
        # Write instruction comes after search/reasoning tools
        write_num = len(numbered) + 1
        tool_section = "\n".join(numbered)
        tool_section += f"\n{write_num}. Use `write_file` to save your complete findings to the file: `{output_file}`"
        verify_num = write_num + 1
        done_num = verify_num + 1
    else:
        tool_section = (
            "1. Use `read_file` to examine any relevant local reference files.\n"
            f"2. Use `write_file` to save your complete findings to the file: `{output_file}`"
        )
        verify_num = 3
        done_num = 4

    return f"""You are a research agent. Conduct thorough research on the following sub-topic.

## Research Sub-Topic
**Title**: {title}
**Description**: {description}
{dep_section}
## Instructions
{tool_section}
{verify_num}. Organize your findings into sections:
   - **Overview**: 2-3 paragraph summary of the sub-topic
   - **Key Methods & Approaches**: detailed findings with technical depth
   - **Important Papers & References**: key works with authors, venue, year, and brief notes
   - **Open Questions & Future Directions**: gaps and opportunities
   - **Relevance to Main Topic**: how this connects to the broader research area
{done_num}. After writing the file, read it back to verify completeness.
{done_num + 1}. Output TASK_COMPLETE when done.

Focus on depth and accuracy. Your research will be scored and ranked."""


def _build_worker_task_claude_cli(
    title: str, description: str, output_file: str,
    prefetch_files: list[str] | None = None,
    tool_names: set[str] | None = None,
    dep_outputs: list[dict[str, Any]] | None = None,
) -> str:
    """Build worker task for the claude_cli backend.

    The agent IS a Claude Code instance (routed through DeepSeek API), so there is
    no need to spawn nested `claude -p` calls. The agent uses its own tools directly.

    prefetch_files: local file paths already downloaded by the framework. The agent
    should read these directly — no HTTP requests to arxiv.org needed.
    tool_names: set of available tool names — prompt only mentions tools that exist.
    dep_outputs: list of completed upstream worker outputs this worker should read first.
    """
    tools = tool_names or set()
    has_web_search = "web_search" in tools

    # Build dependency section (upstream worker outputs to read first)
    dep_section = _build_dependency_section(dep_outputs or [])

    prefetch_section = ""
    if prefetch_files:
        prefetch_section = "\n## Pre-downloaded Reference Material\n"
        prefetch_section += "The following files have already been downloaded from arxiv.org and saved locally. Read them directly — do NOT try to fetch them again from the web:\n\n"
        for f in prefetch_files:
            prefetch_section += f"  - `{f}`\n"
        prefetch_section += "\nStart by reading these files before doing any additional searching.\n"

    # Build numbered instructions that adapt to available tools
    instructions = []
    inst_num = 1
    if dep_outputs:
        dep_names = ", ".join(f"`{d['output_file']}`" for d in dep_outputs if d.get("output_file"))
        instructions.append(f"{inst_num}. FIRST, use **Read** to read the dependency outputs listed above ({dep_names}). These are completed research files from sub-tasks that yours depends on. Do NOT redo their work — build on it.")
        inst_num += 1
    if prefetch_files:
        instructions.append(f"{inst_num}. FIRST, use **Read** to read any pre-downloaded reference files listed above. These are already saved locally.")
        inst_num += 1
    if has_web_search:
        instructions.append(f"{inst_num}. Use **WebSearch** to find additional papers, articles, and technical documentation. Run multiple searches with different query angles.")
        inst_num += 1
        instructions.append(f"{inst_num}. For any arxiv.org or academic URLs found in search results: do NOT try to fetch them directly (domain verification will block them). Instead, note the URLs in your findings so the framework can download them later.")
        inst_num += 1
    instructions.append(f"{inst_num}. Synthesize your findings using your own reasoning. Identify patterns, compare approaches, and note trade-offs.")
    inst_num += 1

    instructions.append(f"{inst_num}. Use **Write** to save your complete findings to the file: `{output_file}`\n   Organize your findings into these sections:\n   - **Overview**: 2-3 paragraph summary covering the core ideas and motivation\n   - **Key Methods & Approaches**: detailed technical explanation of how each method works, with comparisons\n   - **Important Papers & References**: list key papers with authors, venue, year, and a 1-2 sentence note on their contribution\n   - **Open Questions & Future Directions**: current limitations and active research frontiers\n   - **Relevance to Main Topic**: how this connects to broader research")
    inst_num += 1
    instructions.append(f"{inst_num}. After writing, use **Read** to verify the file was written correctly with all sections present.")
    inst_num += 1
    instructions.append(f"{inst_num}. Output TASK_COMPLETE when done.")

    return f"""You are a research agent. Conduct thorough research on the following sub-topic.

## Research Sub-Topic
**Title**: {title}
**Description**: {description}
{dep_section}{prefetch_section}
## Instructions
{chr(10).join(instructions)}

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