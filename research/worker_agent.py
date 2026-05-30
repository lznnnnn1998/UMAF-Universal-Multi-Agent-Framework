from typing import Any

from agent import run_agent
from tools import TOOL_MAP

WORKER_TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file.",
        "parameters": {"path": "str"},
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Use this to save research findings.",
        "parameters": {"path": "str", "content": "str"},
    },
    {
        "name": "run_command",
        "description": "Run a shell command. Use for quick checks.",
        "parameters": {"command": "str"},
    },
    {
        "name": "call_claude",
        "description": "Call the Claude Code CLI for deep research, analysis, or synthesis on a specific question. Provide a detailed prompt with what you need investigated.",
        "parameters": {"prompt": "str"},
    },
    {
        "name": "web_search",
        "description": "Search the web for research papers, articles, and information. Returns titles, URLs, and snippets. Essential for finding sources.",
        "parameters": {"query": "str", "max_results": "int (optional, default 10)"},
    },
]


def _build_worker_task_deepseek(title: str, description: str, output_file: str) -> str:
    """Build worker task for the deepseek backend — uses call_claude for deep reasoning."""
    return f"""You are a research agent. Conduct thorough research on the following sub-topic.

## Research Sub-Topic
**Title**: {title}
**Description**: {description}

## Instructions
1. Use `call_claude` for deep research on specific questions within this sub-topic. Break the sub-topic into 2-3 specific questions and research each one.
2. Use `web_search` to find recent papers, articles, and references on each question.
3. Use `write_file` to save your complete findings to the file: `{output_file}`
4. Organize your findings into sections:
   - **Overview**: 2-3 paragraph summary of the sub-topic
   - **Key Methods & Approaches**: detailed findings
   - **Important Papers & References**: key works with brief notes
   - **Open Questions & Future Directions**: gaps and opportunities
   - **Relevance to Main Topic**: how this connects to the broader research area
5. After writing the file, read it back to verify completeness.
6. Output TASK_COMPLETE when done.

Focus on depth and accuracy. Your research will be scored and ranked."""


def _build_worker_task_claude_cli(title: str, description: str, output_file: str) -> str:
    """Build worker task for the claude_cli backend — uses the agent's own tools directly.

    The agent IS a full Claude Code instance (routed through DeepSeek API), so there is
    no need to spawn nested `claude -p` calls. The agent uses WebSearch, Read/curl, and
    its own reasoning to research, then writes findings with Write.
    """
    return f"""You are a research agent. Conduct thorough research on the following sub-topic.

## Research Sub-Topic
**Title**: {title}
**Description**: {description}

## Instructions
1. Use **WebSearch** to find recent papers, articles, and technical documentation about this sub-topic. Run multiple searches with different query angles for broad coverage.
2. For promising papers or articles, use **Bash** with `curl -sL <url> | head -300` to fetch and read their content directly. Skip paywalled papers and look for arXiv preprints, open-access versions, or technical blog posts.
3. Synthesize your findings using your own reasoning. Identify patterns, compare approaches, and note trade-offs.
4. Use **Write** to save your complete findings to the file: `{output_file}`
   Organize your findings into these sections:
   - **Overview**: 2-3 paragraph summary covering the core ideas and motivation
   - **Key Methods & Approaches**: detailed technical explanation of how each method works, with comparisons
   - **Important Papers & References**: list key papers with authors, venue, year, and a 1-2 sentence note on their contribution
   - **Open Questions & Future Directions**: current limitations and active research frontiers
   - **Relevance to Main Topic**: how this connects to broader transformer and attention optimization research
5. After writing, use **Read** to verify the file was written correctly with all sections present.
6. Output TASK_COMPLETE when done.

Focus on technical depth, concrete details, and accuracy. Your research will be scored and ranked on depth, accuracy, relevance, clarity, and originality."""


def research_subtask(
    sub_task: dict[str, Any],
    working_dir: str,
    backend: str = "deepseek",
) -> dict[str, Any]:
    """Research a single sub-topic and write findings to a file.

    Args:
        sub_task: dict with id, title, description keys.
        working_dir: base working directory.
        backend: LLM backend to use.

    Returns:
        dict with sub_task_id, title, output_file, summary keys.
    """
    sub_id = sub_task["id"]
    title = sub_task["title"]
    description = sub_task["description"]
    safe_title = title.replace(' ', '_').replace('/', '_')[:60]
    output_file = f"research_{sub_id:02d}_{safe_title}.md"

    if backend == "claude_cli":
        task = _build_worker_task_claude_cli(title, description, output_file)
    else:
        task = _build_worker_task_deepseek(title, description, output_file)

    result = run_agent(
        task=task,
        working_dir=working_dir,
        tools=WORKER_TOOLS,
        tool_map=TOOL_MAP,
        max_steps=12,
        backend=backend,
        agent_name=f"worker_{sub_id:02d}",
    )

    # Extract the final content for a summary
    summary = ""
    for msg in reversed(result["messages"]):
        content = msg.content if hasattr(msg, "content") else str(msg)
        if len(content) > 100:
            summary = content[:500] + "..." if len(content) > 500 else content
            break

    return {
        "sub_task_id": sub_id,
        "title": title,
        "output_file": output_file,
        "summary": summary,
    }
