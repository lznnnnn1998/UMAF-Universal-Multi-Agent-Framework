import json
import re
from typing import Any

from agent import run_agent
from tools import TOOL_MAP

DECOMPOSE_TOOLS_DEEPSEEK = [
    {
        "name": "run_command",
        "description": "Run a shell command. Useful for quick exploration or counting.",
        "parameters": {"command": "str"},
    },
    {
        "name": "call_claude",
        "description": "Call Claude Code CLI for brainstorming or validation.",
        "parameters": {"prompt": "str"},
    },
    {
        "name": "web_search",
        "description": "Search the web to discover sub-topics or verify topic scope.",
        "parameters": {"query": "str", "max_results": "int (optional, default 10)"},
    },
]

# For claude_cli backend: give Read only — the agent should reason directly without tools.
# Having at least one tool ensures --allowedTools is passed, restricting the agent's tool set.
DECOMPOSE_TOOLS_CLAUDE_CLI = [
    {
        "name": "read_file",
        "description": "Read a file. Not needed for decomposition — use your own knowledge.",
        "parameters": {"path": "str"},
    },
]


def decompose_topic(topic: str, working_dir: str, backend: str = "deepseek") -> list[dict[str, Any]]:
    """Decompose a research topic into 2-8 sub-topics, scaled to its complexity.

    Returns a list of dicts with keys: id, title, description.
    """
    json_template = """[
  {
    "id": 1,
    "title": "Short descriptive title",
    "description": "Detailed description of what to research, including specific questions to answer and angles to explore."
  },
  ...
]"""

    common = f"""You are a research coordinator. Analyze the complexity of this research topic and decompose it into an appropriate number of sub-topics — at least 2 and at most 8.

Research topic: {topic}

## How to size the decomposition
- **Narrow/specific topic** (e.g. a single method or technique): 2-3 sub-topics
- **Moderate topic** (e.g. a family of techniques or one research area): 4-5 sub-topics
- **Broad/complex topic** (e.g. a whole field or comparing multiple paradigms): 6-8 sub-topics

## Requirements for each sub-topic
- Be specific and self-contained
- Include concrete research questions or areas to explore
- Cover different angles of the main topic
- Be suitable for a 15-minute focused research session

Output format (MUST be valid JSON):
```json
{json_template}
```"""

    if backend == "claude_cli":
        tools = DECOMPOSE_TOOLS_CLAUDE_CLI
        task = f"{common}\n\nUse your own knowledge to decompose the topic — do NOT search the web. Output ONLY the JSON array, nothing else before or after. Then write TASK_COMPLETE."
    else:
        tools = DECOMPOSE_TOOLS_DEEPSEEK
        task = f"{common}\n\nOutput ONLY the JSON array, nothing else before or after. Then write TASK_COMPLETE."

    result = run_agent(
        task=task,
        working_dir=working_dir,
        tools=tools,
        tool_map=TOOL_MAP,
        max_steps=8,
        backend=backend,
        agent_name="head_decompose",
    )

    # Extract JSON from the last assistant message
    sub_tasks = []
    for msg in reversed(result["messages"]):
        content = msg.content if hasattr(msg, "content") else str(msg)
        # Try to find a JSON array in the content
        match = re.search(r"\[[\s\S]*\]", content)
        if match:
            try:
                sub_tasks = json.loads(match.group(0))
                if isinstance(sub_tasks, list) and len(sub_tasks) > 0:
                    break
            except json.JSONDecodeError:
                continue

    if not sub_tasks:
        # Fallback: manual decomposition
        sub_tasks = _fallback_decompose(topic)

    return sub_tasks


def _fallback_decompose(topic: str) -> list[dict[str, Any]]:
    """Generate a fallback decomposition if the LLM fails.

    Extracts keywords from the topic and scales sub-topic count from 2 to 8
    based on the number of distinct keywords found (min 2, max 8).
    """
    import re

    keywords = [s.strip() for s in re.split(r',| and | vs |;', topic) if len(s.strip()) >= 2]
    if not keywords:
        keywords = [topic]

    templates = []
    # One sub-topic per keyword (up to 6), plus comparative + future directions = up to 8
    kw_limit = min(len(keywords), 6)
    for i in range(kw_limit):
        kw = keywords[i].strip().rstrip('.')
        templates.append({
            "id": i + 1,
            "title": f"{kw}: Mechanisms, Methods, and Key Results",
            "description": (
                f"Deep-dive into '{kw}' within the context of {topic}. "
                f"Investigate the underlying mechanisms, established methods, representative results, "
                f"and identify the most influential papers and benchmarks."
            ),
        })

    # Always add comparative analysis and future directions
    base_id = len(templates)
    templates.append({
        "id": base_id + 1,
        "title": f"Comparative Analysis of Approaches in {topic}",
        "description": (
            f"Compare and contrast the major approaches within {topic}. "
            f"Analyze trade-offs in accuracy, efficiency, implementation complexity, and applicability. "
            f"Identify which methods work best under which conditions."
        ),
    })
    templates.append({
        "id": base_id + 2,
        "title": f"Open Problems and Emerging Directions in {topic}",
        "description": (
            f"Identify open research questions, recent breakthroughs, and promising future directions in {topic}. "
            f"Survey papers from the last 1-2 years for emerging trends and unresolved challenges."
        ),
    })

    # Ensure at least 2 sub-tasks (from single-keyword topics)
    if len(templates) < 2:
        templates.append({
            "id": len(templates) + 1,
            "title": f"Overview and Key Techniques in {topic}",
            "description": (
                f"Provide a comprehensive overview of {topic}, covering the foundational concepts, "
                f"key techniques, and major milestones in the field."
            ),
        })

    # Cap at 8
    return templates[:8]
