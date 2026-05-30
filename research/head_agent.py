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
    """Decompose a broad research topic into 5-7 specific sub-topics.

    Returns a list of dicts with keys: id, title, description.
    """
    if backend == "claude_cli":
        tools = DECOMPOSE_TOOLS_CLAUDE_CLI
        task = f"""You are a research coordinator. Decompose the following broad research topic into 5-7 specific, well-scoped sub-topics suitable for independent investigation.

Research topic: {topic}

Requirements for each sub-topic:
- Be specific and self-contained
- Include concrete research questions or areas to explore
- Cover different angles of the main topic
- Be suitable for a 15-minute focused research session

Output format (MUST be valid JSON):
```json
[
  {{
    "id": 1,
    "title": "Short descriptive title",
    "description": "Detailed description of what to research, including specific questions to answer and angles to explore."
  }},
  ...
]
```

Use your own knowledge to decompose the topic — do NOT search the web. Output ONLY the JSON array, nothing else before or after. Then write TASK_COMPLETE."""
    else:
        tools = DECOMPOSE_TOOLS_DEEPSEEK
        task = f"""You are a research coordinator. Decompose the following broad research topic into 5-7 specific, well-scoped sub-topics suitable for independent investigation.

Research topic: {topic}

Requirements for each sub-topic:
- Be specific and self-contained
- Include concrete research questions or areas to explore
- Cover different angles of the main topic
- Be suitable for a 15-minute focused research session

Output format (MUST be valid JSON):
```json
[
  {{
    "id": 1,
    "title": "Short descriptive title",
    "description": "Detailed description of what to research, including specific questions to answer and angles to explore."
  }},
  ...
]
```

Output ONLY the JSON array, nothing else before or after. Then write TASK_COMPLETE."""

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

    Extracts keywords from the topic string to make sub-topics more specific.
    Splits on commas, 'and', and 'vs' to find candidate sub-topics.
    """
    import re

    # Extract likely sub-topic names from the topic string
    keywords = [s.strip() for s in re.split(r',| and | vs |;', topic) if len(s.strip()) >= 3]
    if not keywords:
        keywords = [topic]

    templates = []
    for i, kw in enumerate(keywords[:4]):
        clean_kw = kw.strip().rstrip('.')
        templates.append({
            "id": i + 1,
            "title": f"{clean_kw}: Mechanisms, Methods, and Key Results",
            "description": (
                f"Deep-dive into '{clean_kw}' within the context of {topic}. "
                f"Investigate the underlying mechanisms, established methods, representative results, "
                f"and identify the most influential papers and benchmarks."
            ),
        })

    # Always add a comparative analysis and future directions
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

    # Pad to at least 5 sub-tasks
    generic_fillers = [
        ("Benchmarking and Evaluation Methodologies",
         f"Survey standard benchmarks, evaluation protocols, and metrics used to assess approaches in {topic}. Compare reported results and analyze reproducibility."),
        ("Practical Deployment and Systems Integration",
         f"Investigate real-world deployment challenges for {topic}: hardware constraints, latency requirements, memory budgets, and integration with production systems."),
    ]
    while len(templates) < 5:
        idx = len(templates) - base_id - 2
        ftitle, fdesc = generic_fillers[idx]
        templates.append({
            "id": len(templates) + 1,
            "title": f"{ftitle} for {topic}",
            "description": fdesc,
        })

    return templates
