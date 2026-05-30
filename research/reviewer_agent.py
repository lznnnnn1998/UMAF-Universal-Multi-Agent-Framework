import json
import re
from typing import Any

from agent import run_agent
from tools import TOOL_MAP

REVIEWER_TOOLS = [
    {
        "name": "read_file",
        "description": "Read a file. Use this to read each research output file.",
        "parameters": {"path": "str"},
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Use to save the scoring report.",
        "parameters": {"path": "str", "content": "str"},
    },
    {
        "name": "web_search",
        "description": "Search the web to verify factual claims found in research outputs.",
        "parameters": {"query": "str", "max_results": "int (optional, default 10)"},
    },
]


def review_and_score(
    worker_outputs: list[dict[str, Any]],
    topic: str,
    working_dir: str,
    backend: str = "deepseek",
) -> list[dict[str, Any]]:
    """Review all worker outputs, score them, and return top 3 ranked.

    Returns a list of dicts with keys: sub_task_id, title, output_file,
    scores (dict of dimension->score), total_score, rank.
    """
    if not worker_outputs:
        return []

    file_list = "\n".join(
        f"  - [{wo['sub_task_id']}] {wo['output_file']}: {wo['title']}"
        for wo in worker_outputs
    )

    task = f"""You are a rigorous research reviewer. Read each research output file and score it on five dimensions.

## Research Topic
{topic}

## Files to Review
{file_list}

## Scoring Rubric (each dimension 1-10):
- **Depth** (1-10): How thorough and detailed is the research? Does it go beyond surface-level?
- **Accuracy** (1-10): Are claims well-supported? Are methods and findings correctly described?
- **Relevance** (1-10): How well does this sub-topic contribute to the main research topic?
- **Clarity** (1-10): Is the writing clear, well-organized, and easy to follow?
- **Originality** (1-10): Are there novel insights, unique angles, or creative synthesis?

## Instructions
1. Read EACH file listed above using `read_file`.
2. For each file, score all five dimensions with a brief justification.
3. Calculate total_score = sum of all five dimensions (max 50).
4. Rank all submissions by total_score (highest first).
5. Write the complete scoring report to `scoring_report.json` in valid JSON format:
```json
[
  {{
    "sub_task_id": 1,
    "title": "...",
    "output_file": "...",
    "scores": {{
      "depth": X,
      "accuracy": X,
      "relevance": X,
      "clarity": X,
      "originality": X,
      "justification": "Brief justification for each score..."
    }},
    "total_score": X,
    "rank": 1
  }}
]
```
6. Output TASK_COMPLETE when done.

Be fair but rigorous. The scoring will determine which proposals go into the final document."""

    result = run_agent(
        task=task,
        working_dir=working_dir,
        tools=REVIEWER_TOOLS,
        tool_map=TOOL_MAP,
        max_steps=15,
        backend=backend,
    )

    # Parse scores from the scoring report file or from agent output
    scores = _extract_scores(result, working_dir)
    return scores


def _extract_scores(result: dict, working_dir: str) -> list[dict[str, Any]]:
    """Extract scoring data from agent output or the scoring report file."""
    import os

    # Try reading the scoring report file first
    report_path = os.path.join(working_dir, "scoring_report.json")
    try:
        with open(report_path) as f:
            scores = json.load(f)
            if isinstance(scores, list) and len(scores) > 0:
                return sorted(scores, key=lambda s: s.get("total_score", 0), reverse=True)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Fallback: try to extract JSON from agent messages
    for msg in reversed(result["messages"]):
        content = msg.content if hasattr(msg, "content") else str(msg)
        match = re.search(r"\[[\s\S]*?\{[\s\S]*?\"sub_task_id\"[\s\S]*?\]", content)
        if match:
            try:
                scores = json.loads(match.group(0))
                if isinstance(scores, list) and len(scores) > 0:
                    return sorted(scores, key=lambda s: s.get("total_score", 0), reverse=True)
            except json.JSONDecodeError:
                continue

    return []
