import json
import re
from typing import Any

from agent import AgentResult, AgentRole
from tools import ToolRegistry


class ResearchReviewerRole(AgentRole):
    """Review all worker outputs, score them, and return ranked results."""

    agent_name = "reviewer"
    max_steps = 15

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        return ToolRegistry.to_dicts(ToolRegistry.research_reviewer_tools())

    def build_task(self, backend: str, topic: str = "",
                   worker_outputs: list[dict[str, Any]] | None = None, **context: Any) -> str:
        assert worker_outputs is not None
        file_list = "\n".join(
            f"  - [{wo['sub_task_id']}] {wo['output_file']}: {wo['title']}"
            for wo in worker_outputs
        )

        return f"""You are a rigorous research reviewer. Read each research output file and score it on five dimensions.

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

    def parse_result(self, result: AgentResult, working_dir: str, **context: Any) -> list[dict[str, Any]]:
        return _extract_scores_from_result(result, working_dir)


def review_and_score(
    worker_outputs: list[dict[str, Any]],
    topic: str,
    working_dir: str,
    backend: str = "deepseek",
    version: int = 1,
) -> list[dict[str, Any]]:
    """Review all worker outputs, score them, and return top 3 ranked.

    Returns a list of dicts with keys: sub_task_id, title, output_file,
    scores (dict of dimension->score), total_score, rank.
    """
    if not worker_outputs:
        return []

    role = ResearchReviewerRole()
    return role.execute(
        working_dir=working_dir,
        backend=backend,
        version=version,
        topic=topic,
        worker_outputs=worker_outputs,
    )


def _extract_scores_from_result(result: AgentResult, working_dir: str) -> list[dict[str, Any]]:
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
    for msg in reversed(result.messages):
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
