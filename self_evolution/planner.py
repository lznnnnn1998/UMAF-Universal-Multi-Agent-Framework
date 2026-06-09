"""Self-evolution planner — creates an improvement plan from analysis results."""

from __future__ import annotations

import json
import os
from typing import Any

from agent import AgentResult, AgentRole
from tools import ToolRegistry


class SelfEvolutionPlannerRole(AgentRole):
    """Create an actionable improvement plan based on the analysis report.

    Reads the analysis_report.json, prioritizes improvements, and produces
    an implementation_plan.json with concrete file modifications.
    """

    agent_name = "self_evolution_planner"
    max_steps = 15

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        return ToolRegistry.to_dicts(ToolRegistry.self_evolution_planner_tools())

    def build_task(self, backend: str, working_dir: str = "",
                   analysis_report: dict[str, Any] | None = None,
                   project_dir: str = ".", **context: Any) -> str:
        report_summary = ""
        if analysis_report:
            opportunities = analysis_report.get("improvement_opportunities", [])
            opportunity_list = "\n".join(
                f"  - [{o['id']}] ({o['severity']}) {o['category']}: {o['title']}"
                for o in opportunities
            )
            report_summary = f"""
## Analysis Report
{analysis_report.get('summary', 'No summary available.')}

### Improvement Opportunities Found
{opportunity_list if opportunity_list else 'None found.'}
"""

        return f"""You are a self-evolution planner for UMAF. Based on the analysis report, create a concrete implementation plan for improving UMAF's own code.

{report_summary}

## Planning Instructions

1. Read the analysis report at {working_dir}/analysis_report.json
2. Prioritize improvements by severity (high → medium → low) and feasibility (easy wins first)
3. For each improvement, identify:
   - Exact file paths to read and modify
   - What specific code needs to change
   - What the new code should look like
   - How to verify the improvement works

4. Read the actual files you plan to modify to understand current code context.
5. Write `implementation_plan.json` to the working directory:

```json
{{
  "project_dir": "{project_dir}",
  "improvements": [
    {{
      "id": "SEO-001",
      "title": "...",
      "category": "...",
      "severity": "high | medium | low",
      "action": "modify | create | delete",
      "files_to_modify": [
        {{
          "path": "relative/path/to/file.py",
          "section": "line_range or function_name to modify",
          "current_code_snippet": "what the code looks like now (from reading the file)",
          "new_code": "what the code should look like after the change",
          "reasoning": "why this change improves UMAF"
        }}
      ],
      "files_to_create": [
        {{
          "path": "relative/path/to/new_file.py",
          "description": "what this file contains",
          "content_outline": "brief outline of the file structure"
        }}
      ],
      "verification": "How to verify this improvement (e.g., 'Run pytest test/ -q')"
    }}
  ],
  "estimated_impact": "Summary of expected improvements in 1-2 sentences.",
  "risk_assessment": "Any risks associated with these changes."
}}
```

6. Be specific and concrete — the coder will implement exactly what you specify.
7. After writing the plan, output TASK_COMPLETE."""

    def parse_result(self, result: AgentResult, working_dir: str,
                     project_dir: str = ".", **context: Any) -> dict[str, Any]:
        plan_path = os.path.join(working_dir, "implementation_plan.json")
        if os.path.exists(plan_path):
            try:
                with open(plan_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        # Fallback: extract JSON from agent messages
        from utils import extract_json_object
        for msg in reversed(result.messages):
            content = msg.content if hasattr(msg, "content") else str(msg)
            extracted = extract_json_object(content)
            if extracted:
                try:
                    parsed = json.loads(extracted)
                    if "improvements" in parsed:
                        return parsed
                except json.JSONDecodeError:
                    continue

        return self._fallback_plan(project_dir, working_dir)

    @staticmethod
    def _fallback_plan(project_dir: str, working_dir: str) -> dict[str, Any]:
        """Deterministic fallback: create a minimal improvement plan."""
        return {
            "_fallback": True,
            "project_dir": project_dir,
            "improvements": [
                {
                    "id": "SEO-FB-001",
                    "title": "Enhance test coverage for behavioral verification",
                    "category": "test_gaps",
                    "severity": "medium",
                    "action": "modify",
                    "files_to_modify": [],
                    "files_to_create": [],
                    "verification": "Run `python -m pytest test/ -q` to verify all tests pass.",
                },
            ],
            "estimated_impact": "Improve test reliability by verifying actual behavior, not just structure.",
            "risk_assessment": "Low risk — only test files are modified.",
        }
