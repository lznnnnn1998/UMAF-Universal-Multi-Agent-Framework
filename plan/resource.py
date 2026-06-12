"""PlanResourceEstimatorRole — effort and resource estimation agent."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from tools import ToolRegistry
from agent import AgentResult, AgentRole
from utils import extract_json_object


class PlanResourceEstimatorRole(AgentRole):
    """Estimates effort and resource requirements for each leaf task.

    For each leaf: person-hours estimate with low/medium/high confidence,
    required skills/roles, resource types (developer, designer, DevOps, QA).
    Performs bottom-up aggregation so epic/goal nodes contain summed estimates.

    Identifies resource contention: when parallelizable branches require
    the same scarce resource.
    """

    agent_name: str = "resource_estimator"
    max_steps: int = 15

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        """Return Read-only tool specs."""
        if hasattr(ToolRegistry, "plan_resource_estimator_tools"):
            return ToolRegistry.to_dicts(ToolRegistry.plan_resource_estimator_tools())
        return ToolRegistry.to_dicts([ToolRegistry.READ_FILE])

    def build_task(self, backend: str, working_dir: str = ".",
                   task_tree: dict[str, Any] | None = None, **context: Any) -> str:
        """Build the resource estimation prompt."""
        tree_summary = self._summarize_tree(task_tree)

        common = (
            f"You are a resource estimation expert. Your job is to estimate "
            f"effort and resource requirements for every leaf task in a task tree, "
            f"then roll up estimates bottom-up through the hierarchy.\n\n"
            f"## Task Tree Summary\n{tree_summary}\n\n"
            f"## Instructions\n"
            f"1. For each leaf task, estimate:\n"
            f"   - **hours_estimate**: person-hours (best estimate)\n"
            f"   - **hours_low**: optimistic estimate (everything goes well)\n"
            f"   - **hours_high**: pessimistic estimate (with setbacks)\n"
            f"   - **confidence**: low/medium/high — how confident are you in the estimate?\n"
            f"   - **required_skills**: list of specific skills needed\n"
            f"   - **resource_types**: developer, designer, devops, qa, etc.\n"
            f"2. Perform bottom-up roll-up: each parent node's hours = sum of children\n"
            f"3. Identify resource contention:\n"
            f"   - Which parallelizable branches need the same scarce resource?\n"
            f"   - Where would adding more people NOT help (mythical man-month)?\n"
            f"4. Provide allocation recommendations\n\n"
            f"## Output Format\n"
            f'Write "resource_plan.json" with this structure:\n'
            f"```json\n"
            f'{{\n'
            f'  "task_estimates": [\n'
            f'    {{\n'
            f'      "task_id": 5,\n'
            f'      "task_title": "Task name",\n'
            f'      "hours_estimate": 8.0,\n'
            f'      "hours_low": 4.0,\n'
            f'      "hours_high": 16.0,\n'
            f'      "confidence": "medium",\n'
            f'      "required_skills": ["Python", "pytest"],\n'
            f'      "resource_types": ["developer"]\n'
            f'    }}\n'
            f'  ],\n'
            f'  "aggregated_estimates": {{\n'
            f'    "total_hours": 120.0,\n'
            f'    "total_hours_low": 80.0,\n'
            f'    "total_hours_high": 200.0,\n'
            f'    "by_resource_type": {{\n'
            f'      "developer": 100.0,\n'
            f'      "qa": 20.0\n'
            f'    }}\n'
            f'  }},\n'
            f'  "skill_matrix": {{\n'
            f'    "Python": {{"hours_needed": 60, "tasks": [5, 6, 7]}},\n'
            f'    "pytest": {{"hours_needed": 20, "tasks": [8, 9]}}\n'
            f'  }},\n'
            f'  "resource_contention": [\n'
            f'    {{\n'
            f'      "resource": "Python senior dev",\n'
            f'      "contested_tasks": [5, 6],\n'
            f'      "parallelizable": true,\n'
            f'      "recommendation": "Assign one dev per task with shared code review"\n'
            f'    }}\n'
            f'  ],\n'
            f'  "allocation_recommendations": ["Recommendation 1", "Recommendation 2"],\n'
            f'  "generated_at": "<ISO timestamp>"\n'
            f'}}\n'
            f"```\n\n"
            f"The JSON object MUST appear INLINE in your response. "
            f'Also use write_file to save a backup to "resource_plan.json".\n'
            f"Working directory: {working_dir}"
        )

        if backend == "claude_cli":
            backend_note = (
                "\n\nUse your own knowledge — do NOT search the web. "
                "Read task_tree.json first. Write resource_plan.json, "
                "then output TASK_COMPLETE."
            )
        else:
            backend_note = (
                "\n\nRead task_tree.json first. Write resource_plan.json, "
                "then output TASK_COMPLETE."
            )

        return common + backend_note

    def parse_result(self, result: AgentResult, working_dir: str = ".",
                     task_tree: dict[str, Any] | None = None, **context: Any) -> dict[str, Any]:
        """Extract resource plan from agent response, disk file, or fallback."""
        resource_plan: dict[str, Any] = {}

        # 1. Try agent messages
        for msg in reversed(result.messages):
            content = msg.content if hasattr(msg, "content") else str(msg)
            json_str = extract_json_object(content)
            if json_str:
                try:
                    parsed = json.loads(json_str)
                    if "task_estimates" in parsed:
                        resource_plan = parsed
                        break
                except json.JSONDecodeError:
                    continue

        # 2. Try disk
        if not resource_plan:
            path = os.path.join(working_dir, "resource_plan.json")
            if os.path.isfile(path):
                try:
                    with open(path) as f:
                        resource_plan = json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass

        # 3. Fallback
        if not resource_plan:
            resource_plan = self._fallback_resource(task_tree, working_dir)
            out_path = os.path.join(working_dir, "resource_plan.json")
            try:
                with open(out_path, "w") as f:
                    json.dump(resource_plan, f, indent=2)
            except OSError:
                pass

        return resource_plan

    @staticmethod
    def _summarize_tree(task_tree: dict[str, Any] | None) -> str:
        """Create a short summary of the task tree for the prompt."""
        if not task_tree or not task_tree.get("tree"):
            return "(No task tree provided. Read task_tree.json from the working directory.)"

        tree = task_tree.get("tree", [])

        def _count_nodes(nodes):
            c = len(nodes)
            for n in nodes:
                c += _count_nodes(n.get("children", []))
            return c

        total = _count_nodes(tree)

        def _list_leaves(nodes):
            result = []
            for n in nodes:
                if not n.get("children"):
                    result.append(f"  - #{n.get('id', '?')} {n.get('title', '?')} "
                                  f"(complexity: {n.get('complexity', '?')})")
                result.extend(_list_leaves(n.get("children", [])))
            return result

        leaves = _list_leaves(tree)
        leaf_text = "\n".join(leaves[:20])
        if len(leaves) > 20:
            leaf_text += f"\n  ... and {len(leaves) - 20} more tasks"

        return (
            f"- Complexity: {task_tree.get('complexity_level', 'unknown')}\n"
            f"- Total nodes: {total}\n"
            f"- Leaf tasks ({len(leaves)}):\n{leaf_text}"
        )

    @staticmethod
    def _fallback_resource(task_tree: dict[str, Any] | None,
                           working_dir: str = ".") -> dict[str, Any]:
        """Build a default resource plan with conservative estimates."""
        task_estimates: list[dict[str, Any]] = []
        total_hours = 0.0
        by_type: dict[str, float] = {"developer": 0.0}
        skill_matrix: dict[str, dict[str, Any]] = {}

        if task_tree and task_tree.get("tree"):

            def _extract_leaves(nodes):
                nonlocal total_hours
                result = []
                for n in nodes:
                    if not n.get("children"):
                        complexity = n.get("complexity", 3)
                        # Rough estimate: 2-8 hours per complexity point
                        hours = complexity * 3.0
                        hours_low = hours * 0.5
                        hours_high = hours * 2.0
                        total_hours += hours
                        by_type["developer"] = by_type.get("developer", 0) + hours

                        result.append({
                            "task_id": n.get("id"),
                            "task_title": n.get("title", ""),
                            "hours_estimate": hours,
                            "hours_low": hours_low,
                            "hours_high": hours_high,
                            "confidence": "low",
                            "required_skills": ["general programming"],
                            "resource_types": ["developer"],
                        })
                    result.extend(_extract_leaves(n.get("children", [])))
                return result

            task_estimates = _extract_leaves(task_tree.get("tree", []))

        return {
            "_fallback": True,
            "task_estimates": task_estimates,
            "aggregated_estimates": {
                "total_hours": total_hours,
                "total_hours_low": total_hours * 0.5,
                "total_hours_high": total_hours * 2.0,
                "by_resource_type": by_type,
            },
            "skill_matrix": skill_matrix,
            "resource_contention": [],
            "allocation_recommendations": [
                "Fallback estimates — manual review recommended.",
                "Consider adding 20-30% buffer for unknowns.",
            ],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
