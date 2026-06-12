"""PlanRiskAssessorRole — risk assessment agent for implementation planning."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from tools import ToolRegistry
from agent import AgentResult, AgentRole
from utils import extract_json_object


class PlanRiskAssessorRole(AgentRole):
    """Evaluates every task for risk across five dimensions.

    Dimensions: technical complexity, dependency risk, knowledge gap risk,
    integration risk, and schedule risk. Each task receives a 1-10 score per
    dimension and an aggregate risk level (low/medium/high/critical).

    High-risk tasks get specific mitigation recommendations:
    - Prototyping for technical uncertainty
    - Spike research for knowledge gaps
    - Pair programming for complex logic
    - Early integration testing for integration risk
    """

    agent_name: str = "risk_assessor"
    max_steps: int = 15

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        """Return Read-only tool specs."""
        if hasattr(ToolRegistry, "plan_risk_assessor_tools"):
            return ToolRegistry.to_dicts(ToolRegistry.plan_risk_assessor_tools())
        return ToolRegistry.to_dicts([ToolRegistry.READ_FILE])

    def build_task(self, backend: str, working_dir: str = ".",
                   task_tree: dict[str, Any] | None = None, **context: Any) -> str:
        """Build the risk assessment prompt."""
        tree_summary = self._summarize_tree(task_tree)

        common = (
            f"You are a risk assessment expert. Your job is to evaluate every "
            f"task in a task tree for risk across five dimensions and provide "
            f"mitigation recommendations for high-risk items.\n\n"
            f"## Task Tree Summary\n{tree_summary}\n\n"
            f"## Risk Dimensions\n"
            f"For each leaf task, assess these five dimensions (1-10 each):\n"
            f"1. **technical_complexity**: How technically challenging is this task?\n"
            f"2. **dependency_risk**: How much does this task depend on others? "
            f"Do dependencies have their own risks?\n"
            f"3. **knowledge_gap_risk**: How familiar is the team likely to be "
            f"with the required technologies/domain?\n"
            f"4. **integration_risk**: How complex are the integration points "
            f"with other components/systems?\n"
            f"5. **schedule_risk**: How likely is this task to exceed its "
            f"estimated timeline?\n\n"
            f"## Risk Levels\n"
            f"- **low**: aggregate score < 15 (minimal concern)\n"
            f"- **medium**: aggregate 15-25 (some concern, standard mitigation)\n"
            f"- **high**: aggregate 26-35 (significant concern, specific mitigation needed)\n"
            f"- **critical**: aggregate > 35 (severe concern, must address before proceeding)\n\n"
            f"## Mitigation Strategies\n"
            f"For tasks with high or critical risk, recommend specific strategies:\n"
            f"- Prototyping / proof-of-concept (for technical uncertainty)\n"
            f"- Spike research / knowledge building (for knowledge gaps)\n"
            f"- Pair programming / code review (for complex logic)\n"
            f"- Early integration testing / CI/CD (for integration risk)\n"
            f"- Buffer time / parallel exploration (for schedule risk)\n\n"
            f"## Output Format\n"
            f'Write "risk_matrix.json" with this structure:\n'
            f"```json\n"
            f'{{\n'
            f'  "task_risks": [\n'
            f'    {{\n'
            f'      "task_id": 5,\n'
            f'      "task_title": "Implement core functionality",\n'
            f'      "scores": {{\n'
            f'        "technical_complexity": 7,\n'
            f'        "dependency_risk": 5,\n'
            f'        "knowledge_gap_risk": 4,\n'
            f'        "integration_risk": 6,\n'
            f'        "schedule_risk": 5\n'
            f'      }},\n'
            f'      "aggregate_score": 27,\n'
            f'      "risk_level": "high",\n'
            f'      "mitigation": ["Prototype core algorithm first", '
            f'"Schedule pair programming sessions"]\n'
            f'    }}\n'
            f'  ],\n'
            f'  "overall_risk_level": "medium",\n'
            f'  "risk_heatmap_summary": "Brief summary of the risk landscape.",\n'
            f'  "generated_at": "<ISO timestamp>"\n'
            f'}}\n'
            f"```\n\n"
            f"The JSON object MUST appear INLINE in your response. "
            f'Also use write_file to save a backup to "risk_matrix.json".\n'
            f"Working directory: {working_dir}"
        )

        if backend == "claude_cli":
            backend_note = (
                "\n\nUse your own knowledge — do NOT search the web. "
                "Read task_tree.json first. Write risk_matrix.json, "
                "then output TASK_COMPLETE."
            )
        else:
            backend_note = (
                "\n\nRead task_tree.json first. Write risk_matrix.json, "
                "then output TASK_COMPLETE."
            )

        return common + backend_note

    def parse_result(self, result: AgentResult, working_dir: str = ".",
                     task_tree: dict[str, Any] | None = None, **context: Any) -> dict[str, Any]:
        """Extract risk matrix from agent response, disk file, or fallback."""
        risk_matrix: dict[str, Any] = {}

        # 1. Try agent messages
        for msg in reversed(result.messages):
            content = msg.content if hasattr(msg, "content") else str(msg)
            json_str = extract_json_object(content)
            if json_str:
                try:
                    parsed = json.loads(json_str)
                    if "task_risks" in parsed:
                        risk_matrix = parsed
                        break
                except json.JSONDecodeError:
                    continue

        # 2. Try disk
        if not risk_matrix:
            path = os.path.join(working_dir, "risk_matrix.json")
            if os.path.isfile(path):
                try:
                    with open(path) as f:
                        risk_matrix = json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass

        # 3. Fallback
        if not risk_matrix:
            risk_matrix = self._fallback_risk(task_tree, working_dir)
            out_path = os.path.join(working_dir, "risk_matrix.json")
            try:
                with open(out_path, "w") as f:
                    json.dump(risk_matrix, f, indent=2)
            except OSError:
                pass

        return risk_matrix

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

        def _list_leaves(nodes, depth=0):
            result = []
            for n in nodes:
                children = n.get("children", [])
                if not children:
                    result.append(f"  - #{n.get('id', '?')} {n.get('title', '?')} "
                                  f"(complexity: {n.get('complexity', '?')})")
                result.extend(_list_leaves(children, depth + 1))
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
    def _fallback_risk(task_tree: dict[str, Any] | None,
                       working_dir: str = ".") -> dict[str, Any]:
        """Build a default risk matrix with conservative estimates."""
        task_risks: list[dict[str, Any]] = []
        if task_tree and task_tree.get("tree"):

            def _extract_leaves(nodes):
                result = []
                for n in nodes:
                    if not n.get("children"):
                        complexity = n.get("complexity", 3)
                        # Scale risk dimensions proportional to complexity
                        tc = min(10, complexity + 1)
                        dr = min(10, len(n.get("dependencies", [])) * 2 + 1)
                        agg = tc + dr + 3 + 3 + 3  # conservative defaults for other dims
                        level = "low"
                        if agg > 35:
                            level = "critical"
                        elif agg > 25:
                            level = "high"
                        elif agg > 15:
                            level = "medium"
                        result.append({
                            "task_id": n.get("id"),
                            "task_title": n.get("title", ""),
                            "scores": {
                                "technical_complexity": tc,
                                "dependency_risk": dr,
                                "knowledge_gap_risk": 3,
                                "integration_risk": 3,
                                "schedule_risk": 3,
                            },
                            "aggregate_score": agg,
                            "risk_level": level,
                            "mitigation": ["Review risk manually — fallback estimates used."]
                            if level in ("high", "critical") else [],
                        })
                    result.extend(_extract_leaves(n.get("children", [])))
                return result

            task_risks = _extract_leaves(task_tree.get("tree", []))

        high_count = sum(1 for r in task_risks if r["risk_level"] in ("high", "critical"))
        overall = "low"
        if high_count > len(task_risks) * 0.3:
            overall = "high"
        elif high_count > len(task_risks) * 0.1:
            overall = "medium"

        return {
            "_fallback": True,
            "task_risks": task_risks,
            "overall_risk_level": overall,
            "risk_heatmap_summary": (
                f"Fallback risk assessment: {len(task_risks)} tasks analyzed, "
                f"{high_count} high/critical risks. "
                "Manual review recommended."
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
