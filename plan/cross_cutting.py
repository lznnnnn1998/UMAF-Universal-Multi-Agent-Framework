"""PlanCrossCuttingAnalyzerRole — cross-cutting concern identification agent."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from tools import ToolRegistry
from agent import AgentResult, AgentRole
from utils import extract_json_object


class PlanCrossCuttingAnalyzerRole(AgentRole):
    """Identifies orthogonal concerns spanning multiple task tree branches.

    Covers five concern domains:
    - Security: auth, data protection, threat modeling
    - Testing strategy: unit, integration, e2e, performance
    - Documentation: API docs, user guides, ADRs
    - Deployment: CI/CD, infrastructure, monitoring
    - Compliance: regulatory, audit trails

    Maps each concern to specific affected tasks with acceptance criteria.
    Identifies tasks that should be added to the tree to address gaps.
    """

    agent_name: str = "cross_cutting_analyzer"
    max_steps: int = 15

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        """Return Read-only tool specs."""
        if hasattr(ToolRegistry, "plan_cross_cutting_analyzer_tools"):
            return ToolRegistry.to_dicts(
                ToolRegistry.plan_cross_cutting_analyzer_tools()
            )
        return ToolRegistry.to_dicts([ToolRegistry.READ_FILE])

    def build_task(self, backend: str, working_dir: str = ".",
                   task_tree: dict[str, Any] | None = None, **context: Any) -> str:
        """Build the cross-cutting analysis prompt."""
        tree_summary = self._summarize_tree(task_tree)

        common = (
            f"You are a cross-cutting concern analyst. Your job is to identify "
            f"concerns that span multiple task tree branches — things that affect "
            f"or are affected by many tasks across the plan.\n\n"
            f"## Task Tree Summary\n{tree_summary}\n\n"
            f"## Concern Domains to Analyze\n"
            f"1. **Security** — Authentication, authorization, data protection, "
            f"encryption, threat modeling, vulnerability scanning, secret management\n"
            f"2. **Testing Strategy** — Unit tests, integration tests, e2e tests, "
            f"performance/load tests, security tests, test data management\n"
            f"3. **Documentation** — API documentation, architecture decision "
            f"records (ADRs), user guides, developer guides, runbooks\n"
            f"4. **Deployment** — CI/CD pipeline, infrastructure as code, "
            f"containerization, monitoring/alerting, logging, feature flags\n"
            f"5. **Compliance** — Regulatory requirements (GDPR, SOC2, HIPAA), "
            f"audit trails, data retention, accessibility, licensing\n\n"
            f"## Instructions\n"
            f"For each concern domain:\n"
            f"1. List specific concerns relevant to this project\n"
            f"2. Map each concern to the task IDs it affects\n"
            f"3. Define acceptance criteria for each concern\n"
            f"4. Identify GAPS: tasks that should exist in the tree but don't\n"
            f"   (e.g., missing security review for a PII-handling feature)\n\n"
            f"## Output Format\n"
            f'Write "cross_cutting_concerns.json" with this structure:\n'
            f"```json\n"
            f'{{\n'
            f'  "concerns": [\n'
            f'    {{\n'
            f'      "domain": "security",\n'
            f'      "concern": "API authentication",\n'
            f'      "description": "All API endpoints must require valid JWT tokens.",\n'
            f'      "affected_task_ids": [3, 4, 5],\n'
            f'      "acceptance_criteria": [\n'
            f'        "All endpoints return 401 for unauthenticated requests",\n'
            f'        "Token expiry is enforced at 24h"\n'
            f'      ],\n'
            f'      "severity": "high"\n'
            f'    }}\n'
            f'  ],\n'
            f'  "gap_tasks": [\n'
            f'    {{\n'
            f'      "title": "Security review for PII handling",\n'
            f'      "type": "task",\n'
            f'      "description": "Conduct security review of all PII data flows.",\n'
            f'      "complexity": 6,\n'
            f'      "suggested_parent_id": 3,\n'
            f'      "reason": "Task tree has PII-handling features but no security review"\n'
            f'    }}\n'
            f'  ],\n'
            f'  "domain_summaries": {{\n'
            f'    "security": "3 concerns identified, 1 gap found.",\n'
            f'    "testing_strategy": "2 concerns identified, 0 gaps.",\n'
            f'    "documentation": "1 concern identified, 2 gaps found.",\n'
            f'    "deployment": "2 concerns identified, 0 gaps.",\n'
            f'    "compliance": "0 concerns identified, 0 gaps."\n'
            f'  }},\n'
            f'  "generated_at": "<ISO timestamp>"\n'
            f'}}\n'
            f"```\n\n"
            f"The JSON object MUST appear INLINE in your response. "
            f'Also use write_file to save to "cross_cutting_concerns.json".\n'
            f"Working directory: {working_dir}"
        )

        if backend == "claude_cli":
            backend_note = (
                "\n\nUse your own knowledge — do NOT search the web. "
                "Read task_tree.json first. Write cross_cutting_concerns.json, "
                "then output TASK_COMPLETE."
            )
        else:
            backend_note = (
                "\n\nRead task_tree.json first. Write cross_cutting_concerns.json, "
                "then output TASK_COMPLETE."
            )

        return common + backend_note

    def parse_result(self, result: AgentResult, working_dir: str = ".",
                     task_tree: dict[str, Any] | None = None, **context: Any) -> dict[str, Any]:
        """Extract cross-cutting concerns from agent response, disk, or fallback."""
        concerns: dict[str, Any] = {}

        # 1. Try agent messages
        for msg in reversed(result.messages):
            content = msg.content if hasattr(msg, "content") else str(msg)
            json_str = extract_json_object(content)
            if json_str:
                try:
                    parsed = json.loads(json_str)
                    if "concerns" in parsed:
                        concerns = parsed
                        break
                except json.JSONDecodeError:
                    continue

        # 2. Try disk
        if not concerns:
            path = os.path.join(working_dir, "cross_cutting_concerns.json")
            if os.path.isfile(path):
                try:
                    with open(path) as f:
                        concerns = json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass

        # 3. Fallback
        if not concerns:
            concerns = self._fallback_cross_cutting(task_tree, working_dir)
            out_path = os.path.join(working_dir, "cross_cutting_concerns.json")
            try:
                with open(out_path, "w") as f:
                    json.dump(concerns, f, indent=2)
            except OSError:
                pass

        return concerns

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

        def _list_all(nodes):
            result = []
            for n in nodes:
                result.append(f"  - #{n.get('id', '?')} [{n.get('type', 'task')}] "
                              f"{n.get('title', '?')}")
                result.extend(_list_all(n.get("children", [])))
            return result

        items = _list_all(tree)
        item_text = "\n".join(items[:20])
        if len(items) > 20:
            item_text += f"\n  ... and {len(items) - 20} more nodes"

        return (
            f"- Complexity: {task_tree.get('complexity_level', 'unknown')}\n"
            f"- Total nodes: {total}\n"
            f"- Tree structure:\n{item_text}"
        )

    @staticmethod
    def _fallback_cross_cutting(task_tree: dict[str, Any] | None,
                                working_dir: str = ".") -> dict[str, Any]:
        """Build a default cross-cutting analysis with standard concerns."""
        domains = ["security", "testing_strategy", "documentation", "deployment",
                   "compliance"]
        summaries: dict[str, str] = {}
        for d in domains:
            summaries[d] = "No analysis available — fallback generated."

        return {
            "_fallback": True,
            "concerns": [
                {
                    "domain": "testing_strategy",
                    "concern": "Test coverage for all implementation tasks",
                    "description": "Every implementation task should have "
                                   "corresponding unit and integration tests.",
                    "affected_task_ids": [],
                    "acceptance_criteria": [
                        "All new code has test coverage",
                        "Tests pass in CI before merge",
                    ],
                    "severity": "medium",
                },
                {
                    "domain": "documentation",
                    "concern": "Implementation documentation",
                    "description": "Document all new code, APIs, and "
                                   "architectural decisions.",
                    "affected_task_ids": [],
                    "acceptance_criteria": [
                        "README updated with new features",
                        "API docs generated for new endpoints",
                    ],
                    "severity": "medium",
                },
                {
                    "domain": "deployment",
                    "concern": "CI/CD pipeline integration",
                    "description": "New code must integrate with existing "
                                   "CI/CD pipeline.",
                    "affected_task_ids": [],
                    "acceptance_criteria": [
                        "All tests pass in CI",
                        "Dependencies are properly declared",
                    ],
                    "severity": "medium",
                },
            ],
            "gap_tasks": [],
            "domain_summaries": summaries,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
