"""PlanWriterRole — plan synthesis and deliverable generation agent."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from tools import ToolRegistry
from agent import AgentResult, AgentRole
from utils import extract_json_object, safe_read


class PlanWriterRole(AgentRole):
    """Collector/aggregator consuming all 6 upstream outputs to produce final plan deliverables.

    Performs cross-output consistency checks:
    1. Risk-criticality alignment: critical path tasks should not have unmitigated high risk
    2. Cross-cutting coverage: resource estimates should include hours for cross-cutting tasks
    3. Contention detection: parallelizable branches should not exceed resource plan capacity

    Synthesizes:
    - plan_spec.json: machine-readable full plan specification
    - plan_report.md: human-readable executive summary with timeline, risk heatmap, etc.
    """

    agent_name: str = "plan_writer"
    max_steps: int = 20

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        """Return Read + Write tool specs."""
        if hasattr(ToolRegistry, "plan_writer_tools"):
            return ToolRegistry.to_dicts(ToolRegistry.plan_writer_tools())
        return ToolRegistry.to_dicts([
            ToolRegistry.READ_FILE, ToolRegistry.WRITE_FILE,
        ])

    def build_task(self, backend: str, working_dir: str = ".",
                   project_context: dict[str, Any] | None = None,
                   task_tree: dict[str, Any] | None = None,
                   dependency_graph: dict[str, Any] | None = None,
                   risk_matrix: dict[str, Any] | None = None,
                   resource_plan: dict[str, Any] | None = None,
                   cross_cutting_concerns: dict[str, Any] | None = None,
                   **context: Any) -> str:
        """Build the writer prompt with all upstream data summaries."""
        tc = self._summarize_task_tree(task_tree)
        dg = self._summarize_dep_graph(dependency_graph)
        rm = self._summarize_risk(risk_matrix)
        rp = self._summarize_resource(resource_plan)
        cc = self._summarize_cross_cutting(cross_cutting_concerns)

        common = (
            f"You are a technical writer and planning expert. Your job is to "
            f"collect all upstream analysis outputs, perform cross-output "
            f"consistency checks, and produce the final plan deliverables.\n\n"
            f"## Upstream Analysis Summary\n\n"
            f"### Task Tree\n{tc}\n\n"
            f"### Dependency Graph\n{dg}\n\n"
            f"### Risk Matrix\n{rm}\n\n"
            f"### Resource Plan\n{rp}\n\n"
            f"### Cross-Cutting Concerns\n{cc}\n\n"
            f"## Cross-Output Consistency Checks\n"
            f"Perform these checks before writing:\n"
            f"1. **Risk-Criticality Alignment**: Are any critical path tasks "
            f"rated high or critical risk without mitigation? Flag them.\n"
            f"2. **Cross-Cutting Coverage**: Do resource estimates include "
            f"hours for cross-cutting concern tasks (testing, docs, deployment)?\n"
            f"3. **Contention Detection**: Do parallelizable branches require "
            f"the same scarce resource? If so, flag as schedule risk.\n\n"
            f"## Deliverable 1: plan_spec.json\n"
            f"Complete machine-readable plan specification:\n"
            f"```json\n"
            f'{{\n'
            f'  "plan_title": "Implementation Plan for ...",\n'
            f'  "input_spec": "original task description",\n'
            f'  "generated_at": "<ISO timestamp>",\n'
            f'  "complexity_level": "medium",\n'
            f'  "task_tree": {{...}},\n'
            f'  "dependency_graph": {{...}},\n'
            f'  "risk_matrix": {{...}},\n'
            f'  "resource_plan": {{...}},\n'
            f'  "cross_cutting_concerns": {{...}},\n'
            f'  "milestones": [\n'
            f'    {{\n'
            f'      "name": "Milestone 1: Core Setup",\n'
            f'      "tasks": [1, 2, 3],\n'
            f'      "target_hours": 20,\n'
            f'      "depends_on": []\n'
            f'    }}\n'
            f'  ],\n'
            f'  "timeline": {{\n'
            f'    "total_estimated_hours": 120,\n'
            f'    "recommended_team_size": 2,\n'
            f'    "estimated_calendar_weeks": 4\n'
            f'  }},\n'
            f'  "consistency_checks": {{\n'
            f'    "risk_criticality_issues": [],\n'
            f'    "cross_cutting_coverage_gaps": [],\n'
            f'    "contention_warnings": []\n'
            f'  }}\n'
            f'}}\n'
            f"```\n\n"
            f"## Deliverable 2: plan_report.md\n"
            f"Human-readable report containing:\n"
            f"- Executive summary (2-3 paragraphs)\n"
            f"- Milestone timeline with task breakdown\n"
            f"- Risk heatmap (table of top risks with severity and mitigation)\n"
            f"- Resource allocation table (role, hours, tasks)\n"
            f"- Cross-cutting concern checklist\n"
            f"- Next steps and recommendations\n\n"
            f"Write both files, then output TASK_COMPLETE.\n"
            f"Working directory: {working_dir}"
        )

        if backend == "claude_cli":
            backend_note = (
                "\n\nUse your own knowledge — do NOT search the web. "
                "Read all input JSON files from the working directory first. "
                "Write plan_spec.json and plan_report.md, "
                "then output TASK_COMPLETE."
            )
        else:
            backend_note = (
                "\n\nRead all input JSON files from the working directory first. "
                "Write plan_spec.json and plan_report.md, "
                "then output TASK_COMPLETE."
            )

        return common + backend_note

    def parse_result(self, result: AgentResult, working_dir: str = ".",
                     project_context: dict[str, Any] | None = None,
                     task_tree: dict[str, Any] | None = None,
                     dependency_graph: dict[str, Any] | None = None,
                     risk_matrix: dict[str, Any] | None = None,
                     resource_plan: dict[str, Any] | None = None,
                     cross_cutting_concerns: dict[str, Any] | None = None,
                     **context: Any) -> dict[str, Any]:
        """Verify output files exist; generate fallback files if missing."""
        spec_path = os.path.join(working_dir, "plan_spec.json")
        report_path = os.path.join(working_dir, "plan_report.md")

        spec_exists = os.path.isfile(spec_path)
        report_exists = os.path.isfile(report_path)

        # Try to extract plan_spec from agent messages if not on disk
        spec_data: dict[str, Any] = {}
        if not spec_exists:
            for msg in reversed(result.messages):
                content = msg.content if hasattr(msg, "content") else str(msg)
                json_str = extract_json_object(content)
                if json_str:
                    try:
                        parsed = json.loads(json_str)
                        if isinstance(parsed, dict) and "plan_title" in parsed:
                            spec_data = parsed
                            break
                    except json.JSONDecodeError:
                        continue

        # If no spec data from agent, generate fallback
        if not spec_data and not spec_exists:
            spec_data = self._fallback_writer(
                working_dir, task_tree=task_tree,
                dependency_graph=dependency_graph,
                risk_matrix=risk_matrix, resource_plan=resource_plan,
                cross_cutting_concerns=cross_cutting_concerns,
            )

        # Write spec to disk if needed
        if spec_data and not spec_exists:
            self._write_plan_spec(working_dir, spec_data)
            spec_exists = True
        elif spec_exists and not spec_data:
            try:
                with open(spec_path) as f:
                    spec_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                spec_data = {}

        # Generate fallback report if missing
        if not report_exists:
            self._fallback_report_md(
                working_dir, task_tree=task_tree,
                dependency_graph=dependency_graph,
                risk_matrix=risk_matrix, resource_plan=resource_plan,
                cross_cutting_concerns=cross_cutting_concerns,
                spec=spec_data,
            )
            report_exists = True

        return {
            "spec_path": spec_path,
            "report_path": report_path,
            "spec": spec_data,
            "success": spec_exists and report_exists,
        }

    # -- Summary helpers for prompt --

    @staticmethod
    def _summarize_task_tree(task_tree: dict[str, Any] | None) -> str:
        if not task_tree:
            return "(Not available)"
        complexity = task_tree.get("complexity_level", "?")
        total = task_tree.get("total_nodes", "?")
        return f"Complexity: {complexity}, Total nodes: {total}"

    @staticmethod
    def _summarize_dep_graph(dep_graph: dict[str, Any] | None) -> str:
        if not dep_graph:
            return "(Not available)"
        nodes = len(dep_graph.get("nodes", []))
        edges = len(dep_graph.get("edges", []))
        cp = dep_graph.get("critical_path", {})
        cp_nodes = len(cp.get("nodes", []))
        acyclic = dep_graph.get("is_acyclic", "?")
        return (
            f"{nodes} nodes, {edges} edges, "
            f"Critical path: {cp_nodes} nodes, Acyclic: {acyclic}"
        )

    @staticmethod
    def _summarize_risk(risk_matrix: dict[str, Any] | None) -> str:
        if not risk_matrix:
            return "(Not available)"
        overall = risk_matrix.get("overall_risk_level", "?")
        tasks = len(risk_matrix.get("task_risks", []))
        return f"Overall risk: {overall}, {tasks} tasks assessed"

    @staticmethod
    def _summarize_resource(resource_plan: dict[str, Any] | None) -> str:
        if not resource_plan:
            return "(Not available)"
        agg = resource_plan.get("aggregated_estimates", {})
        total = agg.get("total_hours", "?")
        tasks = len(resource_plan.get("task_estimates", []))
        return f"Total: {total} hours across {tasks} tasks"

    @staticmethod
    def _summarize_cross_cutting(concerns: dict[str, Any] | None) -> str:
        if not concerns:
            return "(Not available)"
        count = len(concerns.get("concerns", []))
        gaps = len(concerns.get("gap_tasks", []))
        return f"{count} concerns, {gaps} gap tasks identified"

    # -- Fallback methods --

    @staticmethod
    def _fallback_writer(working_dir: str = ".", **context: Any) -> dict[str, Any]:
        """Build a fallback plan_spec from available context."""
        task_tree = context.get("task_tree") or {}
        dep_graph = context.get("dependency_graph") or {}
        risk_matrix = context.get("risk_matrix") or {}
        resource_plan = context.get("resource_plan") or {}
        cross_cutting = context.get("cross_cutting_concerns") or {}

        return {
            "_fallback": True,
            "plan_title": "Implementation Plan (Fallback)",
            "input_spec": task_tree.get("input_spec", ""),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "complexity_level": task_tree.get("complexity_level", "unknown"),
            "task_tree": task_tree,
            "dependency_graph": dep_graph,
            "risk_matrix": risk_matrix,
            "resource_plan": resource_plan,
            "cross_cutting_concerns": cross_cutting,
            "milestones": [],
            "timeline": {
                "total_estimated_hours": resource_plan.get(
                    "aggregated_estimates", {}).get("total_hours", 0),
                "recommended_team_size": 2,
                "estimated_calendar_weeks": 2,
            },
            "consistency_checks": {
                "risk_criticality_issues": [],
                "cross_cutting_coverage_gaps": [],
                "contention_warnings": [],
            },
        }

    @staticmethod
    def _write_plan_spec(working_dir: str, spec: dict[str, Any]) -> None:
        """Write plan_spec.json to working_dir."""
        path = os.path.join(working_dir, "plan_spec.json")
        try:
            with open(path, "w") as f:
                json.dump(spec, f, indent=2, default=str)
        except OSError:
            pass

    @staticmethod
    def _fallback_report_md(working_dir: str = ".", **context: Any) -> None:
        """Generate a fallback plan_report.md from available context."""
        task_tree = context.get("task_tree") or {}
        risk_matrix = context.get("risk_matrix") or {}
        resource_plan = context.get("resource_plan") or {}
        spec = context.get("spec") or {}
        dep_graph = context.get("dependency_graph") or {}
        cross_cutting = context.get("cross_cutting_concerns") or {}

        input_spec = task_tree.get("input_spec", spec.get("input_spec", "N/A"))
        complexity = task_tree.get("complexity_level", spec.get("complexity_level", "unknown"))
        total_hours = resource_plan.get("aggregated_estimates", {}).get("total_hours",
                         spec.get("timeline", {}).get("total_estimated_hours", "N/A"))
        overall_risk = risk_matrix.get("overall_risk_level",
                        spec.get("risk_matrix", {}).get("overall_risk_level", "unknown"))
        total_nodes = task_tree.get("total_nodes", 0)
        total_edges = len(dep_graph.get("edges", []))
        concerns_count = len(cross_cutting.get("concerns", []))
        gaps_count = len(cross_cutting.get("gap_tasks", []))

        # Build risk heatmap rows
        risk_rows = ""
        task_risks = risk_matrix.get("task_risks", [])
        if task_risks:
            high_risks = [r for r in task_risks if r.get("risk_level") in ("high", "critical")]
            for r in high_risks[:10]:
                mitigation = "; ".join(r.get("mitigation", [])[:2])
                risk_rows += (
                    f"| {r.get('task_id', '?')} | {r.get('task_title', '?')} | "
                    f"{r.get('risk_level', '?')} | {r.get('aggregate_score', '?')} | "
                    f"{mitigation or 'None'} |\n"
                )
        if not risk_rows:
            risk_rows = "| - | No high/critical risks identified | - | - | - |\n"

        report = f"""# Implementation Plan Report

> **Status:** Fallback generated — manual review recommended

## Executive Summary

This implementation plan was generated for the following task:

> {input_spec[:200]}{"..." if len(input_spec) > 200 else ""}

**Complexity:** {complexity}
**Total Estimated Effort:** {total_hours} hours
**Overall Risk Level:** {overall_risk}
**Task Tree Nodes:** {total_nodes}

## Milestone Timeline

| Milestone | Tasks | Est. Hours | Dependencies |
|-----------|-------|-----------|--------------|
| Planning & Analysis | Analyze + Design tasks | ~{total_hours}/3 | None |
| Core Implementation | Implementation tasks | ~{total_hours}/3 | Planning |
| Testing & Validation | Test + Document tasks | ~{total_hours}/3 | Implementation |

## Risk Heatmap

| Task ID | Task | Risk Level | Score | Mitigation |
|---------|------|-----------|-------|------------|
{risk_rows}

## Resource Allocation

| Resource Type | Estimated Hours | Tasks |
|--------------|----------------|-------|
| Developer | {total_hours} | All |

## Cross-Cutting Concern Checklist

- [ ] **Security:** {concerns_count} concerns identified, {gaps_count} gaps found
- [ ] **Testing Strategy:** Tests written for all implementation tasks
- [ ] **Documentation:** README and API docs updated
- [ ] **Deployment:** CI/CD pipeline verified
- [ ] **Compliance:** Regulatory requirements reviewed

## Dependency Graph Summary

- **Total Nodes:** {total_nodes}
- **Total Edges:** {total_edges}
- **Critical Path:** {len(dep_graph.get("critical_path", {}).get("nodes", []))} nodes
- **Acyclic:** {dep_graph.get("is_acyclic", True)}

## Next Steps

1. Review and refine the task tree in `task_tree.json`
2. Adjust risk assessments and mitigation strategies in `risk_matrix.json`
3. Validate resource estimates in `resource_plan.json`
4. Update cross-cutting concerns in `cross_cutting_concerns.json`
5. Execute tasks in dependency order, starting with the critical path

---
*Report generated by PlanWriterRole (UMAF Plan Pipeline)*
"""

        path = os.path.join(working_dir, "plan_report.md")
        try:
            with open(path, "w") as f:
                f.write(report)
        except OSError:
            pass
