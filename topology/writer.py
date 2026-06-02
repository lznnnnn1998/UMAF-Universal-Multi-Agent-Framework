"""TopologyWriterRole — selects best topology and produces final spec + report."""

import json
import os
from typing import Any

from agent import AgentResult, AgentRole
from tools import ToolRegistry


class TopologyWriterRole(AgentRole):
    """Select the highest-scoring topology and produce the final output files.

    Writes two files:
    - topology_spec.json: Full specification with implementation guide
    - topology_report.md: Markdown report with comparison table and recommendation
    """

    agent_name: str = "topology_writer"
    max_steps: int = 8

    # ── Tools ───────────────────────────────────────────────────────────

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        """Return Write-only tool specs."""
        return ToolRegistry.to_dicts(
            ToolRegistry.topology_writer_tools()
        )

    # ── Task prompt ─────────────────────────────────────────────────────

    def build_task(self, backend: str, evaluated_topologies: list[dict[str, Any]] | None = None,
                   candidate_topologies: list[dict[str, Any]] | None = None,
                   input_spec: str = "", working_dir: str = ".",
                   **context: Any) -> str:
        """Build the writer prompt with backend-aware instructions."""
        eval_text = self._format_evaluated(evaluated_topologies)

        # Determine the best topology name for the prompt
        best_name = ""
        if evaluated_topologies:
            best_name = evaluated_topologies[0].get("name", "")

        common = (
            f"You are a topology documentation expert. Your job is to select "
            f"the best agent topology and produce the final specification and "
            f"comparison report.\n\n"
            f"## Requirement\n{input_spec}\n\n"
            f"## Evaluated Topologies (ranked by score)\n{eval_text}\n\n"
            f"## Task\n"
            f"The highest-scoring topology is **{best_name}**. Produce TWO "
            f"output files:\n\n"
            f"### 1. topology_spec.json\n"
            f"A complete specification for implementing the pipeline:\n"
            f"```json\n"
            f"{{\n"
            f'  "pipeline_name": "topology",\n'
            f'  "recommended_topology": "name of best topology",\n'
            f'  "total_score": 42,\n'
            f'  "max_possible_score": 50,\n'
            f'  "design_pattern": "fan_out_fan_in",\n'
            f'  "agents": [...],\n'
            f'  "connections": [...],\n'
            f'  "parallelism_strategy": "...",\n'
            f'  "pipeline_implementation_guide": {{\n'
            f'    "overview": "High-level description of the pipeline flow.",\n'
            f'    "nodes": [\n'
            f'      {{"name": "node_name", "role": "TopologyXxxRole", '
            f'"description": "...", "transitions_to": ["next_node"]}}\n'
            f'    ],\n'
            f'    "flow_diagram": "ASCII art showing the node graph",\n'
            f'    "key_design_decisions": ["decision 1", "decision 2"],\n'
            f'    "configuration_notes": "Any special config needed."\n'
            f'  }}\n'
            f"}}\n"
            f"```\n\n"
            f"### 2. topology_report.md\n"
            f"A markdown report containing:\n"
            f"- Title and requirement summary\n"
            f"- Comparison table of all candidate topologies with scores\n"
            f"- Recommendation section explaining why the best was chosen\n"
            f"- Implementation notes\n"
            f"- Next steps\n\n"
            f"Write both files using write_file, then output TASK_COMPLETE.\n"
            f'Working directory: {working_dir}'
        )

        if backend == "claude_cli":
            backend_note = (
                "\n\nUse your own knowledge — do NOT search the web. "
                "Read evaluated_topologies.json first if it exists. Write "
                "topology_spec.json and topology_report.md, then output "
                "TASK_COMPLETE."
            )
        else:
            backend_note = (
                "\n\nRead evaluated_topologies.json first if it exists. Write "
                "topology_spec.json and topology_report.md, then output "
                "TASK_COMPLETE."
            )

        return common + backend_note

    # ── Parse result ────────────────────────────────────────────────────

    def parse_result(self, result: AgentResult, working_dir: str,
                     evaluated_topologies: list[dict[str, Any]] | None = None,
                     candidate_topologies: list[dict[str, Any]] | None = None,
                     input_spec: str = "", **context: Any) -> dict[str, Any]:
        """Verify output files exist; generate fallback files if missing.

        Returns a dict with keys: spec_path, report_path, spec (loaded JSON),
        and success (bool).
        """
        spec_path = os.path.join(working_dir, "topology_spec.json")
        report_path = os.path.join(working_dir, "topology_report.md")

        spec_exists = os.path.isfile(spec_path)
        report_exists = os.path.isfile(report_path)

        spec_data: dict[str, Any] = {}

        # If spec doesn't exist, generate a fallback
        if not spec_exists:
            spec_data = self._generate_spec(
                evaluated_topologies or [],
                candidate_topologies or [],
                input_spec,
            )
            try:
                with open(spec_path, "w") as f:
                    json.dump(spec_data, f, indent=2)
                spec_exists = True
            except OSError:
                pass
        else:
            try:
                with open(spec_path) as f:
                    spec_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                spec_data = {}

        # If report doesn't exist, generate a fallback
        if not report_exists:
            report_content = self._generate_report(
                evaluated_topologies or [],
                spec_data,
                input_spec,
            )
            try:
                with open(report_path, "w") as f:
                    f.write(report_content)
                report_exists = True
            except OSError:
                pass

        return {
            "spec_path": spec_path,
            "report_path": report_path,
            "spec": spec_data,
            "success": spec_exists and report_exists,
        }

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _format_evaluated(evaluated: list[dict[str, Any]] | None) -> str:
        """Format evaluated topologies as a markdown table in the prompt."""
        if not evaluated:
            return "(No evaluated topologies available.)"

        lines = ["| # | Topology | Latency | Reliability | Cost | Simplicity | Scalability | Total |",
                  "|---|----------|---------|-------------|------|------------|-------------|-------|"]
        for i, t in enumerate(evaluated):
            name = t.get("name", "?")
            scores = t.get("scores", {})
            s = lambda d: scores.get(d, {}).get("score", "?")
            total = t.get("total_score", "?")
            lines.append(
                f"| {i+1} | {name} | {s('latency')} | {s('reliability')} | "
                f"{s('cost_efficiency')} | {s('simplicity')} | "
                f"{s('scalability')} | **{total}** |"
            )
        return "\n".join(lines)

    @staticmethod
    def _generate_spec(
        evaluated: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        input_spec: str,
    ) -> dict[str, Any]:
        """Generate a topology_spec.json from evaluated data."""
        if not evaluated and not candidates:
            return {
                "pipeline_name": "topology",
                "recommended_topology": "Unknown",
                "total_score": 0,
                "max_possible_score": 50,
                "design_pattern": "sequential",
                "agents": [],
                "connections": [],
                "parallelism_strategy": "none",
                "pipeline_implementation_guide": {
                    "overview": "Fallback spec — LLM output was not available. Review manually.",
                    "nodes": [],
                    "flow_diagram": "N/A",
                    "key_design_decisions": ["Manual review required"],
                    "configuration_notes": "Generated by fallback writer.",
                },
            }

        best = evaluated[0] if evaluated else candidates[0] if candidates else {}
        # Find matching candidate topology for full details
        full_topo = {}
        best_name = best.get("name", "")
        for c in candidates:
            if c.get("name") == best_name:
                full_topo = c
                break
        if not full_topo and candidates:
            full_topo = candidates[0]

        agents = full_topo.get("agents", [])
        connections = full_topo.get("connections", [])
        pattern = full_topo.get("pattern", "sequential")

        # Build implementation guide nodes from agents
        nodes = []
        agent_names = [a.get("agent_name", "") for a in agents]
        for i, agent in enumerate(agents):
            next_nodes = []
            for conn in connections:
                if conn.get("from") == agent.get("agent_name"):
                    next_nodes.append(conn.get("to", ""))
            nodes.append({
                "name": agent.get("agent_name", f"agent_{i}"),
                "role": agent.get("role_type", "worker"),
                "description": agent.get("role_type", ""),
                "transitions_to": next_nodes,
            })

        # Build ASCII flow diagram
        flow = TopologyWriterRole._build_flow_diagram(pattern, agents, connections)

        return {
            "pipeline_name": "topology",
            "recommended_topology": best_name,
            "total_score": best.get("total_score", 0),
            "max_possible_score": 50,
            "design_pattern": pattern,
            "agents": agents,
            "connections": connections,
            "parallelism_strategy": full_topo.get("parallelism_strategy", "none"),
            "pipeline_implementation_guide": {
                "overview": (
                    f"A {pattern} pipeline implementing the topology optimizer "
                    f"for: {input_spec[:150]}"
                ),
                "nodes": nodes,
                "flow_diagram": flow,
                "key_design_decisions": [
                    f"Selected {pattern} pattern based on evaluation scores.",
                    f"Pipeline has {len(agents)} agents and {len(connections)} connections.",
                ],
                "configuration_notes": (
                    "All agents use backend-aware tool selection. "
                    "Timeout values follow UMAF defaults unless overridden."
                ),
            },
        }

    @staticmethod
    def _build_flow_diagram(pattern: str, agents: list[dict[str, Any]],
                            connections: list[dict[str, Any]]) -> str:
        """Build an ASCII flow diagram from agents and connections."""
        if not agents:
            return "No agents defined."

        agent_names = [a.get("agent_name", f"agent_{i}") for i, a in enumerate(agents)]

        if pattern == "sequential":
            parts = []
            for i, name in enumerate(agent_names):
                parts.append(f"[{name}]")
                if i < len(agent_names) - 1:
                    parts.append("→")
            return " " + " ".join(parts)

        if pattern == "fan_out_fan_in":
            head = agent_names[0] if agent_names else "head"
            workers = agent_names[1:-1] if len(agent_names) > 2 else ["worker_1", "worker_2"]
            tail = agent_names[-1] if len(agent_names) > 1 else "collector"
            lines = [f"  [{head}]"]
            lines.append("  /  |  \\")
            lines.append(" " + "  ".join(f"[{w}]" for w in workers[:3]))
            lines.append("  \\  |  /")
            lines.append(f"  [{tail}]")
            return "\n".join(lines)

        if pattern == "debate_consensus":
            lines = ["  " + "  ".join(f"[{n}]" for n in agent_names[:3])]
            lines.append("   \\  |  /")
            lines.append(f"  [judge/synthesizer]")
            return "\n".join(lines)

        if pattern == "hierarchical":
            orch = agent_names[0] if agent_names else "orchestrator"
            mids = agent_names[1:3] if len(agent_names) > 2 else ["lead_a", "lead_b"]
            rest = agent_names[3:] if len(agent_names) > 3 else ["worker_a", "worker_b"]
            lines = [f"  [{orch}]"]
            lines.append("  /    \\")
            lines.append(f"[{mids[0]}]    [{mids[1] if len(mids) > 1 else 'lead_b'}]")
            lines.append(" |      |")
            lines.append(f"[{rest[0] if rest else 'w1'}]    [{rest[1] if len(rest) > 1 else 'w2'}]")
            return "\n".join(lines)

        # Generic: show connections
        lines = []
        for conn in connections:
            lines.append(f"  [{conn.get('from', '?')}] → [{conn.get('to', '?')}]")
        return "\n".join(lines) if lines else " ".join(f"[{n}]" for n in agent_names)

    @staticmethod
    def _generate_report(
        evaluated: list[dict[str, Any]],
        spec: dict[str, Any],
        input_spec: str,
    ) -> str:
        """Generate a topology_report.md from evaluated data."""
        best_name = spec.get("recommended_topology", "Unknown")
        total = spec.get("total_score", 0)
        pattern = spec.get("design_pattern", "unknown")

        # Build comparison table
        table = (
            "| # | Topology | Latency | Reliability | Cost Eff. | Simplicity |"
            " Scalability | **Total** |\n"
            "|---|----------|---------|-------------|-----------|------------|"
            "-------------|-----------|\n"
        )
        for i, t in enumerate(evaluated):
            name = t.get("name", "?")
            scores = t.get("scores", {})
            s = lambda d: scores.get(d, {}).get("score", "?")
            total_s = t.get("total_score", "?")
            table += (
                f"| {i+1} | {name} | {s('latency')} | {s('reliability')} | "
                f"{s('cost_efficiency')} | {s('simplicity')} | "
                f"{s('scalability')} | **{total_s}** |\n"
            )

        report = f"""# Topology Optimization Report

## Requirement
{input_spec}

## Comparison of Candidate Topologies

{table}

## Recommendation

**Selected Topology:** {best_name}
**Design Pattern:** {pattern}
**Total Score:** {total}/50

### Why This Topology Was Selected

The {best_name} topology was selected because it achieved the highest overall
score ({total}/50) across the five evaluation dimensions: latency, reliability,
cost efficiency, simplicity, and scalability. This topology represents the best
balance of performance, reliability, and implementation complexity for the
given requirement.

### Implementation Notes

- Pipeline name: `topology`
- Agents: {len(spec.get('agents', []))} total
- Connections: {len(spec.get('connections', []))} edges
- See `topology_spec.json` for the complete implementation guide including
  node definitions, flow diagram, and configuration notes.

### Next Steps

1. Review the implementation guide in `topology_spec.json`
2. Add the `TopologyState` TypedDict to `pipeline.py`
3. Implement the `TopologyPipeline` class extending `BasePipeline`
4. Register the new pipeline in `main.py`
5. Write integration tests for the full pipeline

---
*Report generated by TopologyWriterRole (UMAF v1.4)*
"""
        return report
