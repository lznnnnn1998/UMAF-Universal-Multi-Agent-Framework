"""TopologyEvaluatorRole — scores candidate topologies on 5 dimensions."""

import json
import os
from typing import Any

from agent import AgentResult, AgentRole
from tools import ToolRegistry
from utils import extract_json_array


class TopologyEvaluatorRole(AgentRole):
    """Score each candidate topology on 5 dimensions and produce a ranking.

    Dimensions: latency, reliability, cost_efficiency, simplicity, scalability.
    Each scored 1-10 with reasoning, producing a total_score and ranked list.
    """

    agent_name: str = "topology_evaluator"
    max_steps: int = 10

    _DIMENSIONS = [
        "latency",
        "reliability",
        "cost_efficiency",
        "simplicity",
        "scalability",
    ]

    # ── Tools ───────────────────────────────────────────────────────────

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        """Return Read + Write tool specs."""
        return ToolRegistry.to_dicts(
            ToolRegistry.topology_evaluator_tools()
        )

    # ── Task prompt ─────────────────────────────────────────────────────

    def build_task(self, backend: str, candidate_topologies: list[dict[str, Any]] | None = None,
                   working_dir: str = ".", **context: Any) -> str:
        """Build the evaluation prompt with backend-aware instructions."""
        topo_text = self._format_topologies(candidate_topologies)

        dims = "\n".join(
            f"  {i+1}. **{d}** — {self._dimension_guide(d)}"
            for i, d in enumerate(self._DIMENSIONS)
        )

        common = (
            f"You are a topology evaluation expert. Your job is to score "
            f"candidate agent topologies on 5 key dimensions and produce a "
            f"ranked evaluation.\n\n"
            f"## Candidate Topologies\n{topo_text}\n\n"
            f"## Evaluation Dimensions (each scored 1-10)\n{dims}\n\n"
            f"## Instructions\n"
            f"For each topology, assign a score (1-10) for each dimension "
            f"with 1-2 sentences of reasoning. Then compute a total_score "
            f"(sum of all 5 dimensions, max 50).\n\n"
            f"### Scoring Guidelines\n"
            f"- **latency**: Lower end-to-end time = higher score. Consider "
            f"parallelism and critical path length.\n"
            f"- **reliability**: Fault tolerance, retry ability, error isolation. "
            f"More redundancy = higher score.\n"
            f"- **cost_efficiency**: Fewer agent calls and tokens = higher score. "
            f"Consider API cost per run.\n"
            f"- **simplicity**: Easier to implement and maintain = higher score. "
            f"Prefer clear data flow.\n"
            f"- **scalability**: Ability to handle growing task size or sub-task count. "
            f"More parallelism headroom = higher score.\n\n"
            f"## Output Format\n"
            f'Write your evaluation as a JSON array to '
            f'"evaluated_topologies.json":\n'
            f"```json\n"
            f"[\n"
            f"  {{\n"
            f'    "name": "...",\n'
            f'    "scores": {{\n'
            f'      "latency": {{"score": 8, "reasoning": "..."}},\n'
            f'      "reliability": {{"score": 7, "reasoning": "..."}},\n'
            f'      "cost_efficiency": {{"score": 6, "reasoning": "..."}},\n'
            f'      "simplicity": {{"score": 9, "reasoning": "..."}},\n'
            f'      "scalability": {{"score": 7, "reasoning": "..."}}\n'
            f'    }},\n'
            f'    "total_score": 37,\n'
            f'    "overall_assessment": "..."\n'
            f'  }}\n'
            f']\n'
            f"```\n\n"
            f"The JSON array MUST appear INLINE in your response text. "
            f'Also use write_file to save a backup copy to '
            f'"evaluated_topologies.json".\n'
            f'Working directory: {working_dir}'
        )

        if backend == "claude_cli":
            backend_note = (
                "\n\nUse your own knowledge — do NOT search the web. "
                "Read candidate_topologies.json first. Write "
                "evaluated_topologies.json, then output TASK_COMPLETE."
            )
        else:
            backend_note = (
                "\n\nRead candidate_topologies.json first. Write "
                "evaluated_topologies.json, then output TASK_COMPLETE."
            )

        return common + backend_note

    # ── Parse result ────────────────────────────────────────────────────

    def parse_result(self, result: AgentResult, working_dir: str,
                     candidate_topologies: list[dict[str, Any]] | None = None,
                     **context: Any) -> list[dict[str, Any]]:
        """Extract evaluated topologies from agent response or disk file."""
        evaluated: list[dict[str, Any]] = []

        # 1. Try extracting JSON array from agent response messages
        for msg in reversed(result.messages):
            content = msg.content if hasattr(msg, "content") else str(msg)
            json_str = extract_json_array(content)
            if json_str:
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        evaluated = parsed
                        break
                except json.JSONDecodeError:
                    continue

        # 2. Try reading from disk
        if not evaluated:
            path = os.path.join(working_dir, "evaluated_topologies.json")
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        parsed = json.load(f)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        evaluated = parsed
                except (json.JSONDecodeError, OSError):
                    pass

        # 3. Fallback: score topologies with a heuristic
        if not evaluated:
            evaluated = self._fallback_evaluate(candidate_topologies or [])

        # Sort by total_score descending
        evaluated.sort(key=lambda t: t.get("total_score", 0), reverse=True)
        return evaluated

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _dimension_guide(dimension: str) -> str:
        """Return a short scoring guide for each dimension."""
        guides = {
            "latency": (
                "End-to-end execution time. Sequential=lower, parallel=higher. "
                "Critical path length matters."
            ),
            "reliability": (
                "Fault tolerance and error recovery. Independent workers isolate "
                "failures. Retry loops improve score."
            ),
            "cost_efficiency": (
                "Total tokens and API calls per run. Fewer agents and simpler "
                "prompts reduce cost."
            ),
            "simplicity": (
                "Ease of implementation and maintenance. Linear flows are simpler "
                "than complex DAGs."
            ),
            "scalability": (
                "Ability to handle larger/more tasks. Fan-out parallelism scales "
                "better than fixed pipelines."
            ),
        }
        return guides.get(dimension, "Score 1-10 with reasoning.")

    @staticmethod
    def _format_topologies(topologies: list[dict[str, Any]] | None) -> str:
        """Format candidate topologies for the prompt."""
        if not topologies:
            return "(No topologies provided — evaluate from the requirement context.)"

        lines = []
        for i, t in enumerate(topologies):
            name = t.get("name", f"Topology {i+1}")
            pattern = t.get("pattern", "unknown")
            desc = t.get("description", "")
            agents = t.get("agents", [])
            agent_names = [a.get("agent_name", "?") for a in agents]
            lines.append(
                f"### {i+1}. {name}\n"
                f"  Pattern: {pattern}\n"
                f"  Description: {desc}\n"
                f"  Agents: {', '.join(agent_names)} ({len(agents)} total)\n"
                f"  Connections: {len(t.get('connections', []))} edges\n"
            )
        return "\n".join(lines)

    @staticmethod
    def _fallback_evaluate(
        topologies: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Heuristic-based evaluation when LLM extraction fails.

        Scores are derived from topology pattern and structure:
        - Sequential: high simplicity, low latency/scalability
        - Fan-out: high latency/scalability, lower simplicity
        - Hierarchical: high scalability, lower cost/simplicity
        - Debate: high reliability, lower cost/latency
        """
        def _score_one(topo: dict[str, Any]) -> dict[str, Any]:
            pattern = topo.get("pattern", "sequential")
            agents = topo.get("agents", [])
            conns = topo.get("connections", [])
            n_agents = max(len(agents), 1)

            # Heuristic scoring based on pattern characteristics
            if pattern == "sequential":
                scores = {
                    "latency": {"score": 4, "reasoning": "Linear chain: total time is sum of all agent steps."},
                    "reliability": {"score": 5, "reasoning": "Single point of failure at each stage; no redundancy."},
                    "cost_efficiency": {"score": 8, "reasoning": f"Only {n_agents} agents; minimal token overhead."},
                    "simplicity": {"score": 10, "reasoning": "Simplest possible topology; trivial to implement."},
                    "scalability": {"score": 3, "reasoning": "Cannot add parallel workers; fixed throughput."},
                }
            elif pattern == "fan_out_fan_in":
                scores = {
                    "latency": {"score": 8, "reasoning": "Parallel workers reduce wall-clock time significantly."},
                    "reliability": {"score": 7, "reasoning": "Independent workers isolate failures; aggregator handles partial results."},
                    "cost_efficiency": {"score": 6, "reasoning": f"{n_agents} agents with moderate coordination overhead."},
                    "simplicity": {"score": 7, "reasoning": "Well-understood pattern; clear data flow."},
                    "scalability": {"score": 8, "reasoning": "Easy to add more workers for more sub-tasks."},
                }
            elif pattern == "debate_consensus":
                scores = {
                    "latency": {"score": 3, "reasoning": "Multiple agents run redundantly; no parallelism gain."},
                    "reliability": {"score": 10, "reasoning": "Consensus mechanism provides highest fault tolerance."},
                    "cost_efficiency": {"score": 3, "reasoning": f"Redundant execution costs {n_agents}x more."},
                    "simplicity": {"score": 5, "reasoning": "Consensus logic adds complexity to result synthesis."},
                    "scalability": {"score": 5, "reasoning": "Adding more debaters improves confidence but increases cost."},
                }
            elif pattern == "hierarchical":
                scores = {
                    "latency": {"score": 7, "reasoning": "Two-level parallelism; domains run concurrently."},
                    "reliability": {"score": 6, "reasoning": "Domain leads can handle per-domain errors; orchestrator is SPoF."},
                    "cost_efficiency": {"score": 4, "reasoning": f"{n_agents} agents with significant coordination cost."},
                    "simplicity": {"score": 4, "reasoning": "Most complex topology; harder to debug and maintain."},
                    "scalability": {"score": 9, "reasoning": "Excellent scaling: add domains or workers per domain."},
                }
            else:
                scores = {d: {"score": 5, "reasoning": "Unknown pattern; neutral score."} for d in TopologyEvaluatorRole._DIMENSIONS}

            total = sum(s["score"] for s in scores.values())
            return {
                "name": topo.get("name", "Unknown"),
                "scores": scores,
                "total_score": total,
                "overall_assessment": (
                    f"Heuristic evaluation: {pattern} pattern with {n_agents} agents "
                    f"and {len(conns)} connections. Total score {total}/50."
                ),
            }

        result = [_score_one(t) for t in topologies]
        result.sort(key=lambda t: t.get("total_score", 0), reverse=True)
        return result
