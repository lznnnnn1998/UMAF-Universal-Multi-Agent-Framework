"""TopologyDesignerRole — proposes candidate agent topologies."""

import json
import os
from typing import Any

from agent import AgentResult, AgentRole
from tools import ToolRegistry
from utils import extract_json_array


class TopologyDesignerRole(AgentRole):
    """Propose 2-4 candidate topologies following design patterns.

    Evaluates complexity factors from the analyzer and generates alternative
    topology configurations, each with agent configurations, connections,
    parallelism strategy, strengths, and weaknesses.
    """

    agent_name: str = "topology_designer"
    max_steps: int = 12

    # Design pattern descriptions used in the prompt
    _PATTERNS = {
        "sequential": (
            "Agents execute one after another in a fixed order. "
            "Best for: strictly ordered workflows with no parallelism opportunities."
        ),
        "fan_out_fan_in": (
            "A dispatcher fans out work to parallel workers, then a collector "
            "aggregates results. Best for: embarrassingly parallel sub-tasks."
        ),
        "debate_consensus": (
            "Multiple agents independently solve the same problem, then a judge "
            "selects or synthesizes the best answer. Best for: high-stakes "
            "decisions where correctness matters more than speed."
        ),
        "hierarchical": (
            "A top-level orchestrator delegates to sub-orchestrators, which in "
            "turn manage worker agents. Best for: complex, multi-layered tasks."
        ),
    }

    # ── Tools ───────────────────────────────────────────────────────────

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        """Return Read + Write tool specs."""
        return ToolRegistry.to_dicts(
            ToolRegistry.topology_designer_tools()
        )

    # ── Task prompt ─────────────────────────────────────────────────────

    def build_task(self, backend: str, complexity_factors: dict[str, Any] | None = None,
                   input_spec: str = "", working_dir: str = ".",
                   evaluation_feedback: str = "",
                   **context: Any) -> str:
        """Build the topology design prompt with backend-aware instructions.

        Args:
            backend: LLM backend name (deepseek or claude_cli).
            complexity_factors: Analysis results from TopologyAnalyzerRole.
            input_spec: Original task/requirement description.
            working_dir: Working directory for file I/O.
            evaluation_feedback: Previous evaluation feedback (non-empty on retry).

        Returns:
            Prompt string for the LLM.
        """
        factors_text = self._format_factors(complexity_factors)

        # Build evaluation feedback section for retry iterations
        feedback_section = ""
        if evaluation_feedback:
            feedback_section = (
                f"\n## Previous Evaluation Feedback\n"
                f"The evaluator returned the following feedback on the previous "
                f"set of topologies. Use this feedback to propose improved "
                f"topologies that address the identified weaknesses:\n\n"
                f"**{evaluation_feedback}**\n\n"
                f"Focus on improving the low-scoring dimensions by adjusting "
                f"agent configurations, connections, or parallelism strategies.\n"
            )

        common = (
            f"You are a topology design expert. Your job is to propose "
            f"candidate agent topologies based on a complexity analysis of "
            f"the target system.\n\n"
            f"## Requirement\n{input_spec}\n\n"
            f"## Complexity Analysis\n{factors_text}\n"
            f"{feedback_section}"
            f"## Task\n"
            f"Propose 2-4 candidate topologies. Each topology should follow "
            f"one of the four design patterns below. Every topology must be "
            f"feasible — you must be able to actually implement it as a "
            f"LangGraph StateGraph.\n\n"
            f"### Design Patterns\n"
            f"1. **sequential** — {self._PATTERNS['sequential']}\n"
            f"2. **fan_out_fan_in** — {self._PATTERNS['fan_out_fan_in']}\n"
            f"3. **debate_consensus** — {self._PATTERNS['debate_consensus']}\n"
            f"4. **hierarchical** — {self._PATTERNS['hierarchical']}\n\n"
            f"### For Each Topology, Provide\n"
            f"- **name**: Short descriptive name (e.g. \"Sequential Pipeline\")\n"
            f"- **pattern**: One of the four patterns above\n"
            f"- **description**: Why this topology fits the requirement (2-3 sentences)\n"
            f"- **agents**: List of agent configurations, each with:\n"
            f"  - agent_name, role_type (description of what it does),\n"
            f"  - tools (list of tool names), max_steps\n"
            f"- **connections**: List of edges {{from: agent_name, to: agent_name}}\n"
            f"- **parallelism_strategy**: How parallel execution is used "
            f"(or \"none\" if sequential)\n"
            f"- **strengths**: 2-3 advantages of this topology\n"
            f"- **weaknesses**: 2-3 limitations or risks\n\n"
            f"## Output Format\n"
            f'Write your candidate topologies as a JSON array to '
            f'"candidate_topologies.json":\n'
            f"```json\n"
            f"[\n"
            f"  {{\n"
            f'    "name": "...",\n'
            f'    "pattern": "sequential|fan_out_fan_in|debate_consensus|hierarchical",\n'
            f'    "description": "...",\n'
            f'    "agents": [\n'
            f'      {{"agent_name": "...", "role_type": "...", '
            f'"tools": ["read_file", "write_file"], "max_steps": 10}}\n'
            f'    ],\n'
            f'    "connections": [\n'
            f'      {{"from": "agent_a", "to": "agent_b"}}\n'
            f'    ],\n'
            f'    "parallelism_strategy": "...",\n'
            f'    "strengths": ["...", "..."],\n'
            f'    "weaknesses": ["...", "..."]\n'
            f'  }}\n'
            f']\n'
            f"```\n\n"
            f"The JSON array MUST appear INLINE in your response text. "
            f'Also use write_file to save a backup copy to '
            f'"candidate_topologies.json".\n'
            f'Working directory: {working_dir}'
        )

        if backend == "claude_cli":
            backend_note = (
                "\n\nUse your own knowledge — do NOT search the web. "
                "Read the complexity_analysis.json file first to understand "
                "the complexity factors. Write candidate_topologies.json, "
                "then output TASK_COMPLETE."
            )
        else:
            backend_note = (
                "\n\nRead complexity_analysis.json first to understand the "
                "complexity factors. Write candidate_topologies.json, "
                "then output TASK_COMPLETE."
            )

        return common + backend_note

    # ── Parse result ────────────────────────────────────────────────────

    def parse_result(self, result: AgentResult, working_dir: str,
                     complexity_factors: dict[str, Any] | None = None,
                     input_spec: str = "",
                     evaluation_feedback: str = "",
                     **context: Any) -> list[dict[str, Any]]:
        """Extract candidate topologies from agent response or disk file.

        Args:
            result: AgentResult from the agent execution.
            working_dir: Working directory for disk fallback.
            complexity_factors: Analysis results from TopologyAnalyzerRole.
            input_spec: Original task/requirement description.
            evaluation_feedback: Previous evaluation feedback (non-empty on retry).
                Passed through — build_task consumes this for prompt generation.

        Returns:
            List of candidate topology dicts.
        """
        topologies: list[dict[str, Any]] = []

        # 1. Try extracting JSON array from agent response messages
        for msg in reversed(result.messages):
            content = msg.content if hasattr(msg, "content") else str(msg)
            json_str = extract_json_array(content)
            if json_str:
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        topologies = parsed
                        break
                except json.JSONDecodeError:
                    continue

        # 2. Try reading from disk
        if not topologies:
            path = os.path.join(working_dir, "candidate_topologies.json")
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        parsed = json.load(f)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        topologies = parsed
                except (json.JSONDecodeError, OSError):
                    pass

        # 3. Fallback: build default topologies
        if not topologies:
            topologies = self._fallback_design(complexity_factors or {}, input_spec)

        return topologies

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _format_factors(factors: dict[str, Any] | None) -> str:
        """Format complexity factors as a readable summary string."""
        if not factors:
            return "(No complexity analysis available — use the requirement text directly.)"

        f = factors.get("factors", factors)
        lines = []
        factor_names = [
            "data_dependencies",
            "parallelism_opportunities",
            "tool_requirements",
            "error_domains",
            "latency_sensitivity",
            "scale",
        ]
        for name in factor_names:
            entry = f.get(name, {})
            if isinstance(entry, dict):
                level = entry.get("level", "unknown")
                reasoning = entry.get("reasoning", "")
                lines.append(f"  - {name}: {level.upper()} — {reasoning}")
        overall = factors.get("overall_complexity", "unknown")
        insights = factors.get("key_insights", [])
        result = "\n".join(lines)
        result += f"\n  Overall complexity: {overall}"
        if insights:
            result += "\n  Key insights:"
            for ins in insights:
                result += f"\n    - {ins}"
        return result

    @staticmethod
    def _fallback_design(
        complexity_factors: dict[str, Any],
        input_spec: str,
    ) -> list[dict[str, Any]]:
        """Build conservative default topologies when extraction fails.

        Returns 4 topologies covering all design patterns: sequential,
        fan_out_fan_in, debate_consensus, and hierarchical.
        """
        return [
            {
                "_fallback": True,
                "name": "Sequential Pipeline",
                "pattern": "sequential",
                "description": (
                    "A linear chain of agents where each stage's output feeds "
                    "the next. Simple, predictable, easy to debug."
                ),
                "agents": [
                    {"agent_name": "analyzer", "role_type": "analyzes input",
                     "tools": ["read_file", "write_file"], "max_steps": 10},
                    {"agent_name": "worker", "role_type": "processes work",
                     "tools": ["read_file", "write_file", "run_command"], "max_steps": 10},
                    {"agent_name": "writer", "role_type": "produces output",
                     "tools": ["write_file"], "max_steps": 8},
                ],
                "connections": [
                    {"from": "analyzer", "to": "worker"},
                    {"from": "worker", "to": "writer"},
                ],
                "parallelism_strategy": "none",
                "strengths": [
                    "Simplest to implement and debug",
                    "Clear data flow and dependency chain",
                    "Low coordination overhead",
                ],
                "weaknesses": [
                    "No parallelism — total latency is sum of all agent times",
                    "Single point of failure at each stage",
                    "Does not scale to large numbers of independent sub-tasks",
                ],
            },
            {
                "_fallback": True,
                "name": "Fan-Out/Fan-In Pipeline",
                "pattern": "fan_out_fan_in",
                "description": (
                    "A head agent decomposes work, parallel workers execute "
                    "independently, and an aggregator combines results."
                ),
                "agents": [
                    {"agent_name": "head", "role_type": "decomposes work",
                     "tools": ["read_file", "write_file"], "max_steps": 10},
                    {"agent_name": "worker_1", "role_type": "processes sub-task 1",
                     "tools": ["read_file", "write_file", "run_command"], "max_steps": 10},
                    {"agent_name": "worker_2", "role_type": "processes sub-task 2",
                     "tools": ["read_file", "write_file", "run_command"], "max_steps": 10},
                    {"agent_name": "aggregator", "role_type": "combines results",
                     "tools": ["read_file", "write_file"], "max_steps": 8},
                ],
                "connections": [
                    {"from": "head", "to": "worker_1"},
                    {"from": "head", "to": "worker_2"},
                    {"from": "worker_1", "to": "aggregator"},
                    {"from": "worker_2", "to": "aggregator"},
                ],
                "parallelism_strategy": "parallel workers after head; aggregator after all workers complete",
                "strengths": [
                    "Parallel workers reduce end-to-end latency",
                    "Independent workers isolate failures",
                    "Scales well with sub-task count",
                ],
                "weaknesses": [
                    "Aggregator is a bottleneck if worker output is large",
                    "Requires careful decomposition by the head agent",
                    "No fallback if a worker fails (unless retries are added)",
                ],
            },
            {
                "_fallback": True,
                "name": "Debate Consensus",
                "pattern": "debate_consensus",
                "description": (
                    "Multiple agents independently analyze the same problem, "
                    "then a judge selects or synthesizes the best answer. Best "
                    "for high-stakes decisions where correctness matters more "
                    "than speed."
                ),
                "agents": [
                    {"agent_name": "debater_a", "role_type": "independent analyst A",
                     "tools": ["read_file", "write_file"], "max_steps": 10},
                    {"agent_name": "debater_b", "role_type": "independent analyst B",
                     "tools": ["read_file", "write_file"], "max_steps": 10},
                    {"agent_name": "debater_c", "role_type": "independent analyst C",
                     "tools": ["read_file", "write_file"], "max_steps": 10},
                    {"agent_name": "judge", "role_type": "selects/synthesizes best answer",
                     "tools": ["read_file", "write_file"], "max_steps": 10},
                ],
                "connections": [
                    {"from": "debater_a", "to": "judge"},
                    {"from": "debater_b", "to": "judge"},
                    {"from": "debater_c", "to": "judge"},
                ],
                "parallelism_strategy": "all debaters run in parallel; judge runs after all debaters complete",
                "strengths": [
                    "Highest fault tolerance via redundant analysis",
                    "Different perspectives catch edge cases",
                    "Judge can select or synthesize best parts of each analysis",
                ],
                "weaknesses": [
                    "Redundant execution costs 3x more tokens",
                    "Higher total latency if debaters are sequential",
                    "Consensus logic adds complexity to synthesis",
                ],
            },
            {
                "_fallback": True,
                "name": "Hierarchical Orchestrator",
                "pattern": "hierarchical",
                "description": (
                    "A two-level hierarchy: orchestrator manages sub-orchestrators, "
                    "each handling a domain. Best for complex multi-domain tasks."
                ),
                "agents": [
                    {"agent_name": "orchestrator", "role_type": "top-level coordinator",
                     "tools": ["read_file", "write_file"], "max_steps": 12},
                    {"agent_name": "domain_lead_a", "role_type": "domain A coordinator",
                     "tools": ["read_file", "write_file", "run_command"], "max_steps": 10},
                    {"agent_name": "domain_lead_b", "role_type": "domain B coordinator",
                     "tools": ["read_file", "write_file", "run_command"], "max_steps": 10},
                    {"agent_name": "worker_a1", "role_type": "domain A worker",
                     "tools": ["read_file", "write_file"], "max_steps": 8},
                    {"agent_name": "worker_b1", "role_type": "domain B worker",
                     "tools": ["read_file", "write_file"], "max_steps": 8},
                    {"agent_name": "synthesizer", "role_type": "merges domain outputs",
                     "tools": ["read_file", "write_file"], "max_steps": 8},
                ],
                "connections": [
                    {"from": "orchestrator", "to": "domain_lead_a"},
                    {"from": "orchestrator", "to": "domain_lead_b"},
                    {"from": "domain_lead_a", "to": "worker_a1"},
                    {"from": "domain_lead_b", "to": "worker_b1"},
                    {"from": "worker_a1", "to": "synthesizer"},
                    {"from": "worker_b1", "to": "synthesizer"},
                ],
                "parallelism_strategy": "parallel domain leads; parallel workers within domains",
                "strengths": [
                    "Natural separation of concerns across domains",
                    "Two levels of parallelism maximize throughput",
                    "Each domain lead can manage its own error handling",
                ],
                "weaknesses": [
                    "Higher coordination complexity",
                    "More agents = higher total cost",
                    "Orchestrator failure halts the entire pipeline",
                ],
            },
        ]
