"""TopologyAnalyzerRole — assesses task complexity for topology optimization."""

import json
import os
from typing import Any

from agent import AgentResult, AgentRole
from tools import ToolRegistry
from utils import extract_json_object


class TopologyAnalyzerRole(AgentRole):
    """Assess task complexity across 6 factors to inform topology design.

    Evaluates: data_dependencies, parallelism_opportunities, tool_requirements,
    error_domains, latency_sensitivity, scale — each with level (low/medium/high)
    and reasoning.
    """

    agent_name: str = "topology_analyzer"
    max_steps: int = 8

    # ── Tools ───────────────────────────────────────────────────────────

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        """Return Read + Write tool specs. Same tools for both backends."""
        return ToolRegistry.to_dicts(
            ToolRegistry.topology_analyzer_tools()
        )

    # ── Task prompt ─────────────────────────────────────────────────────

    def build_task(self, backend: str, input_spec: str = "",
                   working_dir: str = ".", **context: Any) -> str:
        """Build the complexity analysis prompt with backend-aware
        instructions."""
        common = (
            f"You are a topology analysis expert. Your job is to assess the "
            f"complexity of a multi-agent system task and identify the key "
            f"factors that will influence the optimal agent topology.\n\n"
            f"## Requirement\n{input_spec}\n\n"
            f"## Task\n"
            f"Analyze the requirement across 6 complexity factors. For each "
            f"factor, assign a level (low, medium, high) and provide concise "
            f"reasoning (1-2 sentences).\n\n"
            f"### Factors to Assess\n"
            f"1. **data_dependencies** — How much data must flow between agents? "
            f"Do agents need outputs from other agents to proceed?\n"
            f"2. **parallelism_opportunities** — Can sub-tasks be executed "
            f"independently in parallel, or is the work inherently sequential?\n"
            f"3. **tool_requirements** — How many distinct tools does each agent "
            f"need? Are there conflicting or restricted tool needs?\n"
            f"4. **error_domains** — How many distinct failure modes exist? "
            f"What is the blast radius of a single agent failure?\n"
            f"5. **latency_sensitivity** — Is end-to-end latency critical "
            f"(real-time, interactive) or relaxed (batch processing)?\n"
            f"6. **scale** — How many sub-tasks / agents are needed? Is the "
            f"workload compute-bound or I/O-bound?\n\n"
            f"## Output Format\n"
            f'Write your analysis as a JSON file to "complexity_analysis.json" '
            f"with this structure:\n"
            f"```json\n"
            f"{{\n"
            f'  "input_spec": "{input_spec[:100]}",\n'
            f'  "factors": {{\n'
            f'    "data_dependencies": {{"level": "low|medium|high", "reasoning": "..."}},\n'
            f'    "parallelism_opportunities": {{"level": "low|medium|high", "reasoning": "..."}},\n'
            f'    "tool_requirements": {{"level": "low|medium|high", "reasoning": "..."}},\n'
            f'    "error_domains": {{"level": "low|medium|high", "reasoning": "..."}},\n'
            f'    "latency_sensitivity": {{"level": "low|medium|high", "reasoning": "..."}},\n'
            f'    "scale": {{"level": "low|medium|high", "reasoning": "..."}}\n'
            f'  }},\n'
            f'  "overall_complexity": "low|medium|high",\n'
            f'  "key_insights": ["insight 1", "insight 2", ...]\n'
            f"}}\n"
            f"```\n\n"
            f"The JSON object MUST appear INLINE in your response text. "
            f'Also use write_file to save a backup copy to '
            f'"complexity_analysis.json".\n'
            f'Working directory: {working_dir}'
        )

        if backend == "claude_cli":
            backend_note = (
                "\n\nUse your own knowledge and reasoning — do NOT search the "
                "web. Read any provided input files first. Write the JSON "
                "analysis file, then output TASK_COMPLETE."
            )
        else:
            backend_note = (
                "\n\nIf the requirement references a file, read it first to "
                "extract relevant sections. Write the JSON analysis file, "
                "then output TASK_COMPLETE."
            )

        return common + backend_note

    # ── Parse result ────────────────────────────────────────────────────

    def parse_result(self, result: AgentResult, working_dir: str,
                     input_spec: str = "", **context: Any) -> dict[str, Any]:
        """Extract complexity analysis from agent response or disk file."""
        factors: dict[str, Any] = {}

        # 1. Try extracting JSON from agent response messages
        for msg in reversed(result.messages):
            content = msg.content if hasattr(msg, "content") else str(msg)
            json_str = extract_json_object(content)
            if json_str:
                try:
                    parsed = json.loads(json_str)
                    if "factors" in parsed:
                        factors = parsed
                        break
                except json.JSONDecodeError:
                    continue

        # 2. Try reading from disk
        if not factors:
            path = os.path.join(working_dir, "complexity_analysis.json")
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        parsed = json.load(f)
                    if "factors" in parsed:
                        factors = parsed
                except (json.JSONDecodeError, OSError):
                    pass

        # 3. Fallback: build a default analysis
        if not factors:
            factors = self._fallback_analyze(input_spec)

        return factors

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _fallback_analyze(input_spec: str) -> dict[str, Any]:
        """Build a conservative default analysis when extraction fails."""
        word_count = len(input_spec.split())
        scale = "high" if word_count > 100 else ("medium" if word_count > 30 else "low")
        return {
            "input_spec": input_spec[:200],
            "factors": {
                "data_dependencies": {
                    "level": "medium",
                    "reasoning": "Multi-agent systems typically require intermediate "
                                 "data passing between stages."
                },
                "parallelism_opportunities": {
                    "level": "medium",
                    "reasoning": "Some sub-tasks may be parallelizable while others "
                                 "require sequential execution."
                },
                "tool_requirements": {
                    "level": "medium",
                    "reasoning": "Standard tools (Read, Write, Bash) are sufficient "
                                 "for most research and code generation tasks."
                },
                "error_domains": {
                    "level": "medium",
                    "reasoning": "Agent failures can cascade through dependent stages; "
                                 "error isolation should be considered."
                },
                "latency_sensitivity": {
                    "level": "low",
                    "reasoning": "Batch-oriented research and generation tasks are not "
                                 "latency-critical."
                },
                "scale": {
                    "level": scale,
                    "reasoning": f"Based on requirement length ({word_count} words)."
                },
            },
            "overall_complexity": scale,
            "key_insights": [
                "Conservative default analysis generated because LLM extraction failed.",
                "Review and refine manually before proceeding to topology design."
            ],
        }
