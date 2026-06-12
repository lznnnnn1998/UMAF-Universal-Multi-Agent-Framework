"""PlanDecomposerRole — hierarchical task decomposition agent."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from tools import ToolRegistry
from agent import AgentResult, AgentRole
from utils import extract_json_object


class PlanDecomposerRole(AgentRole):
    """Recursively decomposes a task description into a hierarchical task tree.

    Dynamically calibrates depth based on input complexity:
    - 2 levels for simple tasks (goals → tasks)
    - 3 levels for medium tasks (goals → epics → tasks)
    - 4+ levels for large tasks (goals → epics → stories → tasks)

    Each node is typed (goal/epic/story/task), annotated with complexity score
    (1-10), dependency hints, and scope statement. Includes self-validation gate
    checking completeness, coherence, and acyclic dependency structure.
    """

    agent_name: str = "plan_decomposer"
    max_steps: int = 25

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        """Return Read-only tool specs."""
        if hasattr(ToolRegistry, "plan_decomposer_tools"):
            return ToolRegistry.to_dicts(ToolRegistry.plan_decomposer_tools())
        return ToolRegistry.to_dicts([ToolRegistry.READ_FILE])

    def build_task(self, backend: str, working_dir: str = ".",
                   input_spec: str = "", project_context: dict[str, Any] | None = None,
                   **context: Any) -> str:
        """Build the decomposition prompt with adaptive depth instructions."""
        word_count = len(input_spec.split())
        if word_count < 30:
            depth_guide = (
                "2 levels: goals → implementation tasks. "
                "Expect 1-2 goals with 3-5 tasks each."
            )
        elif word_count < 100:
            depth_guide = (
                "3 levels: goals → epics → implementation tasks. "
                "Expect 1-3 goals, 2-3 epics per goal, 2-4 tasks per epic."
            )
        else:
            depth_guide = (
                "4+ levels: goals → epics → user stories → implementation tasks. "
                "Expect 2-4 goals, 2-4 epics per goal, 2-4 stories per epic, "
                "2-5 tasks per story."
            )

        ctx_text = ""
        if project_context:
            lang = project_context.get("language", "unknown")
            src_dirs = project_context.get("source_directories", [])
            ctx_text = (
                f"\n## Project Context\n"
                f"- Language: {lang}\n"
                f"- Source directories: {', '.join(src_dirs) if src_dirs else '(root)'}\n"
            )

        common = (
            f"You are a senior technical project planner. Your job is to decompose "
            f"a task description into a hierarchical task tree suitable for "
            f"implementation planning.\n\n"
            f"## Task Description\n{input_spec}\n"
            f"{ctx_text}\n"
            f"## Decomposition Depth\n"
            f"Based on input complexity ({word_count} words): {depth_guide}\n\n"
            f"## Node Types and Schema\n"
            f"Each node in the tree must have:\n"
            f"- **id**: unique integer identifier within the tree\n"
            f"- **type**: one of \"goal\", \"epic\", \"story\", \"task\"\n"
            f"- **title**: short descriptive name (5-12 words)\n"
            f"- **description**: 1-3 sentences describing scope and deliverables\n"
            f"- **complexity**: integer 1-10 (1=trivial, 5=moderate, 10=extremely complex)\n"
            f"- **dependencies**: list of node IDs that must complete before this node starts\n"
            f"- **children**: list of child node objects (empty for leaf tasks)\n"
            f"- **scope**: one-line statement of what is IN and OUT of scope\n\n"
            f"## Self-Validation Gate\n"
            f"After generating the tree, validate it against these criteria:\n"
            f"1. **Completeness**: Does the tree cover EVERY aspect of the input task?\n"
            f"2. **Coherence**: Are all nodes properly typed and correctly nested?\n"
            f"3. **Acyclic**: Verify no dependency cycles exist\n"
            f"4. **Depth Consistency**: Are leaf tasks at appropriate granularity?\n"
            f"If validation fails, self-correct within your step budget.\n\n"
            f"## Output Format\n"
            f'Write the task tree to "task_tree.json" with this structure:\n'
            f"```json\n"
            f'{{\n'
            f'  "input_spec": "the original task description",\n'
            f'  "complexity_level": "simple|medium|large",\n'
            f'  "max_depth": 3,\n'
            f'  "total_nodes": 15,\n'
            f'  "tree": [\n'
            f'    {{\n'
            f'      "id": 1,\n'
            f'      "type": "goal",\n'
            f'      "title": "Goal Title",\n'
            f'      "description": "Detailed description of this goal.",\n'
            f'      "complexity": 5,\n'
            f'      "dependencies": [],\n'
            f'      "scope": "IN: what is covered. OUT: what is excluded.",\n'
            f'      "children": [\n'
            f'        {{\n'
            f'          "id": 2,\n'
            f'          "type": "epic",\n'
            f'          "title": "Epic Title",\n'
            f'          "description": "Detailed description.",\n'
            f'          "complexity": 4,\n'
            f'          "dependencies": [],\n'
            f'          "scope": "IN: ... OUT: ...",\n'
            f'          "children": [...]\n'
            f'        }}\n'
            f'      ]\n'
            f'    }}\n'
            f'  ],\n'
            f'  "validation": {{\n'
            f'    "complete": true,\n'
            f'    "coherent": true,\n'
            f'    "acyclic": true,\n'
            f'    "depth_consistent": true,\n'
            f'    "notes": "Any self-corrections or observations."\n'
            f'  }},\n'
            f'  "generated_at": "<ISO timestamp>"\n'
            f'}}\n'
            f"```\n\n"
            f"The JSON object MUST appear INLINE in your response text. "
            f'Also use write_file to save a backup copy to "task_tree.json".\n'
            f"Working directory: {working_dir}"
        )

        if backend == "claude_cli":
            backend_note = (
                "\n\nUse your own knowledge and reasoning — do NOT search the web. "
                "Read any context files first. Write the JSON task tree file, "
                "then output TASK_COMPLETE."
            )
        else:
            backend_note = (
                "\n\nRead project_context.json first if it exists. "
                "Write task_tree.json, then output TASK_COMPLETE."
            )

        return common + backend_note

    def parse_result(self, result: AgentResult, working_dir: str = ".",
                     input_spec: str = "", **context: Any) -> dict[str, Any]:
        """Extract task tree from agent response, disk file, or fallback."""
        task_tree: dict[str, Any] = {}

        # 1. Try extracting JSON from agent response messages
        for msg in reversed(result.messages):
            content = msg.content if hasattr(msg, "content") else str(msg)
            json_str = extract_json_object(content)
            if json_str:
                try:
                    parsed = json.loads(json_str)
                    if "tree" in parsed:
                        task_tree = parsed
                        break
                except json.JSONDecodeError:
                    continue

        # 2. Try reading from disk
        if not task_tree:
            path = os.path.join(working_dir, "task_tree.json")
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        parsed = json.load(f)
                    if "tree" in parsed:
                        task_tree = parsed
                except (json.JSONDecodeError, OSError):
                    pass

        # 3. Fallback
        if not task_tree:
            task_tree = self._fallback_decompose(input_spec, working_dir)
            out_path = os.path.join(working_dir, "task_tree.json")
            try:
                with open(out_path, "w") as f:
                    json.dump(task_tree, f, indent=2)
            except OSError:
                pass

        return task_tree

    @staticmethod
    def _fallback_decompose(input_spec: str, working_dir: str = ".") -> dict[str, Any]:
        """Build a simple 2-level task tree from the input description."""
        word_count = len(input_spec.split())
        if word_count < 30:
            complexity = "simple"
            max_depth = 2
        elif word_count < 100:
            complexity = "medium"
            max_depth = 3
        else:
            complexity = "large"
            max_depth = 4

        # Build a basic 2-level tree: 1 goal with 3-5 tasks
        tasks = [
            {
                "id": 2, "type": "task",
                "title": "Analyze requirements and constraints",
                "description": f"Parse and understand the input: {input_spec[:80]}...",
                "complexity": 3, "dependencies": [],
                "scope": "IN: requirement analysis. OUT: implementation.",
                "children": [],
            },
            {
                "id": 3, "type": "task",
                "title": "Design solution architecture",
                "description": "Design the high-level architecture and component interactions.",
                "complexity": 5, "dependencies": [2],
                "scope": "IN: architecture design. OUT: detailed implementation.",
                "children": [],
            },
            {
                "id": 4, "type": "task",
                "title": "Implement core functionality",
                "description": "Implement the main logic and core components.",
                "complexity": 7, "dependencies": [3],
                "scope": "IN: core implementation. OUT: testing and docs.",
                "children": [],
            },
            {
                "id": 5, "type": "task",
                "title": "Write tests and validate",
                "description": "Write unit tests, integration tests, and validate correctness.",
                "complexity": 4, "dependencies": [4],
                "scope": "IN: testing. OUT: deployment.",
                "children": [],
            },
            {
                "id": 6, "type": "task",
                "title": "Document and finalize",
                "description": "Write documentation and prepare for handoff.",
                "complexity": 2, "dependencies": [5],
                "scope": "IN: documentation. OUT: future maintenance.",
                "children": [],
            },
        ]

        return {
            "_fallback": True,
            "input_spec": input_spec,
            "complexity_level": complexity,
            "max_depth": max_depth,
            "total_nodes": 6,
            "tree": [
                {
                    "id": 1,
                    "type": "goal",
                    "title": "Complete implementation plan task",
                    "description": f"Implement the full task: {input_spec[:120]}",
                    "complexity": 5,
                    "dependencies": [],
                    "scope": f"IN: {input_spec[:80]}. OUT: unrelated tasks.",
                    "children": tasks,
                }
            ],
            "validation": {
                "complete": True,
                "coherent": True,
                "acyclic": True,
                "depth_consistent": True,
                "notes": "Fallback decomposition — LLM output was not available. Review manually.",
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
