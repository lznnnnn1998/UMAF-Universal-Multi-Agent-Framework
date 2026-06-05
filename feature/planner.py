"""FeaturePlannerRole — integration planning agent."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from tools import ToolRegistry
from agent import AgentResult, AgentRole
from utils import extract_json_object, safe_read


class FeaturePlannerRole(AgentRole):
    """Plans implementation: files_to_create AND files_to_modify."""

    agent_name = "feature_planner"
    max_steps = 12

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        if hasattr(ToolRegistry, "feature_planner_tools"):
            return ToolRegistry.to_dicts(ToolRegistry.feature_planner_tools())
        return ToolRegistry.to_dicts([
            ToolRegistry.READ_FILE, ToolRegistry.WRITE_FILE,
        ])

    def build_task(self, backend: str, working_dir: str = ".",
                   feature_description: str = "", **context: Any) -> str:
        ctx_path = os.path.join(working_dir, "project_context.json")
        ctx_display = safe_read(ctx_path)[:8000] or "(not found — scan the project yourself)"

        common = (
            f"You are an integration planner. Read the project context and feature "
            f"description, then produce an implementation_plan.json with both "
            f"files_to_create AND files_to_modify.\n\n"
            f"## Project Context\n```json\n{ctx_display}\n```\n\n"
            f"## Feature Description\n{feature_description}\n\n"
            f"## Instructions\n"
            f"1. Read project_context.json to understand the project structure\n"
            f"2. For each NEW file needed:\n"
            f"   - Add an entry to `files_to_create` with path, description, "
            f"interfaces to implement, and dependencies\n"
            f"3. For each EXISTING file that needs modification:\n"
            f"   - FIRST read the file to understand its structure\n"
            f"   - Add an entry to `files_to_modify` with:\n"
            f"     * `path`: the file to edit\n"
            f"     * `section`: where to edit (e.g. \"after imports\", "
            f"\"in class Foo\", \"at end of file\")\n"
            f"     * `change`: what to change (e.g. \"add import for X\", "
            f"\"add call to Y\", \"wrap function Z\")\n"
            f"     * `description`: why this change is needed\n"
            f"4. Plan test files for all new/modified code\n"
            f"5. Validate: no circular dependencies, all imports resolve\n"
            f"6. Write implementation_plan.json and output TASK_COMPLETE\n\n"
            f"## Output schema\n"
            f"```json\n"
            f'{{\n'
            f'  "feature": "<description>",\n'
            f'  "files_to_create": [\n'
            f'    {{"path": "src/new_module.py", "description": "...", '
            f'"interfaces": [...], "dependencies": [...]}}\n'
            f'  ],\n'
            f'  "files_to_modify": [\n'
            f'    {{"path": "src/existing.py", "section": "after imports", '
            f'"change": "Add import for new_module", "description": "..."}}\n'
            f'  ],\n'
            f'  "test_files": [\n'
            f'    {{"path": "tests/test_new_module.py", '
            f'"covers": ["src/new_module.py"]}}\n'
            f'  ]\n'
            f'}}\n'
            f"```\n"
        )
        if backend == "claude_cli":
            common += (
                "\nRead existing files before planning modifications. "
                "Output TASK_COMPLETE when implementation_plan.json is written."
            )
        else:
            common += "\nOutput TASK_COMPLETE when done."
        return common

    def parse_result(self, result: AgentResult, working_dir: str = ".",
                     **context: Any) -> dict[str, Any]:
        plan: dict[str, Any] = {}
        # 1. Try agent messages
        for msg in reversed(result.messages):
            content = msg.content if hasattr(msg, "content") else str(msg)
            json_str = extract_json_object(content)
            if json_str:
                try:
                    parsed = json.loads(json_str)
                    if "files_to_create" in parsed or "files_to_modify" in parsed:
                        plan = parsed
                        break
                except json.JSONDecodeError:
                    continue
        # 2. Try disk
        if not plan:
            path = os.path.join(working_dir, "implementation_plan.json")
            if os.path.isfile(path):
                try:
                    with open(path) as f:
                        plan = json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass
        # 3. Fallback
        if not plan:
            plan = self._fallback_plan(working_dir, context.get("feature_description", ""))
            out_path = os.path.join(working_dir, "implementation_plan.json")
            try:
                with open(out_path, "w") as f:
                    json.dump(plan, f, indent=2)
            except OSError:
                pass
        return plan

    @staticmethod
    def _fallback_plan(working_dir: str, feature_description: str) -> dict[str, Any]:
        """Deterministic plan when LLM fails."""
        return {
            "feature": feature_description or "Unnamed feature",
            "files_to_create": [],
            "files_to_modify": [],
            "test_files": [],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "_fallback": True,
            "_note": "Deterministic fallback — no AI planning was performed. "
                     "Review and edit this plan before proceeding.",
        }
