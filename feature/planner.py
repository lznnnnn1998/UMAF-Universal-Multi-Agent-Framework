"""FeaturePlannerRole — integration planning agent with dependency-aware decomposition."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from tools import ToolRegistry
from agent import AgentResult, AgentRole
from utils import extract_json_object, safe_read


class FeaturePlannerRole(AgentRole):
    """Plans implementation: files_to_create, files_to_modify, AND sub_tasks with dependencies."""

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
            f"files_to_create AND files_to_modify, PLUS a sub_tasks decomposition "
            f"with dependency ordering for parallel execution.\n\n"
            f"## Project Context\n```json\n{ctx_display}\n```\n\n"
            f"## Feature Description\n{feature_description}\n\n"
            f"## Instructions\n"
            f"1. Read project_context.json to understand the project structure\n"
            f"2. Decompose the feature into sub-tasks that can be built in parallel:\n"
            f"   - Group related files into sub-tasks by module boundary\n"
            f"   - Identify dependencies between sub-tasks (which modules must be built first)\n"
            f"   - If the feature is simple (single module), produce ONE sub_task with empty "
            f"dependencies — the pipeline degrades to flat parallelism\n"
            f"3. For each NEW file needed:\n"
            f"   - Add an entry to `files_to_create` with path, description, "
            f"interfaces to implement, and dependencies\n"
            f"4. For each EXISTING file that needs modification:\n"
            f"   - FIRST read the file to understand its structure\n"
            f"   - Add an entry to `files_to_modify` with:\n"
            f"     * `path`: the file to edit\n"
            f"     * `section`: where to edit (e.g. \"after imports\", "
            f"\"in class Foo\", \"at end of file\")\n"
            f"     * `change`: what to change (e.g. \"add import for X\", "
            f"\"add call to Y\", \"wrap function Z\")\n"
            f"     * `description`: why this change is needed\n"
            f"5. Plan test files for all new/modified code\n"
            f"6. Validate: no circular dependencies, all imports resolve, all dependency "
            f"references point to existing module_names\n"
            f"7. Write implementation_plan.json AND decomposition.json to the working directory, "
            f"then output TASK_COMPLETE\n\n"
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
            f'  ],\n'
            f'  "sub_tasks": [\n'
            f'    {{\n'
            f'      "id": <int>,\n'
            f'      "module_name": "<str>",\n'
            f'      "description": "<str>",\n'
            f'      "dependencies": ["<module_name_or_int_id>", ...],\n'
            f'      "files_to_create": [\n'
            f'        {{"path": "<str>", "description": "<str>", '
            f'"interfaces": ["<str>"], "dependencies": ["<str>"]}}\n'
            f'      ],\n'
            f'      "files_to_modify": [\n'
            f'        {{"path": "<str>", "section": "<str>", '
            f'"change": "<str>", "description": "<str>"}}\n'
            f'      ],\n'
            f'      "test_files": [\n'
            f'        {{"path": "<str>", "covers": ["<str>"]}}\n'
            f'      ]\n'
            f'    }}\n'
            f'  ]\n'
            f'}}\n'
            f"```\n"
        )
        if backend == "claude_cli":
            common += (
                "\nRead existing files before planning modifications. "
                "Output TASK_COMPLETE when implementation_plan.json AND "
                "decomposition.json are written."
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

        # ── Extract/write decomposition.json (sub_tasks with dependencies) ──
        sub_tasks: list[dict[str, Any]] = plan.get("sub_tasks", [])
        if not sub_tasks:
            # Generate fallback decomposition from files_to_create/files_to_modify
            sub_tasks = self._generate_sub_tasks_from_plan(plan)
            plan["sub_tasks"] = sub_tasks
        else:
            # Validate that all dependency references resolve
            module_names = {t.get("module_name", "") for t in sub_tasks}
            sub_ids = {t.get("id") for t in sub_tasks}
            for t in sub_tasks:
                valid_deps = []
                for d in t.get("dependencies", []):
                    if isinstance(d, int) and d in sub_ids:
                        valid_deps.append(d)
                    elif isinstance(d, str) and d in module_names:
                        valid_deps.append(d)
                    else:
                        # Try to resolve by looking up the module name
                        resolved = False
                        for st in sub_tasks:
                            if d == st.get("module_name") or d == st.get("id"):
                                valid_deps.append(d)
                                resolved = True
                                break
                        if not resolved:
                            print(f"  [planner] WARNING: Unresolved dependency '{d}' "
                                  f"in sub_task '{t.get('module_name', t.get('id'))}' — "
                                  f"removing it")
                t["dependencies"] = valid_deps

        # Write decomposition.json for the _coders_node
        decomp_path = os.path.join(working_dir, "decomposition.json")
        try:
            with open(decomp_path, "w") as f:
                json.dump(sub_tasks, f, indent=2)
        except OSError:
            pass

        return plan

    @staticmethod
    def _generate_sub_tasks_from_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate sub_tasks from flat files_to_create / files_to_modify lists.

        Groups files by estimated module boundaries to produce a basic
        decomposition with empty dependencies (flat parallelism).
        """
        files_to_create: list[dict[str, Any]] = plan.get("files_to_create", [])
        files_to_modify: list[dict[str, Any]] = plan.get("files_to_modify", [])
        test_files: list[dict[str, Any]] = plan.get("test_files", [])

        if not files_to_create and not files_to_modify:
            # Truly empty — single empty task
            return [{
                "id": 1,
                "module_name": "feature_implementation",
                "description": plan.get("feature", "Feature implementation"),
                "dependencies": [],
                "files_to_create": [],
                "files_to_modify": [],
                "test_files": [],
            }]

        # Group files by top-level directory (module boundary heuristic)
        groups: dict[str, dict[str, Any]] = {}

        def _group_key(file_path: str) -> str:
            """Extract the top-level module name from a file path."""
            parts = file_path.replace("\\", "/").split("/")
            # Skip common prefixes
            for prefix in ("src", "lib", "app", "pkg"):
                if parts[0] == prefix and len(parts) > 1:
                    parts = parts[1:]
                    break
            # Use first meaningful directory or filename stem
            if len(parts) > 1:
                return parts[0]
            return parts[0].rsplit(".", 1)[0]

        for entry in files_to_create:
            key = _group_key(entry.get("path", ""))
            if key not in groups:
                groups[key] = {"creates": [], "modifies": [], "tests": []}
            groups[key]["creates"].append(entry)

        for entry in files_to_modify:
            key = _group_key(entry.get("path", ""))
            if key not in groups:
                groups[key] = {"creates": [], "modifies": [], "tests": []}
            groups[key]["modifies"].append(entry)

        for entry in test_files:
            # Try to match test files to module groups by what they cover
            covers = entry.get("covers", [])
            matched = False
            for covered in covers:
                ckey = _group_key(covered)
                if ckey in groups:
                    groups[ckey]["tests"].append(entry)
                    matched = True
                    break
            if not matched:
                key = _group_key(entry.get("path", ""))
                if key not in groups:
                    groups[key] = {"creates": [], "modifies": [], "tests": []}
                groups[key]["tests"].append(entry)

        # Build sub_tasks from groups
        sub_tasks: list[dict[str, Any]] = []
        for i, (module_name, group) in enumerate(sorted(groups.items()), start=1):
            sub_tasks.append({
                "id": i,
                "module_name": module_name,
                "description": f"Implement {module_name} module: "
                               f"{len(group['creates'])} new files, "
                               f"{len(group['modifies'])} modified files",
                "dependencies": [],
                "files_to_create": group["creates"],
                "files_to_modify": group["modifies"],
                "test_files": group["tests"],
            })

        return sub_tasks

    @staticmethod
    def _fallback_plan(working_dir: str, feature_description: str) -> dict[str, Any]:
        """Deterministic plan when LLM fails — includes sub_tasks generation."""
        plan = {
            "feature": feature_description or "Unnamed feature",
            "files_to_create": [],
            "files_to_modify": [],
            "test_files": [],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "_fallback": True,
            "_note": "Deterministic fallback — no AI planning was performed. "
                     "Review and edit this plan before proceeding.",
        }
        # Generate sub_tasks from the empty file lists (single-task fallback)
        plan["sub_tasks"] = FeaturePlannerRole._generate_sub_tasks_from_plan(plan)
        return plan
