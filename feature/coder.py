"""FeatureCoderRole — implements changes (creates, modifies, tests)."""

from __future__ import annotations

import json
import os
from typing import Any

from tools import ToolRegistry
from agent import AgentResult, AgentRole
from utils import safe_read


class FeatureCoderRole(AgentRole):
    """Creates new files, modifies existing files, writes and runs tests."""

    agent_name = "feature_coder"
    max_steps = 25

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        if hasattr(ToolRegistry, "feature_coder_tools"):
            return ToolRegistry.to_dicts(ToolRegistry.feature_coder_tools())
        return ToolRegistry.to_dicts([
            ToolRegistry.READ_FILE, ToolRegistry.WRITE_FILE,
            ToolRegistry.WRITE_LINES, ToolRegistry.RUN_COMMAND,
        ])

    def build_task(self, backend: str, working_dir: str = ".",
                   review_issues: list[str] | None = None,
                   **context: Any) -> str:
        plan_path = os.path.join(working_dir, "implementation_plan.json")
        ctx_path = os.path.join(working_dir, "project_context.json")

        plan_display = safe_read(plan_path)[:10000] or "(no plan found)"
        ctx_display = safe_read(ctx_path)[:6000] or "(no context found)"

        prompt = (
            f"You are a full-stack software engineer implementing a feature. "
            f"You have COMPLETE project analysis and an implementation plan.\n\n"
            f"## Project Context\n"
            f"This describes the EXISTING project — conventions, tech stack, "
            f"test patterns. Follow these EXACTLY.\n"
            f"```json\n{ctx_display}\n```\n\n"
            f"## Implementation Plan\n"
            f"```json\n{plan_display}\n```\n\n"
            f"## Instructions — Follow In Order\n\n"
            f"### Step 1: Read the plan and context\n"
            f"Read both files above. Understand what needs to be created AND "
            f"what existing files need to be modified.\n\n"
            f"### Step 2: Modify existing files FIRST\n"
            f"For each entry in `files_to_modify`:\n"
            f"  a. READ the existing file so you understand its structure\n"
            f"  b. Make the specified change in the specified section\n"
            f"  c. Use write_file or write_lines to write the COMPLETE modified file\n"
            f"  d. Match ALL existing conventions (naming, imports, type hints, "
            f"docstrings, error handling)\n\n"
            f"### Step 3: Create new files\n"
            f"For each entry in `files_to_create`:\n"
            f"  a. Read any dependency files first\n"
            f"  b. Write the complete file matching ALL project conventions\n"
            f"  c. Use correct naming, imports, type hints, docstrings\n"
            f"  d. Implement all listed interfaces\n\n"
            f"### Step 4: Write tests\n"
            f"For each entry in `test_files`:\n"
            f"  a. Match the test patterns from project_context EXACTLY\n"
            f"  b. Use the correct test framework, file naming, and fixtures\n"
            f"  c. Cover happy path, edge cases, and error handling\n"
            f"  d. Use the project's mock library if needed\n\n"
            f"### Step 5: Run tests and fix\n"
            f"  a. Run the tests with the project's test runner\n"
            f"  b. If tests fail, read the error, fix the code, re-run\n"
            f"  c. Iterate until ALL tests pass\n\n"
            f"## CRITICAL Rules\n"
            f"- NEVER skip reading an existing file before modifying it\n"
            f"- Write the COMPLETE file when modifying — not just the changed part\n"
            f"- Match ALL conventions from project_context exactly\n"
            f"- Write EVERY file listed in the plan\n"
            f"- Run tests and fix failures before declaring completion\n"
            f"- Write all files to paths relative to: {working_dir}\n"
        )

        if review_issues:
            issues_text = "\n".join(f"- {i}" for i in review_issues)
            prompt += (
                f"\n\n## Previous Review Issues — FIX THESE\n"
                f"{issues_text}\n\n"
                f"Fix ALL issues listed above, verify tests pass, then output TASK_COMPLETE."
            )

        if backend == "claude_cli":
            prompt += "\n\nUse Read and Write tools. Do NOT search the web. Output TASK_COMPLETE when all files are written and tests pass."
        else:
            prompt += "\n\nOutput TASK_COMPLETE when all files are written and tests pass."

        return prompt

    def parse_result(self, result: AgentResult, working_dir: str = ".",
                     **context: Any) -> dict[str, Any]:
        plan_path = os.path.join(working_dir, "implementation_plan.json")
        plan: dict[str, Any] = {}
        try:
            with open(plan_path) as f:
                plan = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass

        changed: list[str] = []

        # Check all planned files
        for entry in plan.get("files_to_create", []):
            fp = entry.get("path", "")
            full = os.path.join(working_dir, fp)
            if os.path.isfile(full) and os.path.getsize(full) > 0:
                changed.append(fp)

        for entry in plan.get("files_to_modify", []):
            fp = entry.get("path", "")
            if fp and fp not in changed:
                changed.append(fp)

        for entry in plan.get("test_files", []):
            fp = entry.get("path", "")
            full = os.path.join(working_dir, fp)
            if os.path.isfile(full) and os.path.getsize(full) > 0:
                changed.append(fp)

        return {
            "changed_files": changed,
            "success": len(changed) > 0,
            "summary": f"Changed {len(changed)} files",
        }
