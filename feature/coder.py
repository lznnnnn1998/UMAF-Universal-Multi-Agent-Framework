"""FeatureCoderRole — implements changes (creates, modifies, tests).

Also provides _feature_coder_worker for parallel multi-coder execution.
"""

from __future__ import annotations

import json
import os
from typing import Any

from tools import ToolRegistry
from agent import AgentResult, AgentRole
from utils import safe_read


class FeatureCoderRole(AgentRole):
    """Creates new files, modifies existing files, writes and runs tests.

    When dependency_outputs are provided via context, the coder reads and
    verifies upstream module outputs before building its own module.
    """

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
                   project_dir: str = ".", review_issues: list[str] | None = None,
                   **context: Any) -> str:
        plan_path = os.path.join(working_dir, "implementation_plan.json")
        ctx_path = os.path.join(working_dir, "project_context.json")
        decomp_path = os.path.join(working_dir, "decomposition.json")

        plan_display = safe_read(plan_path)[:10000] or "(no plan found)"
        ctx_display = safe_read(ctx_path)[:6000] or "(no context found)"

        # Check for sub-task-specific context (multi-coder mode)
        sub_task = context.get("sub_task", {})
        dependency_outputs: list[dict[str, Any]] | None = context.get(
            "dependency_outputs"
        ) or context.get("_dependency_outputs")

        # If running in multi-coder mode, read the decomposition for per-coder tasks
        decomp_display = ""
        if sub_task:
            module_name = sub_task.get("module_name", "unknown")
            decomp_display = (
                f"\n## Your Sub-Task\n"
                f"- **Module**: {module_name}\n"
                f"- **Description**: {sub_task.get('description', 'N/A')}\n"
                f"- **Dependencies**: {sub_task.get('dependencies', [])}\n"
                f"- **Your files_to_create**: {json.dumps(sub_task.get('files_to_create', []), indent=2)}\n"
                f"- **Your files_to_modify**: {json.dumps(sub_task.get('files_to_modify', []), indent=2)}\n"
                f"- **Your test_files**: {json.dumps(sub_task.get('test_files', []), indent=2)}\n"
                f"\nFocus ONLY on the files assigned to your module above. "
                f"Other coders will handle other modules.\n"
            )

        # Build dependency context section
        dep_section = ""
        if dependency_outputs:
            dep_section = (
                f"\n## Dependency Context — READ AND VERIFY FIRST\n"
                f"You depend on these completed modules. Read their output files "
                f"and verify they work before building on top.\n\n"
                f"For each dependency:\n"
                f"- Read all module files produced by that dependency\n"
                f"- Verify that imports resolve correctly\n"
                f"- Verify that interfaces/APIs match your expectations\n"
                f"- Report any issues with 'DEPENDENCY_ISSUE: <issue>' token\n\n"
                f"### Dependency Outputs:\n"
            )
            for dep in dependency_outputs:
                dep_id = dep.get("dep_id") or dep.get("module_name") or dep.get("sub_task_id", "?")
                dep_name = dep.get("module_name") or dep.get("title", str(dep_id))
                dep_files = dep.get("files", [])
                dep_summary = dep.get("summary", "")
                dep_section += f"\n**{dep_name}** (id={dep_id}):\n"
                if dep_files:
                    for f in dep_files:
                        dep_section += f"  - `{f}`\n"
                else:
                    dep_section += "  - (no files listed — check working directory)\n"
                if dep_summary:
                    dep_section += f"  - Summary: {dep_summary[:200]}\n"
            dep_section += (
                f"\nAfter verifying all dependencies, output 'DEPENDENCY_VERIFIED' "
                f"if all checks pass, or 'DEPENDENCY_ISSUE: <details>' for each "
                f"problem found.\n"
            )

        prompt = (
            f"You are a full-stack software engineer implementing a feature. "
            f"You have COMPLETE project analysis and an implementation plan.\n\n"
            f"## Project Context\n"
            f"This describes the EXISTING project — conventions, tech stack, "
            f"test patterns. Follow these EXACTLY.\n"
            f"```json\n{ctx_display}\n```\n\n"
            f"## Implementation Plan\n"
            f"```json\n{plan_display}\n```\n\n"
            f"{decomp_display}"
            f"{dep_section}"
            f"## File Locations — READ CAREFULLY\n"
            f"- Source code lives in the **project directory**: `{project_dir}`\n"
            f"- Meta files (plan, context, report) are in the **output directory**: `{working_dir}`\n"
            f"- Read existing source files from: `{project_dir}/<path>`\n"
            f"- Write ALL source and test files to: `{project_dir}/<path>`\n"
            f"- Do NOT write source files under `{working_dir}`\n\n"
            f"## Instructions — Follow In Order\n\n"
            f"### Step 1: Read the plan and context\n"
            f"Read both files above. Understand what needs to be created AND "
            f"what existing files need to be modified.\n\n"
            f"### Step 2: Modify existing files FIRST\n"
            f"For each entry in `files_to_modify`:\n"
            f"  a. READ the existing file from `{project_dir}/<path>` "
            f"so you understand its structure\n"
            f"  b. Make the specified change in the specified section\n"
            f"  c. Use write_file or write_lines to write the COMPLETE modified file "
            f"back to `{project_dir}/<path>`\n"
            f"  d. Match ALL existing conventions (naming, imports, type hints, "
            f"docstrings, error handling)\n\n"
            f"### Step 3: Create new files\n"
            f"For each entry in `files_to_create`:\n"
            f"  a. Read any dependency files from `{project_dir}` first\n"
            f"  b. Write the complete file to `{project_dir}/<path>` "
            f"matching ALL project conventions\n"
            f"  c. Use correct naming, imports, type hints, docstrings\n"
            f"  d. Implement all listed interfaces\n\n"
            f"### Step 4: Write tests\n"
            f"For each entry in `test_files`:\n"
            f"  a. Write test files to `{project_dir}/<path>`\n"
            f"  b. Match the test patterns from project_context EXACTLY\n"
            f"  c. Use the correct test framework, file naming, and fixtures\n"
            f"  d. Cover happy path, edge cases, and error handling\n"
            f"  e. Use the project's mock library if needed\n\n"
            f"### Step 5: Run tests and fix\n"
            f"  a. Run tests from the project directory: `cd {project_dir} && <test runner>`\n"
            f"  b. If tests fail, read the error, fix the code, re-run\n"
            f"  c. Iterate until ALL tests pass\n\n"
            f"## CRITICAL Rules\n"
            f"- NEVER skip reading an existing file before modifying it\n"
            f"- Write the COMPLETE file when modifying — not just the changed part\n"
            f"- Match ALL conventions from project_context exactly\n"
            f"- Write EVERY file listed in the plan\n"
            f"- Run tests and fix failures before declaring completion\n"
            f"- ALL source/test files go under `{project_dir}` — NOT under `{working_dir}`\n"
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
                     project_dir: str = ".", **context: Any) -> dict[str, Any]:
        plan_path = os.path.join(working_dir, "implementation_plan.json")
        plan: dict[str, Any] = {}
        try:
            with open(plan_path) as f:
                plan = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass

        changed: list[str] = []

        # Check all planned files — source code lives under project_dir
        for entry in plan.get("files_to_create", []):
            fp = entry.get("path", "")
            full = os.path.join(project_dir, fp)
            if os.path.isfile(full) and os.path.getsize(full) > 0:
                changed.append(fp)

        for entry in plan.get("files_to_modify", []):
            fp = entry.get("path", "")
            if fp and fp not in changed:
                changed.append(fp)

        for entry in plan.get("test_files", []):
            fp = entry.get("path", "")
            full = os.path.join(project_dir, fp)
            if os.path.isfile(full) and os.path.getsize(full) > 0:
                changed.append(fp)

        # ── Dependency verification ─────────────────────────────────────────
        dependency_verification: bool | None = None
        # Check if this coder had dependencies; scan for verification tokens
        has_deps = bool(
            context.get("dependency_outputs")
            or context.get("_dependency_outputs")
            or context.get("sub_task", {}).get("dependencies")
        )
        if has_deps:
            for msg in reversed(result.messages):
                content = msg.content if hasattr(msg, "content") else str(msg)
                if "DEPENDENCY_VERIFIED" in content:
                    dependency_verification = True
                    break
                if "DEPENDENCY_ISSUE:" in content:
                    dependency_verification = False
                    break
            # If neither token found but dependencies exist, report as unverified
            if dependency_verification is None:
                dependency_verification = None  # explicit None = no verification attempted

        return {
            "changed_files": changed,
            "success": len(changed) > 0,
            "summary": f"Changed {len(changed)} files",
            "dependency_verification": dependency_verification,
        }


def _feature_coder_worker(
    item: dict[str, Any],
    working_dir: str,
    backend: str,
    project_dir: str = ".",
    version: int = 1,
    dependency_outputs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run a single FeatureCoderRole for one sub-task (parallel worker entry point).

    Matches the ``research_subtask`` pattern in ``research/worker_agent.py``
    so it can be called by ``BasePipeline._run_parallel_agents``.

    Args:
        item: sub-task dict with id, module_name, description, dependencies,
              files_to_create, files_to_modify, test_files.
        working_dir: base working directory.
        backend: LLM backend to use.
        project_dir: path to the project being modified.
        version: checkpoint version for retries.
        dependency_outputs: outputs from upstream (level[i-1]) coders.

    Returns:
        dict with sub_task_id, module_name, files, summary,
        dependency_verification keys.
    """
    sub_id = item.get("id", 0)
    module_name = item.get("module_name", f"subtask_{sub_id}")

    # Merge item-level dependency outputs with explicitly passed ones
    deps = list(dependency_outputs or [])
    item_deps = item.get("_dependency_outputs", [])
    if item_deps:
        deps.extend(item_deps)

    role = FeatureCoderRole()
    role.agent_name = f"feature_coder_{module_name}"

    result = role.execute(
        working_dir=working_dir,
        backend=backend,
        version=version,
        project_dir=project_dir,
        sub_task=item,
        dependency_outputs=deps if deps else None,
    )

    # role.execute() already calls parse_result() internally
    parsed = result if isinstance(result, dict) else {}
    changed = parsed.get("changed_files", [])

    return {
        "sub_task_id": sub_id,
        "module_name": module_name,
        "files": changed,
        "summary": parsed.get("summary", f"Changed {len(changed)} files"),
        "dependency_verification": parsed.get("dependency_verification"),
    }
