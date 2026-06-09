"""Self-evolution coder — implements improvements to UMAF's own source code."""

from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any

from agent import AgentResult, AgentRole
from tools import ToolRegistry


class SelfEvolutionCoderRole(AgentRole):
    """Implement improvements to UMAF's own code based on the implementation plan."""

    agent_name = "self_evolution_coder"
    max_steps = 30

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        return ToolRegistry.to_dicts(ToolRegistry.self_evolution_coder_tools())

    def build_task(self, backend: str, working_dir: str = "",
                   project_dir: str = ".", implementation_plan: dict[str, Any] | None = None,
                   review_issues: list[str] | None = None, **context: Any) -> str:
        plan_text = ""
        if implementation_plan:
            improvements = implementation_plan.get("improvements", [])
            plan_text = "\n".join(
                f"### {imp['id']}: {imp['title']}\n"
                f"- Action: {imp['action']}\n"
                f"- Files to modify: {imp.get('files_to_modify', [])}\n"
                f"- Files to create: {imp.get('files_to_create', [])}\n"
                f"- Verification: {imp.get('verification', 'N/A')}\n"
                for imp in improvements
            )

        review_text = ""
        if review_issues:
            review_text = "\n## REVIEW ISSUES TO FIX\n" + "\n".join(
                f"- {issue}" for issue in review_issues
            ) + "\n\nFix all issues listed above before re-running verification.\n"

        return f"""You are a self-evolution coder for UMAF. Your job is to modify UMAF's own source code to implement improvements. You are making UMAF better.

## Project Directory
{project_dir}

## Implementation Plan
{plan_text}
{review_text}

## Instructions

### For each file to MODIFY:
1. **Read the file first** using `read_file` to understand the current code.
2. **Make targeted changes** using `write_file` (write the complete file). Keep changes minimal — only change what needs to change.
3. **Document the change** — use clear variable names and type hints (Python >= 3.11 style: `X | None`, not `Optional[X]`).

### For each file to CREATE:
1. **Create the file** using `write_file` with complete, working code.
2. **Follow UMAF conventions**:
   - `from __future__ import annotations` at the top
   - Use `X | None` syntax (Python >= 3.11)
   - No multi-line docstrings — one short line max
   - No comments unless the WHY is non-obvious
   - AgentRole ABC with agent_name, max_steps, tools_for_backend, build_task, parse_result

### Verification
After each change:
1. **Run the affected tests**: `cd {project_dir} && python -m pytest test/ -x -q 2>&1 | tail -5`
2. If tests fail, read the error output, fix the issue, and re-run.
3. If ALL tests pass, move on to the next improvement.

### Important Rules
- Do NOT modify `.env`, `.gitignore`, `CLAUDE.md`, or configuration files unless specifically requested.
- Do NOT introduce new dependencies.
- Keep changes backward-compatible.
- Use `run_command` to verify changes with the test suite.
- Output TASK_COMPLETE when all improvements are implemented and tests pass."""

    def parse_result(self, result: AgentResult, working_dir: str,
                     project_dir: str = ".", **context: Any) -> dict[str, Any]:
        """Detect which files were changed by scanning git diff."""
        changed_files: list[str] = []
        try:
            output = subprocess.run(
                ["git", "-C", project_dir, "diff", "--name-only"],
                capture_output=True, text=True, timeout=5,
            )
            changed_files = [f.strip() for f in output.stdout.split("\n") if f.strip()]
        except (subprocess.TimeoutExpired, OSError):
            pass

        # Also check for untracked files (new files)
        try:
            output = subprocess.run(
                ["git", "-C", project_dir, "ls-files", "--others", "--exclude-standard"],
                capture_output=True, text=True, timeout=5,
            )
            untracked = [f.strip() for f in output.stdout.split("\n") if f.strip()]
            changed_files.extend(untracked)
        except (subprocess.TimeoutExpired, OSError):
            pass

        # Fallback: scan for recently modified .py files (within last 60 seconds)
        if not changed_files:
            cutoff = time.time() - 60
            for root, dirs, files in os.walk(project_dir):
                dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "node_modules", ".venv")]
                for f in files:
                    if f.endswith(".py"):
                        full = os.path.join(root, f)
                        try:
                            if os.path.getmtime(full) > cutoff:
                                changed_files.append(os.path.relpath(full, project_dir))
                        except OSError:
                            pass

        return {
            "changed_files": changed_files,
            "success": result.success,
        }