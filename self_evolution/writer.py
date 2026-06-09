"""Self-evolution writer — produces an evolution report."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from agent import AgentResult, AgentRole
from tools import ToolRegistry


class SelfEvolutionWriterRole(AgentRole):
    """Write the self-evolution report documenting what UMAF changed about itself."""

    agent_name = "self_evolution_writer"
    max_steps = 8

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        return ToolRegistry.to_dicts(ToolRegistry.self_evolution_writer_tools())

    def build_task(self, backend: str, working_dir: str = "",
                   changed_files: list[str] | None = None,
                   review_passed: bool = False,
                   test_results: str = "",
                   **context: Any) -> str:
        files_str = "\n".join(f"  - {f}" for f in (changed_files or []))

        return f"""You are a self-evolution reporter for UMAF. Write an evolution report documenting the changes UMAF made to improve itself.

## Changed Files
{files_str if files_str else "No files changed."}

## Review Status
{"PASSED" if review_passed else "FAILED"}

## Test Results
```
{test_results if test_results else "No test results available."}
```

## Instructions
Write `evolution_report.md` in the working directory with:

1. **Summary**: What improvements were made and why.
2. **Changes**: List each changed file with a brief description of what was modified.
3. **Test Results**: Summary of test suite execution.
4. **Impact**: How these changes improve UMAF's reliability, performance, or capability.
5. **Next Steps**: Suggestions for future self-evolution iterations.

Keep the report concise (~200 words). Output TASK_COMPLETE when done."""

    def parse_result(self, result: AgentResult, working_dir: str,
                     changed_files: list[str] | None = None,
                     **context: Any) -> dict[str, Any]:
        report_path = os.path.join(working_dir, "evolution_report.md")
        if os.path.exists(report_path):
            return {"evolution_report": report_path}
        return self._fallback_report(working_dir, changed_files or [])

    @staticmethod
    def _fallback_report(working_dir: str, changed_files: list[str] | None = None) -> dict[str, Any]:
        """Write a minimal evolution report."""
        report_path = os.path.join(working_dir, "evolution_report.md")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if changed_files:
            changes_section = "\n".join(f"- {f}" for f in changed_files)
        else:
            changes_section = "_No automated changes were made in this cycle._"
        report = f"""# UMAF Self-Evolution Report

**Generated**: {timestamp}

## Summary
UMAF performed a self-evolution cycle, analyzing its own codebase and execution patterns to identify and implement improvements.

## Changes
{changes_section}

## Test Results
Verification was not performed (no changes to verify).

## Next Steps
1. Run with a specific improvement goal to make targeted changes.
2. Enable code modification mode to apply changes automatically.
3. Review and commit any generated improvements.
"""
        with open(report_path, "w") as f:
            f.write(report)
        return {"evolution_report": report_path}
