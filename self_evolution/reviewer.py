"""Self-evolution reviewer — verifies changes by running the test suite."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from agent import AgentResult, AgentRole
from tools import ToolRegistry
from utils import scan_review_verdict


class SelfEvolutionReviewerRole(AgentRole):
    """Verify self-evolution changes by running tests and checking for regressions."""

    agent_name = "self_evolution_reviewer"
    max_steps = 12

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        return ToolRegistry.to_dicts(ToolRegistry.self_evolution_reviewer_tools())

    def build_task(self, backend: str, working_dir: str = "",
                   project_dir: str = ".", changed_files: list[str] | None = None,
                   **context: Any) -> str:
        files_list = ""
        if changed_files:
            files_list = "\n".join(f"  - {f}" for f in changed_files[:30])
            if len(changed_files) > 30:
                files_list += f"\n  ... and {len(changed_files) - 30} more files"

        return f"""You are a self-evolution reviewer for UMAF. Your job is to verify that code changes are correct, safe, and don't break anything.

## Project Directory
{project_dir}

## Changed Files
{files_list if files_list else "No file list provided — scan git diff to find changes."}

## Review Steps

### 1. Run the Test Suite
Run `python -m pytest test/ -q` to verify all tests pass.

### 2. Check Test Results
- All tests must pass (0 failures)
- No new test failures compared to baseline
- If tests fail, identify the specific failures

### 3. Review Code Quality
- Read a sample of changed files
- Check for:
  - Correct Python >= 3.11 syntax (X | None, not Optional[X])
  - No unused imports
  - No dead code
  - Proper error handling at system boundaries

### 4. Output Verdict
- If all tests pass → write REVIEW_PASSED in your response.
- If issues found → describe each issue and write REVIEW_FAILED.

Use `run_command` to execute the test suite. Use `read_file` to review changed files.

Be thorough but fair. If the tests pass and the code follows conventions, this is a PASS."""

    def parse_result(self, result: AgentResult, working_dir: str,
                     changed_files: list[str] | None = None,
                     **context: Any) -> dict[str, Any]:
        if not result.success:
            return {"review_passed": False, "review_issues": ["Agent did not complete successfully."],
                    "test_results": "", "changed_files": changed_files or []}

        verdict = scan_review_verdict(result.messages)
        review_passed = verdict is True
        review_issues: list[str] = []
        test_results = ""

        # Extract test results from messages
        for msg in reversed(result.messages):
            content = getattr(msg, "content", None)
            if content is None:
                content = str(msg)
            if not isinstance(content, str):
                continue
            test_match = re.search(
                r"(\d+\s+passed.*?)(?:\s+in\s+[\d.]+s)?",
                content,
            )
            if test_match:
                test_results = test_match.group(1)
                break

        # Collect issues when review failed
        if verdict is False:
            for msg in reversed(result.messages):
                content = getattr(msg, "content", None)
                if content is None:
                    content = str(msg)
                if not isinstance(content, str):
                    continue
                if "REVIEW_FAILED" in content:
                    for line in content.split("\n"):
                        stripped = line.strip()
                        if stripped.startswith("- ") or stripped.startswith("* "):
                            review_issues.append(stripped[2:])
                    if not review_issues:
                        review_issues.append("Review failed — review message for details.")
                    break

        return {
            "review_passed": review_passed,
            "review_issues": review_issues,
            "test_results": test_results,
            "changed_files": changed_files or [],
        }