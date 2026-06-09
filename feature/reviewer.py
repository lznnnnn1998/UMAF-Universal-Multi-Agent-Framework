"""FeatureReviewerRole — validates implementation via REVIEW_PASSED/REVIEW_FAILED tokens."""

from __future__ import annotations

import os
from typing import Any

from tools import ToolRegistry
from agent import AgentResult, AgentRole
from utils import safe_read, scan_review_verdict


class FeatureReviewerRole(AgentRole):
    """Reviews implementation: completeness, correctness, conventions, tests."""

    agent_name = "feature_reviewer"
    max_steps = 10

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        if hasattr(ToolRegistry, "feature_reviewer_tools"):
            return ToolRegistry.to_dicts(ToolRegistry.feature_reviewer_tools())
        return ToolRegistry.to_dicts([
            ToolRegistry.READ_FILE, ToolRegistry.RUN_COMMAND,
        ])

    def build_task(self, backend: str, working_dir: str = ".",
                   changed_files: list[str] | None = None,
                   **context: Any) -> str:
        plan_path = os.path.join(working_dir, "implementation_plan.json")
        ctx_path = os.path.join(working_dir, "project_context.json")

        plan_display = safe_read(plan_path)[:6000] or "(no plan)"
        ctx_display = safe_read(ctx_path)[:4000] or "(no context)"
        files_list = "\n".join(f"- {f}" for f in (changed_files or []))

        return (
            f"You are a code reviewer. Review ALL files produced by the "
            f"implementation phase.\n\n"
            f"## Feature\n{context.get('feature_description', 'Not specified')}\n\n"
            f"## Changed Files\n{files_list or '(none reported)'}\n\n"
            f"## Project Context (conventions to verify against)\n"
            f"```json\n{ctx_display}\n```\n\n"
            f"## Implementation Plan\n```json\n{plan_display}\n```\n\n"
            f"## Review Dimensions\n\n"
            f"### 1. Completeness\n"
            f"- Have ALL files_to_create been written?\n"
            f"- Have ALL files_to_modify been correctly edited?\n"
            f"- Are ALL test_files present?\n\n"
            f"### 2. Correctness\n"
            f"- Read each changed file: do imports resolve?\n"
            f"- Are there circular dependencies?\n"
            f"- Do tests pass when run?\n"
            f"- Are all interfaces from the plan actually implemented?\n\n"
            f"### 3. Convention Compliance\n"
            f"- Does naming match project_context conventions?\n"
            f"- Are type hints used consistently?\n"
            f"- Are docstrings present in the correct format?\n"
            f"- Does import style match the project?\n\n"
            f"### 4. Test Quality\n"
            f"- Do tests cover happy path, edge cases, and error handling?\n"
            f"- Are tests properly structured with fixtures?\n"
            f"- Do tests assert meaningful behavior?\n\n"
            f"## Instructions\n"
            f"1. Read implementation_plan.json and project_context.json\n"
            f"2. Read every changed file\n"
            f"3. Run the tests to verify they pass\n"
            f"4. Output EXACTLY ONE of these tokens:\n"
            f"   - REVIEW_PASSED — all checks passed, code is ready\n"
            f"   - REVIEW_FAILED — issues found (list each issue clearly)\n"
            f"5. Output TASK_COMPLETE\n\n"
            f"IMPORTANT: Include EXACTLY 'REVIEW_PASSED' or 'REVIEW_FAILED' "
            f"in your response (not both)."
        )

    def parse_result(self, result: AgentResult, working_dir: str = ".",
                     **context: Any) -> dict[str, Any]:
        verdict = scan_review_verdict(result.messages)
        review_passed = verdict is True
        review_issues: list[str] = []

        if verdict is False:
            # Collect issues from the REVIEW_FAILED message
            for msg in reversed(result.messages):
                content = msg.content if hasattr(msg, "content") else str(msg)
                if type(msg).__name__ != "AIMessage":
                    continue
                if "REVIEW_FAILED" in content:
                    for line in content.split("\n"):
                        stripped = line.strip()
                        if stripped.startswith("- ") or stripped.startswith("* "):
                            review_issues.append(stripped[2:])
                        elif "issue" in stripped.lower() or "fail" in stripped.lower():
                            if len(stripped) > 10:
                                review_issues.append(stripped)
                    break

        return {
            "review_passed": review_passed,
            "review_issues": review_issues[:20],  # cap at 20
        }
