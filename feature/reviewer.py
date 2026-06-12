"""FeatureReviewerRole — validates implementation via REVIEW_PASSED/REVIEW_FAILED tokens.

When all_coder_outputs is provided (multi-coder mode), also performs cross-coder
integration verification to ensure modules produced by different coders integrate
correctly.
"""

from __future__ import annotations

import os
from typing import Any

from tools import ToolRegistry
from agent import AgentResult, AgentRole
from utils import safe_read, scan_review_verdict


class FeatureReviewerRole(AgentRole):
    """Reviews implementation: completeness, correctness, conventions, tests,
    and cross-coder integration (when multiple coders are used)."""

    agent_name = "feature_reviewer"
    max_steps = 10

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        if hasattr(ToolRegistry, "feature_reviewer_tools"):
            return ToolRegistry.to_dicts(ToolRegistry.feature_reviewer_tools())
        return ToolRegistry.to_dicts([
            ToolRegistry.READ_FILE, ToolRegistry.RUN_COMMAND,
        ])

    def build_task(self, backend: str, working_dir: str = ".",
                   project_dir: str = ".", changed_files: list[str] | None = None,
                   **context: Any) -> str:
        plan_path = os.path.join(working_dir, "implementation_plan.json")
        ctx_path = os.path.join(working_dir, "project_context.json")
        decomp_path = os.path.join(working_dir, "decomposition.json")

        plan_display = safe_read(plan_path)[:6000] or "(no plan)"
        ctx_display = safe_read(ctx_path)[:4000] or "(no context)"
        files_list = "\n".join(f"- {f}" for f in (changed_files or []))

        # Build cross-coder integration section when multiple coders were used
        all_coder_outputs: list[dict[str, Any]] | None = context.get("all_coder_outputs")
        cross_coder_section = ""
        if all_coder_outputs and len(all_coder_outputs) > 1:
            # Read decomposition to get dependency structure
            decomp_display = safe_read(decomp_path)[:6000] or ""

            cross_coder_section = (
                f"\n\n## Cross-Coder Integration Review\n"
                f"Multiple coders worked on this feature in parallel. Verify that "
                f"their outputs integrate correctly:\n\n"
                f"### Decomposition\n"
                f"```json\n{decomp_display}\n```\n\n"
                f"### Coder Outputs (per module)\n"
            )
            for out in all_coder_outputs:
                mod = out.get("module_name", "?")
                files = out.get("files", [])
                dep_verif = out.get("dependency_verification")
                summary = out.get("summary", "")
                cross_coder_section += (
                    f"\n**{mod}**:\n"
                    f"  - Files: {files}\n"
                    f"  - Dependency verification: {dep_verif}\n"
                    f"  - Summary: {summary[:200]}\n"
                )

            cross_coder_section += (
                f"\n### Integration Checks\n"
                f"1. **Dependency consumption**: For each coder that had "
                f"dependencies, verify it correctly consumed upstream outputs.\n"
                f"   - Read files from the coder and verify imports resolve to files "
                f"produced by earlier coders\n"
                f"   - Check that any config/data files produced by level[i-1] are "
                f"correctly referenced by level[i]\n"
                f"2. **Import resolution across modules**: Check that imports between "
                f"modules produced by DIFFERENT coders resolve correctly.\n"
                f"   - If module A (coder 1) imports from module B (coder 2), verify "
                f"that B's files exist and export the expected symbols\n"
                f"3. **Interface matching**: Verify that interfaces/APIs between "
                f"dependent modules match.\n"
                f"   - Check function signatures, class constructors, and return types\n"
                f"   - Verify data structures passed between modules are compatible\n"
                f"4. **Data flow through module chain**: Trace data from entry points "
                f"through the full module chain.\n"
                f"   - Verify data transformations are correct at each step\n"
                f"5. **Integration tests**: If test files exist that span module "
                f"boundaries, run them from `{project_dir}`.\n"
                f"   - Run: `cd {project_dir} && python -m pytest test/ -v`\n"
            )

        prompt = (
            f"You are a code reviewer. Review ALL files produced by the "
            f"implementation phase.\n\n"
            f"## Feature\n{context.get('feature_description', 'Not specified')}\n\n"
            f"## File Locations\n"
            f"- Source code is in the **project directory**: `{project_dir}`\n"
            f"- Meta files (plan, context) are in the **output directory**: `{working_dir}`\n"
            f"- Read all source/test files from `{project_dir}/<path>`\n\n"
            f"## Changed Files\n{files_list or '(none reported)'}\n\n"
            f"## Project Context (conventions to verify against)\n"
            f"```json\n{ctx_display}\n```\n\n"
            f"## Implementation Plan\n```json\n{plan_display}\n```\n\n"
            f"## Review Dimensions\n\n"
            f"### 1. Completeness\n"
            f"- Have ALL files_to_create been written under `{project_dir}`?\n"
            f"- Have ALL files_to_modify been correctly edited in `{project_dir}`?\n"
            f"- Are ALL test_files present under `{project_dir}`?\n\n"
            f"### 2. Correctness\n"
            f"- Read each changed file from `{project_dir}`: do imports resolve?\n"
            f"- Are there circular dependencies?\n"
            f"- Do tests pass when run from `{project_dir}`?\n"
            f"- Are all interfaces from the plan actually implemented?\n\n"
            f"### 3. Convention Compliance\n"
            f"- Does naming match project_context conventions?\n"
            f"- Are type hints used consistently?\n"
            f"- Are docstrings present in the correct format?\n"
            f"- Does import style match the project?\n\n"
            f"### 4. Test Quality\n"
            f"- Do tests cover happy path, edge cases, and error handling?\n"
            f"- Are tests properly structured with fixtures?\n"
            f"- Do tests assert meaningful behavior?\n"
            f"{cross_coder_section}\n"
            f"## Instructions\n"
            f"1. Read implementation_plan.json and project_context.json from `{working_dir}`\n"
            f"2. Read every changed file from `{project_dir}`\n"
            f"3. Run the tests from `{project_dir}` to verify they pass\n"
            f"4. Output EXACTLY ONE of these tokens:\n"
            f"   - REVIEW_PASSED — all checks passed, code is ready\n"
            f"   - REVIEW_FAILED — issues found (list each issue clearly)\n"
            f"5. Output TASK_COMPLETE\n\n"
            f"IMPORTANT: Include EXACTLY 'REVIEW_PASSED' or 'REVIEW_FAILED' "
            f"in your response (not both)."
        )

        if backend == "claude_cli":
            prompt += "\n\nUse Read tools to review files. Run tests with Bash."
        else:
            prompt += "\n\nRead files and run tests to verify."

        return prompt

    def parse_result(self, result: AgentResult, working_dir: str = ".",
                     **context: Any) -> dict[str, Any]:
        verdict = scan_review_verdict(result.messages)
        review_passed = verdict is True
        review_issues: list[str] = []
        cross_coder_issues: list[dict[str, Any]] = []

        if verdict is False:
            # Collect issues from the REVIEW_FAILED message
            for msg in reversed(result.messages):
                content = msg.content if hasattr(msg, "content") else str(msg)
                if type(msg).__name__ != "AIMessage":
                    continue
                if "REVIEW_FAILED" in content:
                    for line in content.split("\n"):
                        stripped = line.strip()
                        # ── Cross-coder / integration issues ──────────────
                        if (stripped.startswith("CROSS_CODER_ISSUE:")
                                or stripped.startswith("INTEGRATION_ISSUE:")):
                            # Parse colon-separated format:
                            #   CROSS_CODER_ISSUE: <module_name>: <description>
                            #   INTEGRATION_ISSUE: <description>
                            prefix_end = stripped.index(":") + 1
                            rest = stripped[prefix_end:].strip()
                            # Check for module name prefix (word chars + dots)
                            parts = rest.split(":", 1)
                            if len(parts) == 2 and parts[0].strip().isidentifier():
                                cross_coder_issues.append({
                                    "type": "cross_coder",
                                    "module": parts[0].strip(),
                                    "description": parts[1].strip(),
                                })
                            else:
                                cross_coder_issues.append({
                                    "type": "cross_coder",
                                    "module": None,
                                    "description": rest,
                                })
                        elif stripped.startswith("- ") or stripped.startswith("* "):
                            review_issues.append(stripped[2:])
                        elif "issue" in stripped.lower() or "fail" in stripped.lower():
                            if len(stripped) > 10:
                                # Don't duplicate if already caught as cross-coder
                                if not stripped.startswith("CROSS_CODER_ISSUE"):
                                    review_issues.append(stripped)
                    break

        return {
            "review_passed": review_passed,
            "review_issues": review_issues[:20],  # cap at 20
            "cross_coder_issues": cross_coder_issues[:20],
        }
