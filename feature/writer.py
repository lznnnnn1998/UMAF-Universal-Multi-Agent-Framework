"""FeatureReportWriterRole — terminal node, produces feature_report.md."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from tools import ToolRegistry
from agent import AgentResult, AgentRole
from utils import safe_read


class FeatureReportWriterRole(AgentRole):
    """Terminal node — writes feature_report.md."""

    agent_name = "feature_report_writer"
    max_steps = 5

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        if hasattr(ToolRegistry, "feature_writer_tools"):
            return ToolRegistry.to_dicts(ToolRegistry.feature_writer_tools())
        return ToolRegistry.to_dicts([ToolRegistry.WRITE_FILE])

    def build_task(self, backend: str, working_dir: str = ".",
                   changed_files: list[str] | None = None,
                   review_passed: bool = False,
                   review_issues: list[str] | None = None,
                   **context: Any) -> str:
        plan_path = os.path.join(working_dir, "implementation_plan.json")
        plan_display = safe_read(plan_path)[:5000] or "(no plan)"
        files_list = "\n".join(f"- {f}" for f in (changed_files or []))
        issues_list = "\n".join(f"- {i}" for i in (review_issues or []))

        return (
            f"You are a technical report writer. Produce a feature_report.md "
            f"summarizing the feature implementation.\n\n"
            f"## Feature\n{context.get('feature_description', 'Not specified')}\n\n"
            f"## Implementation Plan\n```json\n{plan_display}\n```\n\n"
            f"## Changed Files\n{files_list or '(none)'}\n\n"
            f"## Review Status\n"
            f"- Passed: {review_passed}\n"
            f"- Issues:\n{issues_list or '(none)'}\n\n"
            f"## Report Structure\n"
            f"1. **Summary** — what was implemented\n"
            f"2. **Files Created** — list each file with its purpose\n"
            f"3. **Files Modified** — list each file and what changed\n"
            f"4. **Conventions Followed** — naming, type hints, docstrings, imports\n"
            f"5. **Test Results** — test files written, coverage summary\n"
            f"6. **Review Results** — passed or issues found\n"
            f"7. **Known Limitations** — anything not addressed\n\n"
            f"Write feature_report.md to: {working_dir}\n"
            f"Output TASK_COMPLETE when done."
        )

    def parse_result(self, result: AgentResult, working_dir: str = ".",
                     **context: Any) -> dict[str, Any]:
        report_path = os.path.join(working_dir, "feature_report.md")
        if os.path.isfile(report_path):
            return {"feature_report": report_path}
        # Fallback: write a basic report
        return self._fallback_report(working_dir, **context)

    @staticmethod
    def _fallback_report(working_dir: str, changed_files: list[str] | None = None,
                         review_passed: bool = False,
                         review_issues: list[str] | None = None,
                         feature_description: str = "",
                         **context: Any) -> dict[str, Any]:
        """Write a deterministic feature_report.md."""
        files_list = "\n".join(f"- {f}" for f in (changed_files or []))
        issues_list = "\n".join(f"- {i}" for i in (review_issues or []))
        report = (
            f"# Feature Report\n\n"
            f"**Feature**: {feature_description or 'Not specified'}\n"
            f"**Generated**: {datetime.now(timezone.utc).isoformat()}\n"
            f"**Pipeline**: UMAF Feature Pipeline v2\n\n"
            f"## 1. Summary\n\n"
            f"Implementation of: {feature_description or 'Not specified'}\n\n"
            f"## 2. Files Changed\n\n{files_list or '(none reported)'}\n\n"
            f"## 3. Review Results\n\n"
            f"- **Passed**: {review_passed}\n"
            f"- **Issues**:\n{issues_list or '(none)'}\n\n"
            f"## 4. Known Limitations\n\n"
            f"This report was generated via deterministic fallback. "
            f"AI-assisted report generation was not performed.\n"
        )
        out_path = os.path.join(working_dir, "feature_report.md")
        try:
            with open(out_path, "w") as f:
                f.write(report)
        except OSError:
            pass
        return {"feature_report": out_path, "_fallback": True}
