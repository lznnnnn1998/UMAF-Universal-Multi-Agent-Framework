"""SkillReportWriterRole — produces structured JSON and markdown reports."""

import json
import os
import sys
from typing import Any

# Ensure repo root is on sys.path so we can import agent.py and tools.py
# __file__ is .../coderpp_output/modules/skill_agent_roles/skill/writer.py
# Repo root is 5 dirs up: .../universal_multi_agent_framework/
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agent import AgentResult, AgentRole  # noqa: E402
from tools import ToolRegistry  # noqa: E402


# Proficiency badge mapping
_BADGES: dict[str, str] = {
    "expert": "🟣 Expert",
    "advanced": "🟢 Advanced",
    "intermediate": "🟡 Intermediate",
    "beginner": "⚪ Beginner",
}

# Category emoji mapping
_CATEGORY_EMOJI: dict[str, str] = {
    "Languages & Runtimes": "🔤",
    "Web Frameworks": "🌐",
    "Frontend": "🎨",
    "Backend": "⚙️",
    "Testing": "🧪",
    "Code Quality": "✅",
    "Data Science & ML": "🧠",
    "Build & Tooling": "🔧",
    "Containerization": "📦",
    "Orchestration": "☸️",
    "CI/CD": "🔄",
    "Cloud": "☁️",
    "Infrastructure as Code": "🏗️",
    "Monitoring & Observability": "📊",
    "Web Servers": "🖥️",
    "Databases": "🗄️",
    "Documentation": "📝",
    "Configuration Management": "⚙️",
    "API Specifications": "📋",
    "Other": "📌",
}


class SkillReportWriterRole(AgentRole):
    """Read the aggregated skill inventory and produce two final output files:

    - ``skills.json``: Structured JSON skill report
    - ``skills_report.md``: Human-readable markdown report with proficiency
      badges, category breakdowns, and project statistics.

    Both files are written to the working directory.
    """

    agent_name: str = "skill_report_writer"
    max_steps: int = 8

    # ── Tools ───────────────────────────────────────────────────────────

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        """Only need Read and Write — reads inventory, writes reports."""
        return ToolRegistry.to_dicts(ToolRegistry.skill_writer_tools())

    # ── Task prompt ─────────────────────────────────────────────────────

    def build_task(self, backend: str, working_dir: str = ".",
                   project_name: str = "", **context: Any) -> str:
        """Build the report generation prompt."""
        proj = project_name or os.path.basename(
            os.path.dirname(working_dir)) or "Project"

        common = (
            f"You are a technical report writer. Your job is to read a skill "
            f"inventory and produce two polished output files.\n\n"
            f"## Project\n{proj}\n\n"
            f"## Input\n"
            f"Read `skill_inventory.json` from the working directory "
            f"({working_dir}).\n\n"
            f"## Output 1: `skills.json`\n"
            f"A structured JSON report with this schema:\n"
            f"```json\n"
            f"{{\n"
            f'  "project": "{proj}",\n'
            f'  "generated_at": "<ISO timestamp>",\n'
            f'  "summary": {{\n'
            f'    "total_skills": <int>,\n'
            f'    "domains": ["Python", "JavaScript", ...],\n'
            f'    "top_category": "Web Frameworks",\n'
            f'    "proficiency_levels": {{\n'
            f'      "expert": 0, "advanced": 5, '
            f'"intermediate": 3, "beginner": 2\n'
            f'    }}\n'
            f'  }},\n'
            f'  "skills_by_category": {{\n'
            f'    "Web Frameworks": [\n'
            f'      {{"name": "Django", "proficiency": "advanced", '
            f'"evidence": ["manage.py"]}}\n'
            f'    ]\n'
            f'  }},\n'
            f'  "all_skills": [\n'
            f'    {{"name": "Django", "category": "Web Frameworks", '
            f'"proficiency": "advanced", "version": "4.x"}}\n'
            f'  ]\n'
            f"}}\n"
            f"```\n\n"
            f"## Output 2: `skills_report.md`\n"
            f"A markdown report with:\n"
            f"- Title and generation date\n"
            f"- Executive summary with key stats\n"
            f"- Proficiency distribution chart (text-based)\n"
            f"- Skills by category sections with emojis and badges\n"
            f"- Each skill listed with: name, proficiency badge, version hint, "
            f"evidence files\n"
            f"- File statistics table\n"
            f"- Recommendations section\n\n"
            f"Use these proficiency badges in the markdown:\n"
            f"- Expert: `🟣 Expert`\n"
            f"- Advanced: `🟢 Advanced`\n"
            f"- Intermediate: `🟡 Intermediate`\n"
            f"- Beginner: `⚪ Beginner`\n\n"
            f"Write both files, then output TASK_COMPLETE."
        )

        if backend == "claude_cli":
            backend_note = (
                "\n\nUse your own knowledge — do NOT search the web. "
                "Read skill_inventory.json, write skills.json and "
                "skills_report.md. Output TASK_COMPLETE."
            )
        else:
            backend_note = (
                "\n\nRead skill_inventory.json, write skills.json and "
                "skills_report.md, output TASK_COMPLETE."
            )

        return common + backend_note

    # ── Parse result ────────────────────────────────────────────────────

    def parse_result(self, result: AgentResult, working_dir: str,
                     project_name: str = "", **context: Any) -> dict[str, Any]:
        """Verify output files exist and return paths. Falls back to
        template-based generation if needed."""

        skills_json_exists = os.path.exists(
            os.path.join(working_dir, "skills.json"))
        report_md_exists = os.path.exists(
            os.path.join(working_dir, "skills_report.md"))

        if skills_json_exists and report_md_exists:
            # Load skills.json for return
            try:
                with open(os.path.join(working_dir, "skills.json")) as f:
                    skills_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                skills_data = {}
            return {
                "skills_json_path": os.path.join(working_dir, "skills.json"),
                "report_md_path": os.path.join(working_dir, "skills_report.md"),
                "skills_data": skills_data,
            }

        # Fallback: generate from inventory
        proj = project_name or os.path.basename(
            os.path.dirname(working_dir)) or "Project"
        inventory = self._load_inventory(working_dir)
        if inventory:
            skills_data = self._fallback_skills_json(proj, inventory)
            self._write_skills_json(working_dir, skills_data)
            self._fallback_report_md(proj, inventory, working_dir)
        else:
            skills_data = {}

        return {
            "skills_json_path": os.path.join(working_dir, "skills.json"),
            "report_md_path": os.path.join(working_dir, "skills_report.md"),
            "skills_data": skills_data,
        }

    # ── Fallback writers ────────────────────────────────────────────────

    @staticmethod
    def _load_inventory(working_dir: str) -> dict[str, Any] | None:
        """Load skill_inventory.json from disk."""
        path = os.path.join(working_dir, "skill_inventory.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def _fallback_skills_json(project_name: str,
                              inventory: dict[str, Any]) -> dict[str, Any]:
        """Build skills.json from inventory data."""
        from datetime import datetime, timezone

        skills = inventory.get("skills", [])
        summary = inventory.get("summary", {})
        cats = summary.get("skill_categories", {})

        # Group by category
        skills_by_category: dict[str, list[dict[str, Any]]] = {}
        for s in skills:
            cat = s.get("category", "Other")
            skills_by_category.setdefault(cat, []).append({
                "name": s.get("name", ""),
                "proficiency": s.get("proficiency", "beginner"),
                "evidence": s.get("evidence", []),
                "version_hint": s.get("version_hint", ""),
            })

        top_cat = max(cats, key=cats.get) if cats else "Other"

        return {
            "project": project_name,
            "generated_at": inventory.get("generated_at",
                datetime.now(timezone.utc).isoformat()),
            "summary": {
                "total_skills": summary.get("total_skills", len(skills)),
                "domains": summary.get("domains_covered", []),
                "top_category": top_cat,
                "proficiency_levels": summary.get("proficiency_distribution", {}),
            },
            "skills_by_category": skills_by_category,
            "all_skills": [
                {
                    "name": s.get("name", ""),
                    "category": s.get("category", "Other"),
                    "proficiency": s.get("proficiency", "beginner"),
                    "version": s.get("version_hint", ""),
                }
                for s in skills
            ],
            "file_stats": inventory.get("file_stats", {}),
        }

    @staticmethod
    def _write_skills_json(working_dir: str, data: dict[str, Any]) -> str:
        """Write skills.json to disk."""
        path = os.path.join(working_dir, "skills.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    @staticmethod
    def _fallback_report_md(project_name: str, inventory: dict[str, Any],
                            working_dir: str) -> str:
        """Generate a template-based markdown report from inventory data."""
        from datetime import datetime, timezone

        skills = inventory.get("skills", [])
        summary = inventory.get("summary", {})
        file_stats = inventory.get("file_stats", {})

        # Build markdown sections
        lines: list[str] = []
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # Title
        lines.append(f"# 🔬 Skill Summary Report: {project_name}")
        lines.append("")
        lines.append(f"**Generated**: {now}")
        lines.append(f"**Total Skills Detected**: {summary.get('total_skills', len(skills))}")
        lines.append("")

        # Executive Summary
        lines.append("## 📊 Executive Summary")
        lines.append("")
        domains = summary.get("domains_covered", [])
        lines.append(f"**Domains Analyzed**: {', '.join(domains) if domains else 'N/A'}")
        lines.append("")

        # Proficiency distribution
        prof_dist = summary.get("proficiency_distribution", {})
        if prof_dist:
            lines.append("### Proficiency Distribution")
            lines.append("")
            lines.append("| Level | Count |")
            lines.append("|-------|-------|")
            for level in ("expert", "advanced", "intermediate", "beginner"):
                count = prof_dist.get(level, 0)
                badge = _BADGES.get(level, level)
                bar = "█" * max(1, count)
                lines.append(f"| {badge} | {bar} ({count}) |")
            lines.append("")

        # Skills by Category
        lines.append("## 🏷️ Skills by Category")
        lines.append("")

        # Group skills
        by_category: dict[str, list[dict[str, Any]]] = {}
        for s in skills:
            cat = s.get("category", "Other")
            by_category.setdefault(cat, []).append(s)

        for cat in sorted(by_category.keys()):
            cat_skills = by_category[cat]
            emoji = _CATEGORY_EMOJI.get(cat, "📌")
            lines.append(f"### {emoji} {cat} ({len(cat_skills)})")
            lines.append("")
            lines.append("| Skill | Proficiency | Version | Evidence |")
            lines.append("|-------|-------------|---------|----------|")
            for sk in cat_skills:
                name = sk.get("name", "?")
                prof = sk.get("proficiency", "beginner")
                badge = _BADGES.get(prof, prof)
                ver = sk.get("version_hint", "") or "-"
                ev = ", ".join(sk.get("evidence", [])[:3]) or "-"
                if len(ev) > 80:
                    ev = ev[:77] + "..."
                lines.append(f"| {name} | {badge} | {ver} | {ev} |")
            lines.append("")

        # File Statistics
        if file_stats:
            lines.append("## 📁 File Statistics")
            lines.append("")
            lines.append("| Metric | Count |")
            lines.append("|--------|-------|")
            for key, label in [
                ("total_files", "Total Files"),
                ("python_files", "Python Files"),
                ("javascript_files", "JavaScript Files"),
                ("typescript_files", "TypeScript Files"),
                ("test_files", "Test Files"),
                ("config_files", "Configuration Files"),
                ("doc_files", "Documentation Files"),
            ]:
                val = file_stats.get(key)
                if val is not None:
                    lines.append(f"| {label} | {val} |")
            lines.append("")

        # Recommendations
        lines.append("## 💡 Recommendations")
        lines.append("")

        # Generate recommendations based on proficiency gaps
        has_expert = prof_dist.get("expert", 0) > 0
        has_advanced = prof_dist.get("advanced", 0) > 0
        beginner_count = prof_dist.get("beginner", 0)

        if not has_expert:
            lines.append("- Consider deepening expertise in key technologies to reach expert level.")
        if beginner_count > 3:
            lines.append(f"- {beginner_count} skills at beginner level — prioritize learning paths for these areas.")
        if not file_stats:
            pass
        elif file_stats.get("test_files", 0) == 0:
            lines.append("- ⚠️ No test files detected — consider adding automated testing.")
        if "Infrastructure" not in str(domains):
            lines.append("- No infrastructure tooling detected — consider adding Docker or CI/CD.")

        if len([r for r in lines if r.startswith("- ")]) == 0:
            lines.append("- The project shows a well-rounded skill profile. Continue maintaining and updating dependencies.")

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*Report generated by UMAF Skill Summarizer Pipeline.*")

        # Write to disk
        path = os.path.join(working_dir, "skills_report.md")
        with open(path, "w") as f:
            f.write("\n".join(lines))

        return path

    # ── JSON extraction helper ──────────────────────────────────────────

    @staticmethod
    def _extract_json_object(text: str) -> str | None:
        """Extract the first complete JSON object from text."""
        start = text.find('{')
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == '\\':
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None
