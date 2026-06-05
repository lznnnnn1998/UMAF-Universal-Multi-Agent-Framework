"""SkillReportWriterRole — produces structured JSON and markdown reports."""

import json
import os
from typing import Any

from agent import AgentResult, AgentRole
from tools import ToolRegistry
from utils import extract_json_object


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
                   project_name: str = "",
                   skill_inventory: dict[str, Any] | None = None,
                   **context: Any) -> str:
        """Build the report generation prompt."""
        proj = project_name or os.path.basename(
            os.path.dirname(working_dir)) or "Project"

        # Embed inventory summary inline so the writer doesn't need to
        # discover the file from disk.
        inventory_summary = ""
        if skill_inventory and (skill_inventory.get("tools") or skill_inventory.get("inferred_skills")):
            inv = skill_inventory
            summary = inv.get("summary", {})
            tools = inv.get("tools", inv.get("skills", []))
            skills = inv.get("inferred_skills", [])
            lines = [
                "\n## Skill Inventory (pre-computed — NO need to read from disk)",
                f"**Tools**: {summary.get('total_tools', len(tools))}",
                f"**Inferred skills**: {summary.get('total_inferred_skills', len(skills))}",
                f"**Artifact type**: {inv.get('artifact_type', 'unknown')}",
                "",
                "### Tools",
            ]
            for t in tools[:15]:
                lines.append(f"- **{t.get('name', '?')}** — {t.get('category', 'Other')} — {t.get('proficiency', 'beginner')}")
            if len(tools) > 15:
                lines.append(f"  ... and {len(tools) - 15} more tools")
            lines.append("")
            lines.append("### Inferred Skills")
            for s in skills[:15]:
                lines.append(f"- **{s.get('name', '?')}** — {s.get('category', 'Unknown')} — {s.get('proficiency', 'beginner')} ({s.get('confidence', '?')})")
            if len(skills) > 15:
                lines.append(f"  ... and {len(skills) - 15} more skills")
            lines.append("")
            inventory_summary = "\n".join(lines)

        common = (
            f"You are a technical report writer. Your job is to read a skill "
            f"inventory and produce two polished output files.\n\n"
            f"## Project\n{proj}\n"
            f"{inventory_summary}"
            f"## Input\n"
            f"The skill inventory (tools + inferred skills) is summarized above. "
            f"Read `skill_inventory.json` from the working directory "
            f"({working_dir}) for full details.\n\n"
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
        """Build skills.json from inventory data (v2: tools + inferred skills)."""
        from datetime import datetime, timezone

        tools = inventory.get("tools", inventory.get("skills", []))
        skills = inventory.get("inferred_skills", [])
        summary = inventory.get("summary", {})

        # Group tools by category
        tools_by_category: dict[str, list[dict[str, Any]]] = {}
        for t in tools:
            cat = t.get("category", "Other")
            tools_by_category.setdefault(cat, []).append({
                "name": t.get("name", ""),
                "proficiency": t.get("proficiency", "beginner"),
                "evidence": t.get("evidence", []),
            })

        # Group skills by category
        skills_by_category: dict[str, list[dict[str, Any]]] = {}
        for s in skills:
            cat = s.get("category", "Unknown")
            skills_by_category.setdefault(cat, []).append({
                "name": s.get("name", ""),
                "proficiency": s.get("proficiency", "beginner"),
                "confidence": s.get("confidence", "low"),
                "evidence": s.get("evidence", {}),
            })

        return {
            "project": project_name,
            "artifact_type": inventory.get("artifact_type", "unknown"),
            "generated_at": inventory.get("generated_at",
                datetime.now(timezone.utc).isoformat()),
            "summary": {
                "total_tools": summary.get("total_tools", len(tools)),
                "total_inferred_skills": summary.get("total_inferred_skills", len(skills)),
                "dimensions": summary.get("dimensions_covered", []),
                "proficiency_levels": summary.get("proficiency_distribution", {}),
            },
            "tools_by_category": tools_by_category,
            "skills_by_category": skills_by_category,
            "all_tools": [
                {"name": t.get("name", ""), "category": t.get("category", "Other"),
                 "proficiency": t.get("proficiency", "beginner"),
                 "sources": t.get("sources", [])}
                for t in tools
            ],
            "all_skills": [
                {"name": s.get("name", ""), "category": s.get("category", "Unknown"),
                 "proficiency": s.get("proficiency", "beginner"),
                 "confidence": s.get("confidence", "low")}
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
        """Generate a markdown report from v2 inventory (tools + inferred skills)."""
        from datetime import datetime, timezone

        tools = inventory.get("tools", inventory.get("skills", []))
        skills = inventory.get("inferred_skills", [])
        summary = inventory.get("summary", {})
        file_stats = inventory.get("file_stats", {})
        artifact_type = inventory.get("artifact_type", "unknown")

        lines: list[str] = []
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # Title
        lines.append(f"# 🔬 Skill Analysis Report: {project_name}")
        lines.append("")
        lines.append(f"**Generated**: {now}")
        lines.append(f"**Artifact Type**: {artifact_type}")
        lines.append(f"**Tools Detected**: {summary.get('total_tools', len(tools))}")
        lines.append(f"**Skills Inferred**: {summary.get('total_inferred_skills', len(skills))}")
        lines.append("")

        # Executive Summary
        lines.append("## 📊 Summary")
        lines.append("")
        dims = summary.get("dimensions_covered", [])
        if dims:
            lines.append(f"**Dimensions Analyzed**: {', '.join(dims)}")
            lines.append("")

        # Proficiency distribution
        prof_dist = summary.get("proficiency_distribution", {})
        if prof_dist:
            lines.append("### Proficiency Distribution (tools + skills)")
            lines.append("")
            lines.append("| Level | Count |")
            lines.append("|-------|-------|")
            for level in ("expert", "advanced", "intermediate", "beginner"):
                count = prof_dist.get(level, 0)
                badge = _BADGES.get(level, level)
                bar = "█" * max(1, count)
                lines.append(f"| {badge} | {bar} ({count}) |")
            lines.append("")

        # Detected Tools
        if tools:
            lines.append("## 🛠️ Detected Tools")
            lines.append("")
            lines.append("| Tool | Category | Proficiency |")
            lines.append("|------|----------|-------------|")
            for t in tools[:30]:
                name = t.get("name", "?")
                cat = t.get("category", "Other")
                prof = t.get("proficiency", "beginner")
                badge = _BADGES.get(prof, prof)
                lines.append(f"| {name} | {cat} | {badge} |")
            if len(tools) > 30:
                lines.append(f"| ... | {len(tools) - 30} more tools | |")
            lines.append("")

        # Inferred Skills
        if skills:
            lines.append("## 🧠 Inferred Developer Skills")
            lines.append("")

            # Group by category
            by_category: dict[str, list[dict[str, Any]]] = {}
            for s in skills:
                cat = s.get("category", "Unknown")
                by_category.setdefault(cat, []).append(s)

            for cat in sorted(by_category.keys()):
                cat_skills = by_category[cat]
                lines.append(f"### {cat} ({len(cat_skills)})")
                lines.append("")
                lines.append("| Skill | Proficiency | Confidence |")
                lines.append("|-------|-------------|------------|")
                for sk in cat_skills:
                    name = sk.get("name", "?")
                    prof = sk.get("proficiency", "beginner")
                    conf = sk.get("confidence", "?")
                    badge = _BADGES.get(prof, prof)
                    lines.append(f"| {name} | {badge} | {conf} |")
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

        has_expert = prof_dist.get("expert", 0) > 0
        beginner_count = prof_dist.get("beginner", 0)

        if not has_expert:
            lines.append("- No skills at expert level — consider deepening expertise.")
        if beginner_count > 3:
            lines.append(f"- {beginner_count} entries at beginner level — room for growth.")
        if not skills:
            lines.append("- No inferred skills found — may need more content to analyze.")
        if not tools:
            lines.append("- No tools detected — may not be a software project.")

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*Report generated by UMAF Skill Pipeline v2.*")
        lines.append("")

        # Write to disk
        path = os.path.join(working_dir, "skills_report.md")
        with open(path, "w") as f:
            f.write("\n".join(lines))

        return path
