"""SkillReportWriterRole — produces structured JSON and markdown reports.

v2 improvements:
- Artifact-type-aware section ordering in markdown reports
- Skill Gap Analysis identifying notably absent skills for the artifact type
- skill_graph passthrough from aggregator to skills.json
- Updated build_task prompt requesting type-specific sections
"""

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

# ═══════════════════════════════════════════════════════════════════════════
# v2: Expected skill areas per artifact type for gap analysis
# ═══════════════════════════════════════════════════════════════════════════

# Each area has indicator skill/tool names — if ANY indicator is found,
# the area is considered "covered". Otherwise it's a gap.
_ARTIFACT_EXPECTED_AREAS: dict[str, dict[str, list[str]]] = {
    "software_project": {
        "Testing": [
            "Testing Strategy", "Test Coverage Thoroughness",
            "pytest", "Jest", "Vitest", "Playwright", "MSW", "unittest",
        ],
        "CI/CD": [
            "CI/CD Sophistication",
            "GitHub Actions", "GitLab CI", "Jenkins", "CircleCI",
        ],
        "Version Control": [
            "Git Workflow Maturity", "Git", "GitHub",
        ],
        "Environment Management": [
            "Environment Management", "Docker", "Kubernetes",
        ],
        "Code Quality": [
            "Code Quality Enforcement", "Design Patterns",
            "Error Handling Maturity", "ESLint", "Prettier", "Ruff", "Biome",
        ],
    },
    "research_paper": {
        "Academic Rigor": [
            "Academic Rigor",
        ],
        "Citations & References": [
            "Citations", "References",
        ],
        "Methodology": [
            "Methodology", "Research Design",
        ],
        "Domain Expertise": [
            "Machine Learning", "Scientific Computing",
            "Natural Language Processing", "Computer Vision",
            "Reinforcement Learning", "Data Engineering",
            "Distributed Systems", "Security", "Compiler Design",
            "Database Systems", "Game Development", "Finance",
            "Networking", "Operating Systems", "Embedded Systems",
            "DevOps", "Frontend Development", "Mobile Development",
            "Blockchain",
        ],
    },
    "blog_article": {
        "Clarity": ["Clarity"],
        "Argumentation": ["Argumentation"],
        "Technical Writing": ["Technical Writing", "Documentation Quality"],
        "Domain Knowledge": [
            "Machine Learning", "Data Engineering", "Frontend Development",
            "Security", "DevOps", "Distributed Systems",
        ],
    },
    "documentation": {
        "Technical Writing": ["Technical Writing", "Documentation Quality"],
        "Clarity": ["Clarity"],
        "Code Quality": ["Code Quality Enforcement"],
        "Testing": ["Testing Strategy"],
    },
    "dataset": {
        "Data Engineering": ["Data Engineering"],
        "Documentation Quality": ["Documentation Quality"],
        "Testing": ["Testing Strategy"],
    },
    "design_document": {
        "Architecture": [
            "Component Design", "API Design", "Data Modeling",
            "Scalability", "Infrastructure Design",
        ],
        "Technical Writing": ["Technical Writing", "Clarity"],
        "Domain Knowledge": [
            "Distributed Systems", "Networking", "Database Systems",
        ],
    },
    "presentation": {
        "Narrative Structure": ["Narrative Structure"],
        "Clarity": ["Clarity"],
        "Technical Writing": ["Technical Writing"],
    },
    "configuration": {
        "Infrastructure as Code": [
            "Infrastructure Design", "Terraform", "Ansible",
        ],
        "Environment Management": ["Environment Management", "Docker"],
        "CI/CD": ["CI/CD Sophistication"],
    },
}

# v2: Display ordering of inferred-skill categories by artifact type.
# Categories not listed appear after the ordered ones, in alphabetical order.
_ARTIFACT_CATEGORY_ORDER: dict[str, list[str]] = {
    "software_project": [
        "Testing", "Languages & Runtimes", "Web Frameworks",
        "Containers & Orchestration", "Package Management",
        "Build & Tooling", "Code Quality", "Version Control",
        "CSS & Styling", "API & Data Layer", "State Management",
        "Infrastructure as Code", "Data Science & ML",
        "Documentation Tools", "Monitoring & Observability",
    ],
    "research_paper": [
        "Domain Knowledge", "Quality Assurance", "Engineering Practice",
        "Technical Craft", "Architecture",
    ],
    "blog_article": [
        "Technical Craft", "Quality Assurance",
        "Domain Knowledge", "Engineering Practice", "Architecture",
    ],
}

# v2: Tool category display priority for tools table grouping by artifact type
_ARTIFACT_TOOL_CATEGORY_ORDER: dict[str, list[str]] = {
    "software_project": [
        "Testing", "CI/CD", "Languages & Runtimes", "Web Frameworks",
        "Containers & Orchestration", "Package Management",
        "Build & Tooling", "Version Control",
        "CSS & Styling", "API & Data Layer", "State Management",
        "Infrastructure as Code", "Data Science & ML",
        "Documentation Tools",
    ],
    "research_paper": [
        "Documentation Tools", "Languages & Runtimes",
        "Data Science & ML", "Testing",
    ],
    "blog_article": [
        "Documentation Tools", "Web Frameworks", "CSS & Styling",
        "Build & Tooling",
    ],
}


class SkillReportWriterRole(AgentRole):
    """Read the aggregated skill inventory and produce two final output files:

    - ``skills.json``: Structured JSON skill report (v2: includes skill_graph)
    - ``skills_report.md``: Human-readable markdown report with proficiency
      badges, category breakdowns, artifact-type-aware section ordering,
      Skill Gap Analysis, and project statistics.

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
        """Build the report generation prompt (v2: artifact-type-aware sections + gap analysis)."""
        proj = project_name or os.path.basename(
            os.path.dirname(working_dir)) or "Project"

        # Embed inventory summary inline so the writer doesn't need to
        # discover the file from disk.
        inventory_summary = ""
        artifact_type = "unknown"
        if skill_inventory and (skill_inventory.get("tools") or skill_inventory.get("inferred_skills")):
            inv = skill_inventory
            summary = inv.get("summary", {})
            tools = inv.get("tools", inv.get("skills", []))
            skills = inv.get("inferred_skills", [])
            artifact_type = inv.get("artifact_type", "unknown")
            lines = [
                "\n## Skill Inventory (pre-computed — NO need to read from disk)",
                f"**Tools**: {summary.get('total_tools', len(tools))}",
                f"**Inferred skills**: {summary.get('total_inferred_skills', len(skills))}",
                f"**Artifact type**: {artifact_type}",
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

        # v2: Build artifact-type-specific instructions
        type_instructions = ""
        expected_areas = _ARTIFACT_EXPECTED_AREAS.get(artifact_type, {})
        if artifact_type == "software_project":
            type_instructions = (
                f"## Artifact-Type-Aware Report Structure\n"
                f"This is a **software project**. Your report should emphasize:\n"
                f"- **Tools & Testing**: detected tools, test frameworks, CI/CD pipelines\n"
                f"- **Code Craft**: design patterns, error handling, type system usage\n"
                f"- **Methodology**: dependency management, environment setup, workflows\n"
                f"Order sections so that Tools → Testing → CI/CD → Code Quality appear first.\n"
                f"Expected skill areas for this type: {', '.join(expected_areas.keys())}.\n\n"
            )
        elif artifact_type == "research_paper":
            type_instructions = (
                f"## Artifact-Type-Aware Report Structure\n"
                f"This is a **research paper**. Your report should emphasize:\n"
                f"- **Domain Expertise**: specialized subject matter knowledge\n"
                f"- **Methodology**: research methods, experimental design\n"
                f"- **Academic Rigor**: citations, references, thoroughness\n"
                f"Order sections so that Domain Expertise → Methodology → Academic Rigor appear first.\n"
                f"Expected skill areas for this type: {', '.join(expected_areas.keys())}.\n\n"
            )
        elif artifact_type == "blog_article":
            type_instructions = (
                f"## Artifact-Type-Aware Report Structure\n"
                f"This is a **blog article**. Your report should emphasize:\n"
                f"- **Writing Craft**: clarity, argumentation, narrative structure\n"
                f"- **Domain Knowledge**: subject matter expertise demonstrated\n"
                f"- **Communication**: technical writing quality, accessibility\n"
                f"Order sections so that Writing Craft → Clarity → Argumentation appear first.\n"
                f"Expected skill areas for this type: {', '.join(expected_areas.keys())}.\n\n"
            )
        else:
            type_instructions = (
                f"## Artifact-Type-Aware Report Structure\n"
                f"Artifact type is **{artifact_type}**.\n"
                f"Present skills in a logical order relevant to this artifact type.\n"
                f"Expected skill areas for this type: {', '.join(expected_areas.keys()) if expected_areas else 'general assessment'}.\n\n"
            )

        common = (
            f"You are a technical report writer. Your job is to read a skill "
            f"inventory and produce two polished output files.\n\n"
            f"## Project\n{proj}\n"
            f"{inventory_summary}"
            f"## Input\n"
            f"The skill inventory (tools + inferred skills) is summarized above. "
            f"Read `skill_inventory.json` from the working directory "
            f"({working_dir}) for full details.\n\n"
            f"{type_instructions}"
            f"## Output 1: `skills.json`\n"
            f"A structured JSON report with this schema:\n"
            f"```json\n"
            f"{{\n"
            f'  "project": "{proj}",\n'
            f'  "artifact_type": "{artifact_type}",\n'
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
            f'  ],\n'
            f'  "skill_graph": {{ /* from inventory if present */ }}\n'
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
            f"- **Artifact-type-aware section ordering**: emphasize the most "
            f"relevant categories for `{artifact_type}` artifacts\n"
            f"- **Skill Gap Analysis** section: compare detected skills against "
            f"expected skills for `{artifact_type}` artifacts. Identify which "
            f"expected skill areas are missing and provide growth suggestions.\n"
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
        """Build skills.json from inventory data (v2: tools + inferred skills + skill_graph)."""
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

        result: dict[str, Any] = {
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

        # v2: Pass skill_graph through if present in inventory
        if inventory.get("skill_graph"):
            result["skill_graph"] = inventory["skill_graph"]

        return result

    @staticmethod
    def _write_skills_json(working_dir: str, data: dict[str, Any]) -> str:
        """Write skills.json to disk."""
        path = os.path.join(working_dir, "skills.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    # ═══════════════════════════════════════════════════════════════════════
    # v2: Helper methods for artifact-type-aware report generation
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _get_category_display_order(artifact_type: str,
                                     categories: list[str]) -> list[str]:
        """Return category names sorted by artifact-type-specific priority.

        Categories in the priority list come first (in order), followed by
        remaining categories in alphabetical order.

        Args:
            artifact_type: The artifact type (e.g., "software_project").
            categories: List of category names to sort.

        Returns:
            Sorted list of category names.
        """
        priority = _ARTIFACT_CATEGORY_ORDER.get(artifact_type, [])
        ordered: list[str] = []
        remaining: list[str] = []

        # Place priority categories first, in priority order
        for cat in priority:
            if cat in categories:
                ordered.append(cat)

        # Remaining categories sorted alphabetically
        for cat in sorted(categories):
            if cat not in ordered:
                remaining.append(cat)

        return ordered + remaining

    @staticmethod
    def _get_tool_display_order(artifact_type: str,
                                 tools_by_category: dict[str, list[dict[str, Any]]]) -> list[str]:
        """Return tool category names sorted by artifact-type-specific priority.

        Args:
            artifact_type: The artifact type (e.g., "software_project").
            tools_by_category: Dict mapping category name to list of tool dicts.

        Returns:
            Sorted list of category names.
        """
        priority = _ARTIFACT_TOOL_CATEGORY_ORDER.get(artifact_type, [])
        categories = list(tools_by_category.keys())
        ordered: list[str] = []

        for cat in priority:
            if cat in categories:
                ordered.append(cat)

        for cat in sorted(categories):
            if cat not in ordered:
                ordered.append(cat)

        return ordered

    @staticmethod
    def _generate_skill_gap_analysis(
        artifact_type: str,
        tools: list[dict[str, Any]],
        skills: list[dict[str, Any]],
    ) -> list[str]:
        """Generate the Skill Gap Analysis markdown section.

        Compares detected skills/tools against expected skill areas for the
        artifact type. Reports which expected areas are covered and which are
        notably absent.

        Args:
            artifact_type: The detected artifact type.
            tools: List of detected tool dicts.
            skills: List of inferred skill dicts.

        Returns:
            List of markdown lines for the gap analysis section.
        """
        expected_areas = _ARTIFACT_EXPECTED_AREAS.get(artifact_type, {})
        if not expected_areas:
            return []

        # Collect all detected names for matching
        all_names: list[str] = []
        for t in tools:
            all_names.append(t.get("name", ""))
        for s in skills:
            all_names.append(s.get("name", ""))

        all_names_lower = [n.lower() for n in all_names]

        lines: list[str] = []
        lines.append("## 🔍 Skill Gap Analysis")
        lines.append("")
        lines.append(
            f"The following skill areas are expected for **{artifact_type.replace('_', ' ')}** "
            f"artifacts. Gaps indicate areas where the creator could grow."
        )
        lines.append("")
        lines.append("| Expected Area | Status | Notes |")
        lines.append("|----------------|--------|-------|")

        detected_count = 0
        missing_count = 0
        gap_suggestions: list[str] = []

        for area, indicators in expected_areas.items():
            # Check if any indicator is found in detected names
            found = any(
                any(ind.lower() in name.lower() or name.lower() in ind.lower()
                    for ind in indicators)
                for name in all_names
            )

            if found:
                lines.append(f"| {area} | ✅ Detected | — |")
                detected_count += 1
            else:
                lines.append(f"| {area} | ❌ Missing | "
                           f"Consider developing skills in this area |")
                missing_count += 1

                # Generate specific suggestion
                if artifact_type == "software_project":
                    suggestions: dict[str, str] = {
                        "Testing": "Add unit/integration tests using pytest or Jest",
                        "CI/CD": "Set up GitHub Actions or GitLab CI for automated builds",
                        "Version Control": "Adopt conventional commits and branching strategy",
                        "Environment Management": "Containerize with Docker for reproducible environments",
                        "Code Quality": "Add linting (Ruff/ESLint) and formatting (Prettier) tools",
                    }
                    if area in suggestions:
                        gap_suggestions.append(f"- **{area}**: {suggestions[area]}")
                elif artifact_type == "research_paper":
                    r_suggestions: dict[str, str] = {
                        "Academic Rigor": "Add methodology section, citations, and limitations",
                        "Citations & References": "Include bibliography and cite prior work",
                        "Methodology": "Describe research methods and experimental design",
                        "Domain Expertise": "Demonstrate deeper subject matter knowledge",
                    }
                    if area in r_suggestions:
                        gap_suggestions.append(f"- **{area}**: {r_suggestions[area]}")
                elif artifact_type == "blog_article":
                    b_suggestions: dict[str, str] = {
                        "Clarity": "Use concrete examples and plain language",
                        "Argumentation": "Structure arguments with evidence and counterpoints",
                        "Technical Writing": "Add code examples, diagrams, and structured sections",
                        "Domain Knowledge": "Include more specialized technical content",
                    }
                    if area in b_suggestions:
                        gap_suggestions.append(f"- **{area}**: {b_suggestions[area]}")

        lines.append("")
        lines.append(f"**Summary**: {detected_count} of "
                    f"{detected_count + missing_count} expected areas covered.")

        if missing_count == 0:
            lines.append("")
            lines.append("✅ All expected skill areas are covered — the artifact "
                        "demonstrates well-rounded skills for its type.")
        elif gap_suggestions:
            lines.append("")
            lines.append("### 💡 Growth Suggestions")
            lines.append("")
            lines.extend(gap_suggestions)

        lines.append("")
        return lines

    @staticmethod
    def _fallback_report_md(project_name: str, inventory: dict[str, Any],
                            working_dir: str) -> str:
        """Generate a markdown report from v2 inventory (tools + inferred skills).

        v2 improvements:
        - Artifact-type-aware category ordering
        - Skill Gap Analysis section
        - Emphasis on type-relevant sections
        """
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

        # ═══════════════════════════════════════════════════════════════════
        # v2: Detected Tools — with artifact-type-aware category grouping
        # ═══════════════════════════════════════════════════════════════════
        if tools:
            lines.append("## 🛠️ Detected Tools")
            lines.append("")

            # v2: Group tools by category for artifact-type-aware display
            tools_by_cat: dict[str, list[dict[str, Any]]] = {}
            for t in tools:
                cat = t.get("category", "Other")
                tools_by_cat.setdefault(cat, []).append(t)

            # v2: Order categories by artifact-type priority
            ordered_cats = SkillReportWriterRole._get_tool_display_order(
                artifact_type, tools_by_cat)

            for cat in ordered_cats:
                cat_tools = tools_by_cat[cat]
                emoji = _CATEGORY_EMOJI.get(cat, "📌")
                if len(ordered_cats) > 1:
                    lines.append(f"### {emoji} {cat} ({len(cat_tools)})")
                    lines.append("")
                lines.append("| Tool | Proficiency | Scope |")
                lines.append("|------|-------------|-------|")
                for t in cat_tools:
                    name = t.get("name", "?")
                    prof = t.get("proficiency", "beginner")
                    scope = t.get("scope", "")
                    badge = _BADGES.get(prof, prof)
                    lines.append(f"| {name} | {badge} | {scope} |")
                lines.append("")

        # ═══════════════════════════════════════════════════════════════════
        # v2: Inferred Skills — artifact-type-aware category ordering
        # ═══════════════════════════════════════════════════════════════════
        if skills:
            lines.append("## 🧠 Inferred Developer Skills")
            lines.append("")

            # Group by category
            by_category: dict[str, list[dict[str, Any]]] = {}
            for s in skills:
                cat = s.get("category", "Unknown")
                by_category.setdefault(cat, []).append(s)

            # v2: Order categories by artifact-type priority
            ordered_categories = SkillReportWriterRole._get_category_display_order(
                artifact_type, list(by_category.keys()))

            # v2: Artifact-type emphasis note
            if artifact_type == "software_project":
                lines.append(
                    "> 💡 **Software Project Focus**: Tools, testing, CI/CD, "
                    "and code quality are prioritized below. These are the "
                    "most relevant skill dimensions for software artifacts."
                )
                lines.append("")
            elif artifact_type == "research_paper":
                lines.append(
                    "> 💡 **Research Paper Focus**: Domain expertise, methodology, "
                    "and academic rigor are prioritized below. These are the "
                    "most relevant skill dimensions for research artifacts."
                )
                lines.append("")
            elif artifact_type == "blog_article":
                lines.append(
                    "> 💡 **Blog Article Focus**: Writing craft, clarity, and "
                    "argumentation are prioritized below. These are the most "
                    "relevant skill dimensions for written content."
                )
                lines.append("")

            for cat in ordered_categories:
                cat_skills = by_category[cat]
                emoji = _CATEGORY_EMOJI.get(cat, "📌")
                lines.append(f"### {emoji} {cat} ({len(cat_skills)})")
                lines.append("")
                lines.append("| Skill | Proficiency | Confidence |")
                lines.append("|-------|-------------|------------|")
                for sk in cat_skills:
                    name = sk.get("name", "?")
                    prof = sk.get("proficiency", "beginner")
                    conf = sk.get("confidence", "?")
                    badge = _BADGES.get(prof, prof)
                    # v2: Add cross-referenced marker
                    if sk.get("cross_referenced"):
                        name = f"{name} 🔗"
                    lines.append(f"| {name} | {badge} | {conf} |")
                lines.append("")

        # ═══════════════════════════════════════════════════════════════════
        # v2: Skill Gap Analysis — before File Statistics
        # ═══════════════════════════════════════════════════════════════════
        gap_lines = SkillReportWriterRole._generate_skill_gap_analysis(
            artifact_type, tools, skills)
        if gap_lines:
            lines.extend(gap_lines)

        # ═══════════════════════════════════════════════════════════════════
        # File Statistics
        # ═══════════════════════════════════════════════════════════════════
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

        # v2: Add gap-related recommendation
        if artifact_type in _ARTIFACT_EXPECTED_AREAS:
            expected_areas = _ARTIFACT_EXPECTED_AREAS[artifact_type]
            all_names = [t.get("name", "") for t in tools] + [s.get("name", "") for s in skills]
            all_names_lower = [n.lower() for n in all_names]
            missing = []
            for area, indicators in expected_areas.items():
                found = any(
                    any(ind.lower() in name.lower() or name.lower() in ind.lower()
                        for ind in indicators)
                    for name in all_names
                )
                if not found:
                    missing.append(area)
            if missing:
                lines.append(f"- **Skill gaps detected** in: {', '.join(missing)}. "
                           f"See Skill Gap Analysis section above.")

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
