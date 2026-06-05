"""SkillAggregatorRole — aggregates domain reports and deduplicates skills."""

import json
import os
from typing import Any

from agent import AgentResult, AgentRole
from tools import ToolRegistry
from utils import extract_json_object, _PROFICIENCY_SCORES


# Category mapping for normalization
_CATEGORY_MAP: dict[str, str] = {
    # Python categories
    "web_framework": "Web Frameworks",
    "testing": "Testing",
    "linting": "Code Quality",
    "data_science": "Data Science & ML",
    "tooling": "Tooling",
    # JS categories
    "frontend_framework": "Frontend",
    "backend_framework": "Backend",
    "build_tooling": "Build & Tooling",
    # Infra categories
    "containerization": "Containerization",
    "orchestration": "Orchestration",
    "ci_cd": "CI/CD",
    "cloud": "Cloud",
    "infrastructure_as_code": "Infrastructure as Code",
    "monitoring": "Monitoring & Observability",
    "service_mesh": "Service Mesh",
    "web_server": "Web Servers",
    "database": "Databases",
    "other": "Other",
}

class SkillAggregatorRole(AgentRole):
    """Read 4 skill-dimension reports, deduplicate entries, resolve proficiency,
    categorize skills, and write ``skill_inventory.json``.

    v2: handles both ``detected_tools`` AND ``inferred_skills`` from the
    4 new detectors (Domain Expertise, Technical Craft, Methodology, Rigor).
    """

    agent_name: str = "skill_aggregator"
    max_steps: int = 10

    # ── Tools ───────────────────────────────────────────────────────────

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        """Only need Read and Write — no external commands needed."""
        return ToolRegistry.to_dicts(ToolRegistry.skill_aggregator_tools())

    # ── Task prompt ─────────────────────────────────────────────────────

    def build_task(self, backend: str, working_dir: str = ".",
                   detector_outputs: list[dict[str, Any]] | None = None,
                   **context: Any) -> str:
        """Build the aggregation prompt."""
        # Build inline summary of detector results so the aggregator knows
        # what was found without having to discover files from disk.
        detector_summary = ""
        if detector_outputs:
            lines = [
                "\n## Detector Results (pre-computed — NO need to read from disk)",
                "The following domain reports have already been generated. "
                "Their contents are summarized below. Read the JSON files "
                "ONLY if you need additional detail beyond what is shown here.",
                "",
            ]
            for d in detector_outputs:
                domain = d.get("domain", "Unknown")
                output_file = d.get("output_file", "")
                summary = d.get("summary", "")
                data = d.get("data", {})
                tools = data.get("detected_tools", data.get("skills", []))
                inferred = data.get("inferred_skills", [])
                skill_preview = ""
                if inferred:
                    names = [s.get("name", "?") for s in inferred[:8]]
                    skill_preview = f"\n   Inferred: {', '.join(names)}"
                if tools:
                    tnames = [t.get("name", "?") for t in tools[:5]]
                    skill_preview += f"\n   Tools: {', '.join(tnames)}"
                mark = "✓" if output_file else "✗"
                lines.append(f"- **{mark} {domain}** — {summary}{skill_preview}")
            lines.append("")
            detector_summary = "\n".join(lines)

        common = (
            f"You are a skill inventory aggregator. Your job is to read four "
            f"domain-specific reports and combine them into a unified skill "
            f"inventory.\n\n"
            f"## Input Files (in working directory: {working_dir})\n"
            f"1. `domain_expertise_report.json` — Specialized domain knowledge\n"
            f"2. `technical_craft_report.json` — Craft skills in the medium\n"
            f"3. `methodology_report.json` — Tools, workflows, processes\n"
            f"4. `rigor_report.json` — Thoroughness, testing, documentation\n"
            f"{detector_summary}"
            f"## Task\n"
            f"1. Read all four report files for full details.\n"
            f"2. Each report has BOTH `detected_tools` AND `inferred_skills`. "
            f"Merge them separately.\n"
            f"3. **Deduplicate tools**: same tool in multiple reports → keep "
            f"highest proficiency, merge sources.\n"
            f"4. **Deduplicate skills**: same skill name in multiple reports → "
            f"keep highest proficiency and confidence, merge evidence.\n"
            f"5. **Categorize**: Map each entry to a top-level category. "
            f"Tools → Languages & Runtimes, Testing, Web Frameworks, etc. "
            f"Skills → Domain Knowledge, Technical Craft, Engineering Practice, "
            f"Quality Assurance, Architecture.\n"
            f"6. **Count files**: Compute summary stats from the project scan.\n\n"
            f"7. Write `skill_inventory.json`:\n"
            f"```json\n"
            f"{{\n"
            f'  "project_dir": ".",\n'
            f'  "artifact_type": "software_project|research_paper|...",\n'
            f'  "generated_at": "<ISO timestamp>",\n'
            f'  "summary": {{\n'
            f'    "total_tools": <int>,\n'
            f'    "total_inferred_skills": <int>,\n'
            f'    "dimensions_covered": ["Domain Expertise", "Technical Craft", ...],\n'
            f'    "proficiency_distribution": {{\n'
            f'      "expert": 0, "advanced": 5, "intermediate": 3, "beginner": 2\n'
            f'    }}\n'
            f'  }},\n'
            f'  "tools": [\n'
            f'    {{\n'
            f'      "name": "Python", "category": "Languages & Runtimes",\n'
            f'      "proficiency": "advanced", "sources": ["Methodology"],\n'
            f'      "evidence": [".python-version", "pyproject.toml"]\n'
            f'    }}\n'
            f'  ],\n'
            f'  "inferred_skills": [\n'
            f'    {{\n'
            f'      "name": "Design Patterns", "category": "Technical Craft",\n'
            f'      "proficiency": "advanced", "confidence": "high",\n'
            f'      "sources": ["Technical Craft"],\n'
            f'      "evidence": {{"description": "..."}}\n'
            f'    }}\n'
            f'  ]\n'
            f"}}\n"
            f"```\n\n"
            f"IMPORTANT: Write the JSON file to `skill_inventory.json`, then "
            f"output TASK_COMPLETE."
        )

        if backend == "claude_cli":
            backend_note = (
                "\n\nUse your own knowledge — do NOT search the web. "
                "Read the four domain reports, aggregate them, write "
                "skill_inventory.json. Output TASK_COMPLETE."
            )
        else:
            backend_note = (
                "\n\nRead the four domain reports, aggregate them, write "
                "skill_inventory.json, output TASK_COMPLETE."
            )

        return common + backend_note

    # ── Parse result ────────────────────────────────────────────────────

    def parse_result(self, result: AgentResult, working_dir: str,
                     project_dir: str = ".", **context: Any) -> dict[str, Any]:
        """Parse aggregated inventory from agent or fall back to rule-based."""
        inventory: dict[str, Any] = {}

        # 1. Agent response
        for msg in reversed(result.messages):
            content = msg.content if hasattr(msg, "content") else str(msg)
            json_str = extract_json_object(content)
            if json_str:
                try:
                    parsed = json.loads(json_str)
                    if "skills" in parsed or "summary" in parsed:
                        inventory = parsed
                        break
                except json.JSONDecodeError:
                    continue

        # 2. Disk file
        if not inventory:
            path = os.path.join(working_dir, "skill_inventory.json")
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        parsed = json.load(f)
                    if isinstance(parsed, dict) and "skills" in parsed:
                        inventory = parsed
                except (json.JSONDecodeError, OSError):
                    pass

        # 3. Rule-based fallback
        if not inventory:
            inventory = self._fallback_aggregator(project_dir, working_dir)

        return inventory

    # ── Fallback aggregator ─────────────────────────────────────────────

    @staticmethod
    def _fallback_aggregator(project_dir: str = ".",
                             working_dir: str = ".") -> dict[str, Any]:
        """Rule-based deduplication and categorization from skill-dimension reports.

        Reads the four new detector report files from disk and performs
        deterministic merging without LLM. Handles both detected_tools
        and inferred_skills from each report.
        """
        from datetime import datetime, timezone

        # Load domain reports
        reports: dict[str, dict[str, Any]] = {}
        for fname, domain_key in [
            ("domain_expertise_report.json", "Domain Expertise"),
            ("technical_craft_report.json", "Technical Craft"),
            ("methodology_report.json", "Methodology & Tooling"),
            ("rigor_report.json", "Depth & Rigor"),
        ]:
            path = os.path.join(working_dir, fname)
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        reports[domain_key] = json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass

        # ── Merge tools ───────────────────────────────────────────────
        tools_seen: dict[str, dict[str, Any]] = {}
        for domain, report in reports.items():
            for tool in report.get("detected_tools", []):
                name = tool.get("name", "")
                if not name:
                    continue
                if name in tools_seen:
                    existing = tools_seen[name]
                    new_prof = _PROFICIENCY_SCORES.get(
                        tool.get("proficiency", "beginner"), 1)
                    old_prof = _PROFICIENCY_SCORES.get(
                        existing.get("proficiency", "beginner"), 1)
                    if new_prof > old_prof:
                        existing["proficiency"] = tool.get("proficiency")
                    if domain not in existing.get("sources", []):
                        existing["sources"].append(domain)
                    ex_ev = existing.get("evidence", [])
                    if isinstance(ex_ev, list):
                        ex_ev.extend(tool.get("evidence", []))
                        existing["evidence"] = list(set(ex_ev))[:10]
                else:
                    tool_copy = dict(tool)
                    tool_copy["sources"] = [domain]
                    raw_cat = tool_copy.get("category", "Other Tools")
                    tool_copy["category"] = _CATEGORY_MAP.get(raw_cat, raw_cat)
                    tools_seen[name] = tool_copy

        tools_list = sorted(tools_seen.values(), key=lambda s: (
            _PROFICIENCY_SCORES.get(s.get("proficiency", "beginner"), 1),
            s.get("name", ""),
        ), reverse=True)

        # ── Merge inferred skills ─────────────────────────────────────
        skills_seen: dict[str, dict[str, Any]] = {}
        _SKILL_CATEGORY_MAP: dict[str, str] = {
            "algorithm_design": "Technical Craft",
            "design_patterns": "Technical Craft",
            "error_handling": "Technical Craft",
            "type_system": "Technical Craft",
            "code_organization": "Technical Craft",
            "performance": "Technical Craft",
            "security": "Technical Craft",
            "argumentation": "Technical Craft",
            "technical_writing": "Technical Craft",
            "narrative_structure": "Technical Craft",
            "clarity": "Technical Craft",
            "git_workflow": "Engineering Practice",
            "ci_cd": "Engineering Practice",
            "dependency_management": "Engineering Practice",
            "environment_management": "Engineering Practice",
            "release_management": "Engineering Practice",
            "incremental_development": "Engineering Practice",
            "testing_strategy": "Quality Assurance",
            "test_coverage": "Quality Assurance",
            "documentation_quality": "Quality Assurance",
            "code_quality_enforcement": "Quality Assurance",
            "academic_rigor": "Quality Assurance",
            "monitoring": "Quality Assurance",
            "component_design": "Architecture",
            "api_design": "Architecture",
            "data_modeling": "Architecture",
            "scalability": "Architecture",
            "infrastructure_design": "Architecture",
        }

        for domain, report in reports.items():
            for skill in report.get("inferred_skills", []):
                name = skill.get("name", "")
                if not name:
                    continue
                if name in skills_seen:
                    existing = skills_seen[name]
                    new_prof = _PROFICIENCY_SCORES.get(
                        skill.get("proficiency", "beginner"), 1)
                    old_prof = _PROFICIENCY_SCORES.get(
                        existing.get("proficiency", "beginner"), 1)
                    if new_prof > old_prof:
                        existing["proficiency"] = skill.get("proficiency")
                        existing["confidence"] = skill.get("confidence", "medium")
                    if domain not in existing.get("sources", []):
                        existing["sources"].append(domain)
                else:
                    skill_copy = dict(skill)
                    skill_copy["sources"] = [domain]
                    skill_key = name.lower().replace(" ", "_")
                    cat = _SKILL_CATEGORY_MAP.get(skill_key, domain)
                    skill_copy["category"] = cat
                    skills_seen[name] = skill_copy

        skills_list = sorted(skills_seen.values(), key=lambda s: (
            _PROFICIENCY_SCORES.get(s.get("proficiency", "beginner"), 1),
            s.get("name", ""),
        ), reverse=True)

        # ── Build summary ─────────────────────────────────────────────
        prof_dist: dict[str, int] = {"expert": 0, "advanced": 0, "intermediate": 0, "beginner": 0}
        for entry in tools_list + skills_list:
            prof = entry.get("proficiency", "beginner")
            if prof in prof_dist:
                prof_dist[prof] += 1

        dimensions = list(reports.keys())

        # Artifact type from artifact_analysis.json
        artifact_type = "unknown"
        aa_path = os.path.join(working_dir, "artifact_analysis.json")
        if os.path.exists(aa_path):
            try:
                with open(aa_path) as f:
                    aa = json.load(f)
                artifact_type = aa.get("artifact_type", {}).get("type", "unknown")
            except (json.JSONDecodeError, OSError):
                pass

        # File stats from project_scan
        file_stats: dict[str, int] = {}
        scan_path = os.path.join(working_dir, "project_scan.json")
        if os.path.exists(scan_path):
            try:
                with open(scan_path) as f:
                    scan = json.load(f)
                cats = scan.get("file_categories", {})
                file_stats["total_files"] = scan.get("total_files", 0)
                file_stats["python_files"] = sum(
                    1 for f in cats.get("source", []) if f.endswith(".py"))
                file_stats["javascript_files"] = sum(
                    1 for f in cats.get("source", [])
                    if f.endswith((".js", ".jsx", ".mjs", ".cjs")))
                file_stats["typescript_files"] = sum(
                    1 for f in cats.get("source", [])
                    if f.endswith((".ts", ".tsx", ".mts", ".cts")))
                file_stats["test_files"] = len(cats.get("test", []))
                file_stats["config_files"] = len(cats.get("config", []))
                file_stats["doc_files"] = len(cats.get("docs", []))
            except (json.JSONDecodeError, OSError):
                pass

        return {
            "project_dir": project_dir,
            "artifact_type": artifact_type,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_tools": len(tools_list),
                "total_inferred_skills": len(skills_list),
                "dimensions_covered": dimensions,
                "proficiency_distribution": prof_dist,
            },
            "tools": tools_list,
            "inferred_skills": skills_list,
            "file_stats": file_stats,
            "_fallback": True,
        }
