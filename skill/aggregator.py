"""SkillAggregatorRole — aggregates domain reports and deduplicates skills."""

import json
import os
import sys
from typing import Any

# Ensure repo root is on sys.path so we can import agent.py and tools.py
# __file__ is .../coderpp_output/modules/skill_agent_roles/skill/aggregator.py
# Repo root is 5 dirs up: .../universal_multi_agent_framework/
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agent import AgentResult, AgentRole  # noqa: E402
from tools import ToolRegistry  # noqa: E402


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

# Proficiency scoring for conflict resolution
_PROFICIENCY_SCORES: dict[str, int] = {
    "expert": 4,
    "advanced": 3,
    "intermediate": 2,
    "beginner": 1,
}


class SkillAggregatorRole(AgentRole):
    """Read 4 domain reports, deduplicate entries, resolve proficiency levels,
    categorize skills, and write ``skill_inventory.json``.

    The aggregator is the central orchestrator that combines the outputs of
    the four domain detectors into a single coherent skill inventory.
    """

    agent_name: str = "skill_aggregator"
    max_steps: int = 10

    # ── Tools ───────────────────────────────────────────────────────────

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        """Only need Read and Write — no external commands needed."""
        return ToolRegistry.to_dicts(ToolRegistry.skill_aggregator_tools())

    # ── Task prompt ─────────────────────────────────────────────────────

    def build_task(self, backend: str, working_dir: str = ".",
                   **context: Any) -> str:
        """Build the aggregation prompt."""
        common = (
            f"You are a skill inventory aggregator. Your job is to read four "
            f"domain-specific reports and combine them into a unified skill "
            f"inventory.\n\n"
            f"## Input Files (in working directory: {working_dir})\n"
            f"1. `python_report.json` — Python ecosystem skills\n"
            f"2. `javascript_report.json` — JavaScript ecosystem skills\n"
            f"3. `infrastructure_report.json` — Infrastructure & DevOps skills\n"
            f"4. `configdocs_report.json` — Configuration, docs, API specs\n\n"
            f"## Task\n"
            f"1. Read all four report files.\n"
            f"2. **Deduplicate**: If the same skill appears in multiple reports "
            f"(e.g., \"Docker\" may appear in both infra and config), keep the "
            f"one with higher proficiency and add a `sources` field listing "
            f"the originating domains.\n"
            f"3. **Resolve proficiency**: When the same skill has conflicting "
            f"proficiency levels across domains, use the HIGHEST level found "
            f"and note the conflict.\n"
            f"4. **Categorize**: Map each skill to one of these top-level "
            f"categories:\n"
            f"   - Languages & Runtimes\n"
            f"   - Web Frameworks\n"
            f"   - Frontend\n"
            f"   - Backend\n"
            f"   - Testing\n"
            f"   - Code Quality\n"
            f"   - Data Science & ML\n"
            f"   - Build & Tooling\n"
            f"   - Containerization\n"
            f"   - Orchestration\n"
            f"   - CI/CD\n"
            f"   - Cloud\n"
            f"   - Infrastructure as Code\n"
            f"   - Monitoring & Observability\n"
            f"   - Web Servers\n"
            f"   - Databases\n"
            f"   - Documentation\n"
            f"   - Configuration Management\n"
            f"   - API Specifications\n"
            f"   - Other\n"
            f"5. **Count files**: Compute summary statistics from the project "
            f"scan (total files, language breakdown, test files).\n\n"
            f"6. Write the unified inventory to `skill_inventory.json`:\n"
            f"```json\n"
            f"{{\n"
            f'  "project_dir": ".",\n'
            f'  "generated_at": "<ISO timestamp>",\n'
            f'  "summary": {{\n'
            f'    "total_skills": <int>,\n'
            f'    "domains_covered": ["Python", "JavaScript", ...],\n'
            f'    "skill_categories": {{\n'
            f'      "Web Frameworks": 3,\n'
            f'      "Testing": 2,\n'
            f'      ...\n'
            f'    }},\n'
            f'    "proficiency_distribution": {{\n'
            f'      "expert": 0,\n'
            f'      "advanced": 5,\n'
            f'      "intermediate": 3,\n'
            f'      "beginner": 2\n'
            f'    }}\n'
            f'  }},\n'
            f'  "skills": [\n'
            f'    {{\n'
            f'      "name": "Python",\n'
            f'      "category": "Languages & Runtimes",\n'
            f'      "proficiency": "advanced",\n'
            f'      "sources": ["Python"],\n'
            f'      "evidence": [".python-version", "setup.py"],\n'
            f'      "version_hint": "3.11"\n'
            f'    }}\n'
            f'  ],\n'
            f'  "file_stats": {{\n'
            f'    "total_files": 150,\n'
            f'    "python_files": 45,\n'
            f'    "javascript_files": 20,\n'
            f'    "typescript_files": 10,\n'
            f'    "test_files": 25,\n'
            f'    "config_files": 30,\n'
            f'    "doc_files": 15\n'
            f'  }}\n'
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
            json_str = self._extract_json_object(content)
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
        """Rule-based deduplication and categorization from domain reports.

        Reads the four domain report JSON files from disk and performs
        deterministic merging without LLM.
        """
        from datetime import datetime, timezone

        # Load domain reports
        reports: dict[str, dict[str, Any]] = {}
        for fname, domain_key in [
            ("python_report.json", "Python"),
            ("javascript_report.json", "JavaScript"),
            ("infrastructure_report.json", "Infrastructure"),
            ("configdocs_report.json", "Configuration & Documentation"),
        ]:
            path = os.path.join(working_dir, fname)
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        reports[domain_key] = json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass

        # Merge skills with deduplication
        seen: dict[str, dict[str, Any]] = {}
        for domain, report in reports.items():
            for skill in report.get("skills", []):
                name = skill.get("name", "")
                if not name:
                    continue

                if name in seen:
                    # Resolve proficiency: keep the highest
                    existing = seen[name]
                    new_prof = _PROFICIENCY_SCORES.get(
                        skill.get("proficiency", "beginner"), 1)
                    old_prof = _PROFICIENCY_SCORES.get(
                        existing.get("proficiency", "beginner"), 1)
                    if new_prof > old_prof:
                        existing["proficiency"] = skill.get("proficiency")
                    # Merge sources
                    if domain not in existing.get("sources", []):
                        existing["sources"].append(domain)
                    # Merge evidence
                    existing_evidence = set(existing.get("evidence", []))
                    existing_evidence.update(skill.get("evidence", []))
                    existing["evidence"] = sorted(existing_evidence)
                else:
                    skill_copy = dict(skill)
                    skill_copy["sources"] = [domain]
                    # Normalize category
                    raw_cat = skill_copy.get("category", "other")
                    skill_copy["category"] = _CATEGORY_MAP.get(raw_cat, raw_cat)
                    seen[name] = skill_copy

        # Add language/runtime entries
        if "Python" in reports:
            py_ver = reports["Python"].get("version", {}).get("detected", "unknown")
            if py_ver and py_ver != "unknown":
                seen["Python"] = {
                    "name": "Python",
                    "category": "Languages & Runtimes",
                    "proficiency": "advanced" if reports["Python"].get("total_python_files", 0) > 20
                        else "intermediate",
                    "sources": ["Python"],
                    "evidence": [f"Version {py_ver}"],
                    "version_hint": py_ver,
                }

        if "JavaScript" in reports:
            js_report = reports["JavaScript"]
            js_ver = js_report.get("runtime", {}).get("detected", "unknown")
            if js_ver and js_ver != "unknown":
                seen["Node.js"] = {
                    "name": "Node.js",
                    "category": "Languages & Runtimes",
                    "proficiency": "advanced" if js_report.get("total_js_files", 0) > 20
                        else "intermediate",
                    "sources": ["JavaScript"],
                    "evidence": [f"Version {js_ver}"],
                    "version_hint": js_ver,
                }
            if js_report.get("typescript", {}).get("used"):
                seen["TypeScript"] = {
                    "name": "TypeScript",
                    "category": "Languages & Runtimes",
                    "proficiency": "advanced" if js_report.get("total_ts_files", 0) > 20
                        else "intermediate",
                    "sources": ["JavaScript"],
                    "evidence": js_report.get("typescript", {}).get("config", "tsconfig.json"),
                    "version_hint": "",
                }

        # Add config/docs entries
        if "Configuration & Documentation" in reports:
            cd_report = reports["Configuration & Documentation"]
            for fmt in cd_report.get("config_formats", []):
                name = f"{fmt['format']} Configuration"
                if name not in seen:
                    seen[name] = {
                        "name": name,
                        "category": "Configuration Management",
                        "proficiency": "advanced" if fmt.get("file_count", 0) > 10
                            else "intermediate",
                        "sources": ["Configuration & Documentation"],
                        "evidence": fmt.get("examples", []),
                        "version_hint": "",
                    }
            for doc in cd_report.get("documentation", []):
                name = doc.get("type", "Documentation")
                if name not in seen:
                    seen[f"Documentation: {name}"] = {
                        "name": f"Documentation: {name}",
                        "category": "Documentation",
                        "proficiency": "advanced" if doc.get("completeness") == "comprehensive"
                            else "intermediate",
                        "sources": ["Configuration & Documentation"],
                        "evidence": [doc.get("path", "")],
                        "version_hint": "",
                    }
            for spec in cd_report.get("api_specs", []):
                name = spec.get("type", "")
                if name and name not in seen:
                    seen[name] = {
                        "name": name,
                        "category": "API Specifications",
                        "proficiency": "intermediate",
                        "sources": ["Configuration & Documentation"],
                        "evidence": [spec.get("path", "")],
                        "version_hint": spec.get("version_hint", ""),
                    }
            for tool in cd_report.get("tooling", []):
                name = tool.get("tool", "")
                if name and name not in seen:
                    seen[name] = {
                        "name": name,
                        "category": "Build & Tooling",
                        "proficiency": "intermediate",
                        "sources": ["Configuration & Documentation"],
                        "evidence": [tool.get("config", "")],
                        "version_hint": "",
                    }

        skills_list = sorted(seen.values(), key=lambda s: (
            _PROFICIENCY_SCORES.get(s.get("proficiency", "beginner"), 1),
            s.get("name", ""),
        ), reverse=True)

        # Build summary
        categories_count: dict[str, int] = {}
        prof_dist: dict[str, int] = {"expert": 0, "advanced": 0, "intermediate": 0, "beginner": 0}
        for s in skills_list:
            cat = s.get("category", "Other")
            categories_count[cat] = categories_count.get(cat, 0) + 1
            prof = s.get("proficiency", "beginner")
            if prof in prof_dist:
                prof_dist[prof] += 1

        domains_covered = list(reports.keys())

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
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_skills": len(skills_list),
                "domains_covered": domains_covered,
                "skill_categories": categories_count,
                "proficiency_distribution": prof_dist,
            },
            "skills": skills_list,
            "file_stats": file_stats,
            "_fallback": True,
        }

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
