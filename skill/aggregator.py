"""SkillAggregatorRole — aggregates domain reports and deduplicates skills.

v2 improvements:
- Category inference replacing hardcoded _SKILL_CATEGORY_MAP
- Evidence merging (not just highest proficiency) when skills appear in multiple detectors
- cross_referenced flag and confidence boost for multi-detector skills
- skill_graph generation showing tool↔skill and skill↔skill relationships
- Updated build_task prompt to request cross_referenced and skill_graph fields
"""

from __future__ import annotations

import json
import os
from typing import Any

from agent import AgentResult, AgentRole
from tools import ToolRegistry
from utils import extract_json_object, _PROFICIENCY_SCORES


# Category mapping for tool normalization (not skills — skills use _infer_category)
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

# v2: Confidence level ordering for boost
_CONFIDENCE_LEVELS: dict[str, int] = {
    "low": 1,
    "medium": 2,
    "high": 3,
}
_CONFIDENCE_NAMES: list[str] = ["low", "medium", "high"]


# ═══════════════════════════════════════════════════════════════════════════
# v2: Category inference (replaces hardcoded _SKILL_CATEGORY_MAP)
# ═══════════════════════════════════════════════════════════════════════════

def _infer_category(skill_name: str, detector_domain: str) -> str:
    """Infer the category for a skill based on its name and the detector domain.

    v2: Replaces the hardcoded ``_SKILL_CATEGORY_MAP`` with inference logic.
    The category is determined by:
    (a) The detector domain as primary signal, then
    (b) Keyword analysis of the skill name for sub-category refinement.

    Args:
        skill_name: The skill name (e.g., "Machine Learning", "Testing Strategy").
        detector_domain: The detector domain that found the skill
            (e.g., "Domain Expertise", "Technical Craft").

    Returns:
        A category string: "Domain Knowledge", "Technical Craft",
        "Engineering Practice", "Quality Assurance", "Architecture", or
        the detector domain itself as ultimate fallback.
    """
    name_lower = skill_name.lower().replace(" ", "_")

    # ── Step 1: Detector domain provides the primary category signal ──
    if detector_domain == "Domain Expertise":
        base_category = "Domain Knowledge"
    elif detector_domain == "Technical Craft":
        base_category = "Technical Craft"
    elif detector_domain == "Methodology & Tooling":
        base_category = "Engineering Practice"
    elif detector_domain == "Depth & Rigor":
        base_category = "Quality Assurance"
    else:
        base_category = detector_domain  # fallback to the domain name itself

    # ── Step 2: Sub-category refinement based on skill name patterns ──

    # Architecture indicators — skills about system-level structural design
    architecture_keywords = [
        "component_design", "api_design", "data_modeling", "scalability",
        "infrastructure_design", "architecture", "system_design",
        "microservices", "service_oriented",
    ]
    if any(kw in name_lower for kw in architecture_keywords):
        return "Architecture"

    # Quality Assurance indicators — skills about verification, validation, rigor
    quality_keywords = [
        "testing_strategy", "test_coverage", "documentation_quality",
        "code_quality_enforcement", "academic_rigor", "monitoring",
        "thoroughness", "quality",
    ]
    if any(kw in name_lower for kw in quality_keywords):
        return "Quality Assurance"

    # Engineering Practice indicators — skills about process, workflow, methodology
    tooling_keywords = [
        "git_workflow", "ci/cd", "dependency_management",
        "environment_management", "release_management",
        "incremental_development", "workflow", "sophistication",
        "deployment",
    ]
    if any(kw in name_lower for kw in tooling_keywords):
        return "Engineering Practice"

    # Technical Craft indicators — medium-specific craft skills
    # Check BEFORE domain keywords to prevent false domain matches
    # (e.g., "Security Awareness" is craft, not domain expertise)
    craft_keywords = [
        "design_patterns", "error_handling", "type_system",
        "code_organization", "performance_awareness",
        "security_awareness",
        "argumentation", "technical_writing", "narrative_structure",
        "clarity", "writing_craft",
    ]
    if any(kw in name_lower for kw in craft_keywords):
        return "Technical Craft"

    # Domain Knowledge indicators — specific technical domain expertise
    # Note: "security" alone is too broad; it matches "Security" (the domain)
    # but "security_awareness" is caught by craft_keywords above.
    domain_keywords = [
        "machine_learning", "deep_learning", "computer_vision",
        "natural_language", "reinforcement", "distributed",
        "security", "compiler", "database", "game_development",
        "finance", "scientific", "networking", "operating_system",
        "embedded", "devops", "data_engineering", "frontend",
        "mobile", "blockchain",
    ]
    if any(kw in name_lower for kw in domain_keywords):
        return "Domain Knowledge"

    return base_category


# ═══════════════════════════════════════════════════════════════════════════
# v2: Evidence merging helper
# ═══════════════════════════════════════════════════════════════════════════

def _merge_evidence(existing_evidence: Any, new_evidence: Any) -> dict[str, Any]:
    """Merge two evidence values, combining list fields and preferring non-empty values.

    Used when a skill appears in multiple detector reports — we want to
    preserve ALL evidence from ALL detectors, not just the highest-proficiency one.

    Args:
        existing_evidence: The current evidence value (dict or other).
        new_evidence: The new evidence value to merge in.

    Returns:
        A merged evidence dict.
    """
    result: dict[str, Any] = {}

    existing = existing_evidence if isinstance(existing_evidence, dict) else {}
    new = new_evidence if isinstance(new_evidence, dict) else {}

    all_keys = set(existing.keys()) | set(new.keys())
    for key in all_keys:
        old_val = existing.get(key)
        new_val = new.get(key)

        if isinstance(old_val, list) and isinstance(new_val, list):
            # Merge lists, deduplicate while preserving order
            merged: list[Any] = []
            seen: set[str] = set()
            for item in old_val + new_val:
                item_key = str(item)
                if item_key not in seen:
                    seen.add(item_key)
                    merged.append(item)
            result[key] = merged[:15]  # cap per-field at 15 entries
        elif isinstance(old_val, list):
            result[key] = old_val
        elif isinstance(new_val, list):
            result[key] = new_val
        elif new_val is not None and new_val != "" and new_val != {}:
            result[key] = new_val
        elif old_val is not None and old_val != "" and old_val != {}:
            result[key] = old_val

    return result


def _merge_evidence_refs(
    existing_refs: Any, new_refs: Any
) -> list[dict[str, Any]]:
    """Merge evidence_refs arrays, deduplicating by file path.

    Args:
        existing_refs: Existing evidence_refs list (or other value).
        new_refs: New evidence_refs list (or other value) to merge in.

    Returns:
        Merged, deduplicated evidence_refs list.
    """
    existing = existing_refs if isinstance(existing_refs, list) else []
    new = new_refs if isinstance(new_refs, list) else []

    # Merge and deduplicate by file path
    seen_files: set[str] = set()
    merged: list[dict[str, Any]] = []
    for ref in existing + new:
        if not isinstance(ref, dict):
            continue
        fp = ref.get("file", "")
        if fp not in seen_files:
            seen_files.add(fp)
            merged.append(ref)

    return merged[:20]  # cap at 20 evidence refs


# ═══════════════════════════════════════════════════════════════════════════
# v2: Confidence boosting
# ═══════════════════════════════════════════════════════════════════════════

def _boost_confidence(confidence: str) -> str:
    """Boost a confidence level by one step.

    low → medium, medium → high, high stays high.

    Args:
        confidence: Current confidence level.

    Returns:
        Boosted confidence level.
    """
    idx = _CONFIDENCE_LEVELS.get(confidence, 1)
    boosted_idx = min(idx + 1, len(_CONFIDENCE_NAMES))
    return _CONFIDENCE_NAMES[boosted_idx - 1]


# ═══════════════════════════════════════════════════════════════════════════
# v2: Skill graph generation
# ═══════════════════════════════════════════════════════════════════════════

def _generate_skill_graph(
    tools_list: list[dict[str, Any]],
    skills_list: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate a skill graph showing relationships between tools and skills.

    The graph connects:
    (a) Tools to related skills (e.g., ``pytest`` → ``Testing Strategy``)
    (b) Skills to related skills (e.g., ``Testing Strategy`` ↔ ``Test Coverage Thoroughness``)
    (c) Skills within the same category

    Args:
        tools_list: List of detected tool dicts (each with ``name``, ``category``).
        skills_list: List of inferred skill dicts (each with ``name``, ``category``).

    Returns:
        A dict with ``nodes`` and ``edges``. Each node has ``id``, ``type``, ``category``.
        Each edge has ``source``, ``target``, ``relationship``.
    """
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_node_ids: set[str] = set()

    # ── Known tool → skill relationships ──
    _TOOL_SKILL_RELATIONS: dict[str, list[str]] = {
        # Testing tools
        "pytest": ["Testing Strategy", "Test Coverage Thoroughness"],
        "Jest": ["Testing Strategy", "Test Coverage Thoroughness"],
        "Vitest": ["Testing Strategy", "Test Coverage Thoroughness"],
        "unittest": ["Testing Strategy"],
        "Playwright": ["Testing Strategy"],
        "MSW": ["Testing Strategy"],
        # Container / orchestration
        "Docker": ["Environment Management", "CI/CD Sophistication"],
        "Kubernetes": ["Environment Management", "CI/CD Sophistication"],
        # CI/CD
        "GitHub Actions": ["CI/CD Sophistication"],
        "GitLab CI": ["CI/CD Sophistication"],
        "Jenkins": ["CI/CD Sophistication"],
        "CircleCI": ["CI/CD Sophistication"],
        # Version control
        "Git": ["Git Workflow Maturity"],
        "GitHub": ["Git Workflow Maturity"],
        # Quality / linting
        "ESLint": ["Code Quality Enforcement"],
        "Prettier": ["Code Quality Enforcement"],
        "Ruff": ["Code Quality Enforcement"],
        "Biome": ["Code Quality Enforcement"],
        # Package management
        "Poetry": ["Dependency Management"],
        "uv": ["Dependency Management"],
        "pnpm": ["Dependency Management"],
        "Bun": ["Dependency Management"],
        "Yarn": ["Dependency Management"],
        # Documentation
        "Sphinx": ["Documentation Quality"],
        "MkDocs": ["Documentation Quality"],
        "LaTeX": ["Documentation Quality"],
        "Jupyter": ["Documentation Quality"],
        "Markdown": ["Documentation Quality"],
        # ML / data
        "PyTorch": ["Machine Learning"],
        "scikit-learn": ["Machine Learning"],
        "Pandas": ["Data Engineering", "Scientific Computing"],
        "NumPy": ["Scientific Computing"],
        # Web frameworks
        "Django": ["Web Frameworks", "API Design"],
        "Flask": ["Web Frameworks", "API Design"],
        "FastAPI": ["API Design"],
        "React": ["Frontend Development"],
        "Vue": ["Frontend Development"],
        "Svelte": ["Frontend Development"],
        "SolidJS": ["Frontend Development"],
        "Next.js": ["Frontend Development"],
        "Nuxt": ["Frontend Development"],
        "Astro": ["Frontend Development"],
        # API / data layer
        "GraphQL": ["API Design"],
        "tRPC": ["API Design"],
        "Prisma": ["Data Modeling"],
        "Drizzle": ["Data Modeling"],
        # CSS / styling
        "TailwindCSS": ["Frontend Development"],
        "shadcn/ui": ["Frontend Development"],
        # Build / tooling
        "Vite": ["Build & Tooling"],
        "Webpack": ["Build & Tooling"],
        "Turbopack": ["Build & Tooling"],
        # Type system
        "TypeScript": ["Type System Proficiency"],
        # State management
        "Zustand": ["Design Patterns"],
        "Redux": ["Design Patterns"],
        "TanStack Query": ["Design Patterns"],
        # IaC
        "Terraform": ["Infrastructure Design"],
        "Ansible": ["Infrastructure Design"],
        # Languages (ecosystem indicators)
        "Python": ["Python", "Type System Proficiency"],
        "Rust": ["Performance Awareness"],
        "Go": ["Performance Awareness"],
    }

    # ── Known skill ↔ skill relationships ──
    _SKILL_SKILL_RELATIONS: dict[str, list[tuple[str, str]]] = {
        "Testing Strategy": [
            ("Test Coverage Thoroughness", "reinforces"),
            ("CI/CD Sophistication", "enables"),
        ],
        "Test Coverage Thoroughness": [
            ("Testing Strategy", "reinforces"),
            ("Code Quality Enforcement", "supports"),
        ],
        "Code Quality Enforcement": [
            ("Testing Strategy", "complements"),
            ("Documentation Quality", "related_to"),
        ],
        "Design Patterns": [
            ("Code Organization", "enables"),
            ("Component Design", "supports"),
            ("Type System Proficiency", "reinforces"),
        ],
        "Error Handling Maturity": [
            ("Security Awareness", "reinforces"),
            ("Testing Strategy", "supports"),
        ],
        "Type System Proficiency": [
            ("Design Patterns", "supports"),
            ("Code Organization", "enables"),
        ],
        "Code Organization": [
            ("Design Patterns", "supports"),
            ("Component Design", "enables"),
        ],
        "Performance Awareness": [
            ("Code Organization", "related_to"),
            ("Infrastructure Design", "related_to"),
        ],
        "Security Awareness": [
            ("Error Handling Maturity", "reinforces"),
            ("Code Quality Enforcement", "complements"),
        ],
        "Git Workflow Maturity": [
            ("CI/CD Sophistication", "enables"),
            ("Incremental Development", "supports"),
        ],
        "CI/CD Sophistication": [
            ("Environment Management", "enables"),
            ("Testing Strategy", "enables"),
            ("Git Workflow Maturity", "enables"),
        ],
        "Dependency Management": [
            ("CI/CD Sophistication", "supports"),
            ("Security Awareness", "supports"),
        ],
        "Environment Management": [
            ("CI/CD Sophistication", "enables"),
            ("Dependency Management", "related_to"),
            ("Infrastructure Design", "related_to"),
        ],
        "Documentation Quality": [
            ("Code Organization", "supports"),
            ("Testing Strategy", "related_to"),
        ],
        "Incremental Development": [
            ("Git Workflow Maturity", "supports"),
            ("CI/CD Sophistication", "related_to"),
        ],
        "Component Design": [
            ("Design Patterns", "supported_by"),
            ("API Design", "related_to"),
            ("Code Organization", "supported_by"),
        ],
        "API Design": [
            ("Component Design", "related_to"),
            ("Data Modeling", "related_to"),
        ],
        "Data Modeling": [
            ("API Design", "related_to"),
            ("Infrastructure Design", "related_to"),
        ],
        "Infrastructure Design": [
            ("CI/CD Sophistication", "related_to"),
            ("Environment Management", "related_to"),
        ],
        "Scalability": [
            ("Performance Awareness", "related_to"),
            ("Infrastructure Design", "enables"),
        ],
        "Argumentation": [
            ("Clarity", "reinforces"),
            ("Technical Writing", "supports"),
            ("Narrative Structure", "related_to"),
        ],
        "Technical Writing": [
            ("Clarity", "supports"),
            ("Documentation Quality", "related_to"),
        ],
        "Narrative Structure": [
            ("Argumentation", "supports"),
            ("Clarity", "related_to"),
        ],
        "Clarity": [
            ("Technical Writing", "reinforces"),
            ("Documentation Quality", "supports"),
        ],
        "Academic Rigor": [
            ("Documentation Quality", "supports"),
            ("Argumentation", "related_to"),
        ],
    }

    # ── Collect names for lookup ──
    tool_names: set[str] = {t.get("name", "") for t in tools_list if t.get("name")}
    skill_names: set[str] = {s.get("name", "") for s in skills_list if s.get("name")}

    # Skill name → category lookup
    skill_categories: dict[str, str] = {
        s.get("name", ""): s.get("category", "Unknown")
        for s in skills_list if s.get("name")
    }

    # ── Build nodes ──
    for tool in tools_list:
        name = tool.get("name", "")
        if not name or name in seen_node_ids:
            continue
        seen_node_ids.add(name)
        nodes.append({
            "id": name,
            "type": "tool",
            "category": tool.get("category", "Other"),
        })

    for skill in skills_list:
        name = skill.get("name", "")
        if not name or name in seen_node_ids:
            continue
        seen_node_ids.add(name)
        nodes.append({
            "id": name,
            "type": "skill",
            "category": skill.get("category", "Unknown"),
        })

    # ── Build edges: tool → skill ──
    edge_keys: set[tuple[str, str]] = set()
    for tool_name in tool_names:
        related_skills = _TOOL_SKILL_RELATIONS.get(tool_name, [])
        for skill_name in related_skills:
            if skill_name in skill_names:
                key = (tool_name, skill_name)
                if key not in edge_keys:
                    edge_keys.add(key)
                    edges.append({
                        "source": tool_name,
                        "target": skill_name,
                        "relationship": "indicates",
                    })

    # ── Build edges: skill ↔ skill (known relationships) ──
    for skill_name in skill_names:
        related = _SKILL_SKILL_RELATIONS.get(skill_name, [])
        for target_name, relationship in related:
            if target_name in skill_names:
                # Normalize to avoid duplicate undirected edges
                key = tuple(sorted([skill_name, target_name]))
                if key not in edge_keys:
                    edge_keys.add(key)
                    edges.append({
                        "source": skill_name,
                        "target": target_name,
                        "relationship": relationship,
                    })

    # ── Build edges: same-category skills ──
    by_category: dict[str, list[str]] = {}
    for skill in skills_list:
        cat = skill.get("category", "Unknown")
        name = skill.get("name", "")
        if name:
            by_category.setdefault(cat, []).append(name)

    for cat, names in by_category.items():
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                key = tuple(sorted([names[i], names[j]]))
                if key not in edge_keys and key not in edge_keys:
                    edge_keys.add(key)
                    edges.append({
                        "source": names[i],
                        "target": names[j],
                        "relationship": "same_category",
                    })

    return {
        "nodes": nodes,
        "edges": edges,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Aggregator Role
# ═══════════════════════════════════════════════════════════════════════════

class SkillAggregatorRole(AgentRole):
    """Read 4 skill-dimension reports, deduplicate entries, resolve proficiency,
    categorize skills, and write ``skill_inventory.json``.

    v2: handles both ``detected_tools`` AND ``inferred_skills`` from the
    4 new detectors (Domain Expertise, Technical Craft, Methodology, Rigor).
    Cross-references skills found by multiple detectors, merges evidence
    arrays, and generates a skill graph.
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
        """Build the aggregation prompt (v2: requests cross_referenced + skill_graph)."""
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
            f"highest proficiency, merge sources and evidence.\n"
            f"4. **Deduplicate skills**: same skill name in multiple reports → "
            f"keep highest proficiency and confidence, merge evidence arrays "
            f"from ALL sources (don't discard evidence from lower-proficiency "
            f"detectors).\n"
            f"5. **Cross-referencing**: when the SAME skill is detected by "
            f"MULTIPLE detectors, mark it with `cross_referenced: true` and "
            f"a `cross_referenced_sources` list. Cross-referenced skills are "
            f"stronger signals — boost their confidence by one level "
            f"(low→medium, medium→high).\n"
            f"6. **Categorize**: Map each entry to a top-level category. "
            f"Tools → Languages & Runtimes, Testing, Web Frameworks, etc. "
            f"Skills → Domain Knowledge, Technical Craft, Engineering Practice, "
            f"Quality Assurance, Architecture.\n"
            f"7. **Skill Graph**: generate a skill graph showing relationships:\n"
            f"   - Tool nodes connected to the skills they indicate "
            f"(e.g., pytest → Testing Strategy)\n"
            f"   - Skill-to-skill relationships (e.g., Testing Strategy ↔ "
            f"Test Coverage Thoroughness)\n"
            f"   - Same-category skill connections\n"
            f"8. **Count files**: Compute summary stats from the project scan.\n\n"
            f"9. Write `skill_inventory.json`:\n"
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
            f'      "cross_referenced": false,\n'
            f'      "cross_referenced_sources": [],\n'
            f'      "evidence": {{"description": "..."}}\n'
            f'    }},\n'
            f'    {{\n'
            f'      "name": "Testing Strategy", "category": "Quality Assurance",\n'
            f'      "proficiency": "intermediate", "confidence": "high",\n'
            f'      "sources": ["Depth & Rigor", "Methodology & Tooling"],\n'
            f'      "cross_referenced": true,\n'
            f'      "cross_referenced_sources": ["Depth & Rigor", "Methodology & Tooling"],\n'
            f'      "evidence": {{"description": "..."}}\n'
            f'    }}\n'
            f'  ],\n'
            f'  "skill_graph": {{\n'
            f'    "nodes": [\n'
            f'      {{"id": "pytest", "type": "tool", "category": "Testing"}},\n'
            f'      {{"id": "Testing Strategy", "type": "skill", "category": "Quality Assurance"}}\n'
            f'    ],\n'
            f'    "edges": [\n'
            f'      {{"source": "pytest", "target": "Testing Strategy", "relationship": "indicates"}}\n'
            f'    ]\n'
            f'  }},\n'
            f'  "file_stats": {{}}\n'
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
        """Parse aggregated inventory from agent or fall back to rule-based.

        v2: Accepts inventory with ``tools``, ``inferred_skills``, or ``skills`` keys.
        """
        inventory: dict[str, Any] = {}

        # 1. Agent response
        for msg in reversed(result.messages):
            content = msg.content if hasattr(msg, "content") else str(msg)
            json_str = extract_json_object(content)
            if json_str:
                try:
                    parsed = json.loads(json_str)
                    if ("skills" in parsed or "summary" in parsed or
                        "tools" in parsed or "inferred_skills" in parsed):
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
                    if isinstance(parsed, dict) and (
                        "skills" in parsed or "tools" in parsed or
                        "inferred_skills" in parsed
                    ):
                        inventory = parsed
                except (json.JSONDecodeError, OSError):
                    pass

        # 3. Rule-based fallback
        if not inventory:
            inventory = self._fallback_aggregator(project_dir, working_dir)

        return inventory

    # ── Fallback aggregator (v2) ─────────────────────────────────────────

    @staticmethod
    def _fallback_aggregator(project_dir: str = ".",
                             working_dir: str = ".") -> dict[str, Any]:
        """Rule-based deduplication and categorization from skill-dimension reports.

        Reads the four new detector report files from disk and performs
        deterministic merging without LLM. Handles both detected_tools
        and inferred_skills from each report.

        v2 improvements:
        - Category inference via _infer_category() (replaces hardcoded map)
        - Evidence merging when same skill appears in multiple detectors
        - cross_referenced flag + confidence boost for multi-detector skills
        - skill_graph generation
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
                    # v2: merge evidence arrays
                    ex_ev = existing.get("evidence", [])
                    if isinstance(ex_ev, list):
                        new_ev = tool.get("evidence", [])
                        if isinstance(new_ev, list):
                            # Merge and deduplicate
                            merged_ev: list[Any] = list(ex_ev)
                            for item in new_ev:
                                if item not in merged_ev:
                                    merged_ev.append(item)
                            existing["evidence"] = merged_ev[:15]
                else:
                    tool_copy = dict(tool)
                    tool_copy["sources"] = [domain]
                    raw_cat = tool_copy.get("category", "Other Tools")
                    tool_copy["category"] = _CATEGORY_MAP.get(raw_cat, raw_cat)
                    # Ensure evidence is a list
                    if "evidence" not in tool_copy:
                        tool_copy["evidence"] = []
                    tools_seen[name] = tool_copy

        tools_list = sorted(tools_seen.values(), key=lambda s: (
            _PROFICIENCY_SCORES.get(s.get("proficiency", "beginner"), 1),
            s.get("name", ""),
        ), reverse=True)

        # ── Merge inferred skills (v2: evidence merging + cross_ref) ──
        skills_seen: dict[str, dict[str, Any]] = {}

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

                    # Keep highest proficiency
                    if new_prof > old_prof:
                        existing["proficiency"] = skill.get("proficiency")
                        existing["confidence"] = skill.get("confidence", "medium")

                    # Track sources
                    if domain not in existing.get("sources", []):
                        existing["sources"].append(domain)

                    # ── v2: Merge evidence arrays from ALL sources ──
                    existing["evidence"] = _merge_evidence(
                        existing.get("evidence", {}),
                        skill.get("evidence", {}),
                    )

                    # ── v2: Merge evidence_refs ──
                    existing["evidence_refs"] = _merge_evidence_refs(
                        existing.get("evidence_refs", []),
                        skill.get("evidence_refs", []),
                    )

                    # ── v2: Mark cross_referenced ──
                    existing["cross_referenced"] = True
                    if skill.get("cross_referenced"):
                        existing["cross_referenced"] = True

                    # ── v2: Add cross_referenced_sources ──
                    # Seed cross_referenced_sources from existing sources on first
                    # cross-reference, so both detector domains are captured.
                    existing_srcs = set(existing.get("cross_referenced_sources", []))
                    if not existing_srcs and existing.get("sources"):
                        # First cross-reference: start with the original source(s)
                        existing_srcs = set(existing["sources"])
                    existing_srcs.add(domain)
                    existing["cross_referenced_sources"] = sorted(existing_srcs)

                    # ── v2: Boost confidence for cross-referenced skills ──
                    current_conf = existing.get("confidence", "medium")
                    existing["confidence"] = _boost_confidence(current_conf)

                else:
                    skill_copy = dict(skill)
                    skill_copy["sources"] = [domain]
                    # v2: Use _infer_category instead of hardcoded map
                    skill_copy["category"] = _infer_category(name, domain)
                    skill_copy["cross_referenced"] = False
                    skill_copy["cross_referenced_sources"] = []
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

        # ── v2: Generate skill graph ──────────────────────────────────
        skill_graph = _generate_skill_graph(tools_list, skills_list)

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
            "skill_graph": skill_graph,
            "_fallback": True,
        }
