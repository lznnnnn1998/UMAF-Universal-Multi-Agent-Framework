"""SkillScannerRole v2 — artifact classification + deep content reading + structure analysis.

Produces two output files:
- ``project_scan.json`` — surface-level file listing (backward-compatible)
- ``artifact_analysis.json`` — deep analysis: artifact type, content samples, structure, key_files
"""

import json
import os
import re
import subprocess
from typing import Any

from agent import AgentResult, AgentRole
from tools import ToolRegistry
from utils import extract_json_object, safe_read


# ── Artifact type classification ──────────────────────────────────────────

_ARTIFACT_SIGNATURES: dict[str, dict[str, Any]] = {
    "software_project": {
        "description": "A software project — contains source code, tests, build configs",
        "indicators": {
            "source_exts": {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs",
                           ".java", ".c", ".cpp", ".h", ".hpp", ".rb", ".swift",
                           ".kt", ".scala", ".php", ".cs", ".ex", ".exs"},
            "build_files": {"setup.py", "pyproject.toml", "package.json", "Cargo.toml",
                           "go.mod", "Makefile", "CMakeLists.txt", "build.gradle",
                           "pom.xml", "mix.exs"},
            "min_source_ratio": 0.15,
        },
    },
    "research_paper": {
        "description": "A research paper or academic document — LaTeX, PDF, citations",
        "indicators": {
            "exts": {".tex", ".pdf", ".bib", ".bbl"},
            "section_headers": [r"\section{", r"\subsection{", "Abstract",
                               "Introduction", "Related Work", "Methodology",
                               "Results", "Conclusion", "References"],
        },
    },
    "blog_article": {
        "description": "A blog post or web article — markdown, HTML, images",
        "indicators": {
            "exts": {".md", ".mdx", ".html", ".xml"},
            "frontmatter_keys": ["title:", "date:", "author:", "tags:", "draft:"],
            "platform_indicators": ["_posts/", "content/", "blog/", "articles/"],
        },
    },
    "documentation": {
        "description": "Technical documentation — structured docs, API references",
        "indicators": {
            "exts": {".md", ".rst", ".adoc", ".txt"},
            "doc_tools": {"mkdocs.yml", "conf.py", "docusaurus.config.js",
                         "docsify", "gitbook", "readthedocs", "sphinx"},
            "high_doc_ratio": 0.4,
        },
    },
    "dataset": {
        "description": "A dataset — data files, schemas, processing scripts",
        "indicators": {
            "data_exts": {".csv", ".json", ".jsonl", ".parquet", ".sqlite",
                         ".db", ".h5", ".npz", ".arrow", ".avro"},
            "schema_files": {"schema.sql", "schema.json", "datapackage.json"},
            "min_data_ratio": 0.3,
        },
    },
    "design_document": {
        "description": "A design document or specification",
        "indicators": {
            "exts": {".md", ".pdf", ".png", ".jpg", ".svg", ".fig", ".drawio"},
            "keywords": ["architecture", "design", "specification", "RFC",
                        "proposal", "wireframe", "mockup", "diagram"],
        },
    },
    "presentation": {
        "description": "A presentation or slide deck",
        "indicators": {
            "exts": {".pptx", ".key", ".pdf", ".md", ".html"},
            "tools": {"reveal.js", "remark.js", "slidev", "marp", "beamer"},
        },
    },
    "configuration": {
        "description": "Configuration files, dotfiles, or infrastructure-as-code",
        "indicators": {
            "high_config_ratio": 0.6,
            "config_dirs": {".config", "dotfiles", "infra", "terraform",
                           "kubernetes", "ansible", "puppet"},
        },
    },
}


def _classify_artifact(file_list: list[str], content_samples: dict[str, str],
                       project_dir: str) -> dict[str, Any]:
    """Deterministic artifact type classification based on file patterns and content."""
    if not file_list:
        return {"type": "empty", "confidence": "high",
                "description": "No files found to analyze"}

    scores: dict[str, float] = {}
    total = len(file_list)
    exts = {os.path.splitext(f)[1].lower() for f in file_list}
    basenames = {os.path.basename(f) for f in file_list}
    all_text = " ".join(content_samples.values()).lower()

    for atype, sig in _ARTIFACT_SIGNATURES.items():
        score = 0.0
        ind = sig["indicators"]

        # Source file ratio check
        if "source_exts" in ind:
            source_count = sum(1 for f in file_list
                              if os.path.splitext(f)[1].lower() in ind["source_exts"])
            if source_count / max(total, 1) >= ind.get("min_source_ratio", 0.15):
                score += 3.0

        # Extension check
        if "exts" in ind:
            match = exts & ind["exts"]
            if match:
                score += len(match) * 0.5

        # Data file ratio check
        if "data_exts" in ind:
            data_count = sum(1 for f in file_list
                           if os.path.splitext(f)[1].lower() in ind["data_exts"])
            if data_count / max(total, 1) >= ind.get("min_data_ratio", 0.3):
                score += 3.0

        # Build file check
        if "build_files" in ind:
            build_match = basenames & ind["build_files"]
            if build_match:
                score += len(build_match) * 1.5

        # Doc tool check
        if "doc_tools" in ind:
            doc_match = basenames & ind["doc_tools"]
            if doc_match:
                score += 2.0

        # High config ratio
        if "high_config_ratio" in ind:
            config_exts = {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".conf"}
            config_count = sum(1 for f in file_list
                             if os.path.splitext(f)[1].lower() in config_exts)
            if config_count / max(total, 1) >= ind["high_config_ratio"]:
                score += 3.0

        # Config dirs
        if "config_dirs" in ind:
            for f in file_list:
                parts = f.split("/")
                if any(cd in parts for cd in ind["config_dirs"]):
                    score += 1.0
                    break

        # Platform indicators in paths
        if "platform_indicators" in ind:
            for f in file_list:
                if any(pi in f for pi in ind["platform_indicators"]):
                    score += 1.5
                    break

        # Keyword check in content
        if "keywords" in ind:
            kw_matches = sum(1 for kw in ind["keywords"] if kw.lower() in all_text)
            score += kw_matches * 0.5

        # Section headers in content
        if "section_headers" in ind:
            sh_matches = sum(1 for sh in ind["section_headers"] if sh.lower() in all_text)
            score += sh_matches * 1.0

        # Frontmatter keys in content
        if "frontmatter_keys" in ind:
            fm_matches = sum(1 for fk in ind["frontmatter_keys"] if fk.lower() in all_text)
            score += fm_matches * 0.5

        if score > 0:
            scores[atype] = score

    if not scores:
        return {"type": "unknown", "confidence": "low",
                "description": "Could not classify — no clear signature matched",
                "candidate_scores": {}}

    best = max(scores, key=lambda k: scores[k])  # type: ignore[call-overload]
    best_score = scores[best]
    runner_up = sorted(scores, key=lambda k: scores[k], reverse=True)  # type: ignore[call-overload]

    confidence = "high" if best_score >= 5 else "medium" if best_score >= 2.5 else "low"

    return {
        "type": best,
        "confidence": confidence,
        "description": _ARTIFACT_SIGNATURES.get(best, {}).get("description", ""),
        "candidate_scores": {k: round(scores[k], 1) for k in runner_up[:5]},
    }


# ── File complexity scoring ───────────────────────────────────────────────

# Structural keywords that indicate code complexity and skill demonstration
_STRUCTURAL_KEYWORDS: set[str] = {
    "class ", "def ", "async def ", "async ", "await ", "try:", "except ",
    "import ", "from ", "return ", "yield ", "raise ", "with ", "lambda ",
    "@", "TypeVar", "Generic", "Protocol", "dataclass", "Enum", "ABC",
    "__init__", "__call__", "__enter__", "__exit__", "__iter__", "__next__",
    "if __name__", "assert ", "match ", "case ",
}

# Markers of generated/minified files to exclude from sampling
_GENERATED_FILE_MARKERS: set[str] = {".min.", ".generated.", ".gen.", ".bundle.",
                                     ".compiled.", ".optimized."}
_GENERATED_DIR_MARKERS: set[str] = {"/dist/", "/build/", "/__pycache__/",
                                    "/node_modules/", "/.venv/", "/venv/",
                                    "/vendor/", "/.eggs/", "/.tox/",
                                    "/coverage/", "/htmlcov/", "/site-packages/",
                                    "/migrations/", "/fixtures/"}
_GENERATED_PATH_PATTERNS: list[str] = [
    r"(^|/)dist/", r"(^|/)build/", r"(^|/)__pycache__/",
    r"(^|/)node_modules/", r"(^|/)\.venv/", r"(^|/)venv/",
    r"(^|/)vendor/", r"(^|/)\.eggs/", r"(^|/)\.tox/",
    r"(^|/)coverage/", r"(^|/)htmlcov/", r"(^|/)site-packages/",
    r"(^|/)migrations/", r"(^|/)fixtures/",
]


def _is_generated_file(file_path: str, content: str | None = None) -> bool:
    """Detect generated, minified, or auto-produced files that have low skill signal."""
    basename = os.path.basename(file_path)
    ext = os.path.splitext(basename)[1].lower()

    # Check filename markers
    for marker in _GENERATED_FILE_MARKERS:
        if marker in basename:
            return True

    # Common generated file names
    generated_names = {
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
        "poetry.lock", "Pipfile.lock", "Gemfile.lock",
        "requirements.txt",  # auto-generated lock style
    }
    if basename in generated_names:
        # Lock files are generated but requirements.txt may be hand-written;
        # check surrounding context — if alone among many generated files, skip
        if basename == "requirements.txt":
            return False
        return True

    # Check path patterns for generated directories
    for pattern in _GENERATED_PATH_PATTERNS:
        if re.search(pattern, file_path):
            return True

    # Check content for minification heuristics
    if content and len(content) > 100:
        lines = content.split("\n")
        # Minified files have very few lines relative to content length
        if len(lines) > 0 and len(content) / len(lines) > 500:
            return True
        # Extremely long lines (>1000 chars) are typical of minified JS/CSS
        if any(len(line) > 1000 for line in lines):
            return True

    # Source map files
    if ext == ".map" and basename.endswith(".map"):
        return True

    # Compiled Python bytecode
    if ext in {".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib"}:
        return True

    # Minified CSS/JS naming patterns
    if (basename.endswith(".min.css") or basename.endswith(".min.js") or
        ".min." in basename):
        return True

    return False


def _compute_file_complexity(file_path: str, content: str | None = None) -> float:
    """Compute a complexity score for a file to prioritize skill-demonstrative files.

    Returns a float 0.0-1.0. Higher = more likely to demonstrate skills.
    Factors: structural keyword density, file size, and pattern diversity.
    """
    if content is None:
        content = safe_read(file_path)
        if not content:
            return 0.0

    lines = content.split("\n")
    total_chars = max(len(content), 1)
    num_lines = max(len(lines), 1)

    # 1. Structural keyword density (0.0 - 0.5)
    keyword_count = sum(content.count(kw) for kw in _STRUCTURAL_KEYWORDS)
    # Normalize: expect ~1 keyword per 50 chars in dense code
    keyword_density = min(keyword_count / (total_chars / 50), 1.0)
    keyword_score = keyword_density * 0.5

    # 2. File size score — log-scale, peaks at ~10KB (0.0 - 0.25)
    size_score = min(total_chars / 10_000, 1.0) * 0.25

    # 3. Pattern diversity — unique structural keywords used (0.0 - 0.25)
    unique_keywords = sum(1 for kw in _STRUCTURAL_KEYWORDS if kw in content)
    diversity = min(unique_keywords / max(len(_STRUCTURAL_KEYWORDS) * 0.3, 1), 1.0)
    diversity_score = diversity * 0.25

    return round(keyword_score + size_score + diversity_score, 3)


# ── Body paragraph detection for articles/papers ─────────────────────────

def _is_body_content(content: str) -> float:
    """Estimate how much of a text file is body content vs headers/frontmatter.

    Returns a ratio 0.0-1.0. Higher = more body content.
    Body content = paragraphs of prose, not metadata or headers.
    """
    if not content:
        return 0.0

    lines = content.split("\n")
    if len(lines) < 3:
        return 0.0

    # Count lines that look like body paragraphs (prose, not headers/metadata)
    body_lines = 0
    total_lines = 0

    # Frontmatter detection: YAML-style `---` delimited blocks at file start
    in_frontmatter = False
    fm_delim_count = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        total_lines += 1

        # Detect YAML frontmatter boundaries at file start
        if i < 20 and stripped == "---":
            fm_delim_count += 1
            in_frontmatter = fm_delim_count == 1
            if fm_delim_count == 2:
                in_frontmatter = False
            continue

        if in_frontmatter:
            continue

        # Skip empty lines
        if not stripped:
            continue

        # Skip markdown headers (# Title, ## Section, etc.)
        if re.match(r"^#{1,6}\s", stripped):
            continue

        # Skip horizontal rules
        if re.match(r"^[-*=]{3,}$", stripped):
            continue

        # Skip code blocks (``` fences)
        if stripped.startswith("```"):
            continue

        # Skip table rows (| ... | ... |)
        if stripped.startswith("|") and stripped.endswith("|"):
            continue

        # Skip list items that are very short (metadata-like)
        if re.match(r"^[-*+]\s+\S+:", stripped) and len(stripped) < 40:
            continue

        # A body line is prose: reasonably long, contains spaces, not a single token
        words = stripped.split()
        if len(words) >= 5 and len(stripped) >= 40:
            body_lines += 1
        elif len(stripped) >= 80:
            body_lines += 1

    if total_lines == 0:
        return 0.0

    return round(body_lines / total_lines, 3)


def _sample_content(project_dir: str, working_dir: str,
                    file_list: list[str]) -> dict[str, str]:
    """Read content from key files to understand the artifact.

    Sampling priority (v2):
    1. Skill-demonstrative files: complex source files (software) or body-rich
       documents (articles/papers) — these carry the richest skill signal.
    2. README/config/build files as secondary fallback.
    3. Remaining diverse files by extension to ensure coverage.

    Generated/minified files are excluded from sampling.
    """
    samples: dict[str, str] = {}
    max_samples = 12
    max_chars = 4000  # per file (v2: doubled from 2000)

    # Resolve base directory
    if os.path.isabs(project_dir):
        base = project_dir
    else:
        base = os.path.abspath(project_dir)
    if os.path.isfile(base):
        base = os.path.dirname(base)

    # Source code extensions for complexity scoring
    source_exts = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs",
                   ".java", ".c", ".cpp", ".h", ".hpp", ".rb", ".swift",
                   ".kt", ".scala", ".php", ".cs", ".ex", ".exs"}

    # Article/document extensions for body-content scoring
    prose_exts = {".md", ".mdx", ".rst", ".adoc", ".txt", ".tex"}

    # Determine artifact lean from file extensions
    source_count = sum(1 for f in file_list
                       if os.path.splitext(f)[1].lower() in source_exts)
    prose_count = sum(1 for f in file_list
                      if os.path.splitext(f)[1].lower() in prose_exts)
    total = max(len(file_list), 1)
    is_likely_software = source_count / total >= 0.10
    is_likely_article = prose_count / total >= 0.30 and source_count / total < 0.10

    sampled = 0

    # ═══════════════════════════════════════════════════════════════════════
    # Pass 1: Skill-demonstrative files (highest priority)
    # ═══════════════════════════════════════════════════════════════════════

    if is_likely_software:
        # For software projects: score source files by complexity and sample top ones
        scored_files: list[tuple[float, str]] = []
        for fpath in file_list:
            ext = os.path.splitext(fpath)[1].lower()
            if ext not in source_exts:
                continue
            if _is_generated_file(fpath):
                continue
            full_path = os.path.join(base, fpath)
            raw = safe_read(full_path)
            if not raw:
                continue
            score = _compute_file_complexity(full_path, raw)
            scored_files.append((score, fpath))

        # Sort by complexity descending; sample top files, one per directory
        scored_files.sort(key=lambda x: x[0], reverse=True)
        seen_dirs_complex: set[str] = set()
        for score, fpath in scored_files:
            if sampled >= max_samples:
                break
            d = os.path.dirname(fpath)
            if d in seen_dirs_complex and sampled >= 6:
                # After 6 files, prefer diverse directories
                continue
            if d:
                seen_dirs_complex.add(d)
            full_path = os.path.join(base, fpath)
            content = safe_read(full_path)
            if content:
                samples[fpath] = content[:max_chars]
                sampled += 1

    elif is_likely_article:
        # For articles/papers: prioritize files with high body content ratio
        scored_articles: list[tuple[float, str]] = []
        for fpath in file_list:
            ext = os.path.splitext(fpath)[1].lower()
            if ext not in prose_exts:
                continue
            if _is_generated_file(fpath):
                continue
            full_path = os.path.join(base, fpath)
            raw = safe_read(full_path)
            if not raw:
                continue
            body_ratio = _is_body_content(raw)
            scored_articles.append((body_ratio, fpath))

        scored_articles.sort(key=lambda x: x[0], reverse=True)
        seen_dirs_prose: set[str] = set()
        for body_ratio, fpath in scored_articles:
            if sampled >= max_samples:
                break
            if body_ratio == 0.0:
                continue
            d = os.path.dirname(fpath)
            if d in seen_dirs_prose and sampled >= 6:
                continue
            if d:
                seen_dirs_prose.add(d)
            full_path = os.path.join(base, fpath)
            content = safe_read(full_path)
            if content:
                samples[fpath] = content[:max_chars]
                sampled += 1

    else:
        # Mixed/unknown artifact: use complexity scoring for any source files
        # plus body-content scoring for prose files
        hybrid_scored: list[tuple[float, str, str]] = []  # (score, path, kind)
        for fpath in file_list:
            ext = os.path.splitext(fpath)[1].lower()
            if _is_generated_file(fpath):
                continue
            full_path = os.path.join(base, fpath)
            raw = safe_read(full_path)
            if not raw:
                continue
            if ext in source_exts:
                score = _compute_file_complexity(full_path, raw)
                hybrid_scored.append((score, fpath, "source"))
            elif ext in prose_exts:
                score = _is_body_content(raw)
                if score > 0.0:
                    hybrid_scored.append((score, fpath, "prose"))

        hybrid_scored.sort(key=lambda x: x[0], reverse=True)
        seen_dirs_hybrid: set[str] = set()
        for score, fpath, _kind in hybrid_scored:
            if sampled >= max_samples:
                break
            d = os.path.dirname(fpath)
            if d in seen_dirs_hybrid and sampled >= 6:
                continue
            if d:
                seen_dirs_hybrid.add(d)
            full_path = os.path.join(base, fpath)
            content = safe_read(full_path)
            if content:
                samples[fpath] = content[:max_chars]
                sampled += 1

    # ═══════════════════════════════════════════════════════════════════════
    # Pass 2: README/config/documentation files (secondary priority)
    # ═══════════════════════════════════════════════════════════════════════
    priority_patterns = [
        "README", "readme", "CONTRIBUTING", "CHANGELOG", "LICENSE",
        "pyproject.toml", "package.json", "Cargo.toml", "go.mod",
        "Makefile", "Dockerfile", "docker-compose",
        ".github/workflows", ".gitlab-ci.yml",
    ]

    for pattern in priority_patterns:
        if sampled >= max_samples:
            break
        for fpath in file_list:
            if sampled >= max_samples:
                break
            if fpath in samples:
                continue
            if pattern.lower() in fpath.lower():
                full_path = os.path.join(base, fpath)
                content = safe_read(full_path)
                if content:
                    samples[fpath] = content[:max_chars]
                    sampled += 1

    # ═══════════════════════════════════════════════════════════════════════
    # Pass 3: Remaining diverse files by extension (coverage fallback)
    # ═══════════════════════════════════════════════════════════════════════
    seen_exts: set[str] = set()
    for fpath in file_list:
        if sampled >= max_samples:
            break
        if fpath in samples:
            continue
        if _is_generated_file(fpath):
            continue
        ext = os.path.splitext(fpath)[1].lower()
        if ext not in seen_exts:
            seen_exts.add(ext)
            full_path = os.path.join(base, fpath)
            content = safe_read(full_path)
            if content:
                samples[fpath] = content[:max_chars]
                sampled += 1

    return samples


def _analyze_structure(file_list: list[str], project_dir: str) -> dict[str, Any]:
    """Analyze the artifact's structural organization."""
    if not file_list:
        return {"top_level_components": [], "directory_depth": 0}

    # Build directory tree
    dirs: dict[str, list[str]] = {}
    for fpath in file_list:
        d = os.path.dirname(fpath) or "."
        dirs.setdefault(d, []).append(fpath)

    # Top-level components
    top_dirs: set[str] = set()
    for fpath in file_list:
        parts = fpath.split("/")
        if len(parts) > 1:
            top_dirs.add(parts[0])

    # Directory depth
    max_depth = max((len(f.split("/")) for f in file_list), default=0)

    # Identify component-like dirs (contain multiple related files)
    components: list[dict[str, Any]] = []
    for d, files in sorted(dirs.items()):
        if d == ".":
            continue
        if len(files) >= 2:
            exts = {os.path.splitext(f)[1].lower() for f in files}
            components.append({
                "path": d,
                "file_count": len(files),
                "extensions": sorted(exts),
                "description": f"Directory with {len(files)} files",
            })

    return {
        "top_level_components": sorted(top_dirs) if top_dirs else ["."],
        "directory_depth": max_depth,
        "total_directories": len(dirs),
        "sub_components": components[:20],  # top 20 by file count
    }


# ── Key files identification ──────────────────────────────────────────────

def _identify_key_files(file_list: list[str], content_samples: dict[str, str],
                        project_dir: str) -> list[dict[str, Any]]:
    """Identify the most skill-demonstrative files from the project.

    Each entry contains:
    - path: relative file path
    - rationale: why this file demonstrates skills
    - skill_indicators: structural keywords/patterns found
    - complexity_score: 0.0-1.0 based on keyword density, size, and pattern diversity

    Used by downstream detectors to reference specific evidence and by the
    report writer to cite concrete examples.
    """
    # Resolve base directory
    if os.path.isabs(project_dir):
        base = project_dir
    else:
        base = os.path.abspath(project_dir)
    if os.path.isfile(base):
        base = os.path.dirname(base)

    source_exts = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs",
                   ".java", ".c", ".cpp", ".h", ".hpp", ".rb", ".swift",
                   ".kt", ".scala", ".php", ".cs", ".ex", ".exs"}
    prose_exts = {".md", ".mdx", ".rst", ".adoc", ".txt", ".tex"}

    # Build rationale descriptions from structural indicators
    def _build_rationale(fpath: str, content: str, score: float,
                         indicators: list[str]) -> str:
        """Generate a human-readable rationale for why this file is skill-demonstrative."""
        ext = os.path.splitext(fpath)[1].lower()
        dir_name = os.path.dirname(fpath) or "root"

        parts: list[str] = []

        # File role description
        basename = os.path.basename(fpath).lower()
        if basename.startswith("test_"):
            parts.append("Test file with assertion patterns")
        elif "model" in basename or "schema" in basename:
            parts.append(f"Data model/schema definition")
        elif "service" in basename or "handler" in basename:
            parts.append(f"Service/business logic")
        elif "util" in basename or "helper" in basename:
            parts.append(f"Utility/helper module")
        elif "main" in basename or "app" in basename or "index" in basename or "cli" in basename:
            parts.append(f"Entry point/application bootstrap")
        elif "config" in basename or "settings" in basename:
            parts.append(f"Configuration module")
        elif basename.startswith("readme"):
            parts.append(f"Project documentation")
        elif dir_name in ("src", "lib", "pkg"):
            parts.append(f"Core source module")
        elif ext in source_exts:
            parts.append(f"Source file in {dir_name}")

        # Pattern-based rationale
        pattern_phrases: dict[str, str] = {
            "class ": "class definitions",
            "def ": "function definitions",
            "async def ": "async programming",
            "async ": "async patterns",
            "await ": "async/await usage",
            "try:": "error handling",
            "except ": "exception handling",
            "import ": "module imports",
            "from ": "module imports",
            "return ": "return statements",
            "yield ": "generator patterns",
            "raise ": "exception raising",
            "with ": "context managers",
            "lambda ": "lambda expressions",
            "@": "decorators",
            "TypeVar": "generic typing",
            "Protocol": "structural subtyping",
            "dataclass": "data classes",
            "Enum": "enumeration types",
            "ABC": "abstract base classes",
            "__init__": "class initialization",
            "if __name__": "script entry point",
            "assert ": "assertions",
            "match ": "pattern matching",
            "case ": "pattern matching",
            "# TODO": "documented TODOs",
            "TODO": "documented TODOs",
        }

        found_patterns = []
        for kw, phrase in pattern_phrases.items():
            if kw in content:
                found_patterns.append(phrase)

        if found_patterns:
            # Deduplicate and pick top 3
            unique_patterns = list(dict.fromkeys(found_patterns))[:3]
            parts.append("demonstrates " + ", ".join(unique_patterns))
        else:
            parts.append("contains structured content")

        # Add quality signal
        if score >= 0.7:
            parts.append("high complexity and pattern diversity")
        elif score >= 0.4:
            parts.append("moderate complexity")

        return " — ".join(parts) + "."

    # Score and collect key files
    scored_key_files: list[tuple[float, str]] = []
    for fpath in file_list:
        if _is_generated_file(fpath):
            continue
        ext = os.path.splitext(fpath)[1].lower()
        # Only consider source and prose files for key_files
        if ext not in source_exts and ext not in prose_exts:
            # Also include config files that are skill-demonstrative
            if ext not in {".toml", ".yaml", ".yml", ".json", ".cfg"}:
                continue

        full_path = os.path.join(base, fpath)
        content = safe_read(full_path)
        if not content:
            continue

        score = _compute_file_complexity(full_path, content)
        if score > 0.0:
            scored_key_files.append((score, fpath))

    # Sort by complexity descending
    scored_key_files.sort(key=lambda x: x[0], reverse=True)

    # Build key_files list (top 15, or all with score >= 0.2)
    key_files: list[dict[str, Any]] = []
    seen_dirs: set[str] = set()
    for score, fpath in scored_key_files:
        if len(key_files) >= 15:
            break
        if score < 0.1:
            break

        d = os.path.dirname(fpath)
        # After 8 files, prefer diverse directories
        if d in seen_dirs and len(key_files) >= 8:
            continue
        seen_dirs.add(d)

        full_path = os.path.join(base, fpath)
        content = safe_read(full_path)
        raw_indicators = [kw.strip() for kw in _STRUCTURAL_KEYWORDS
                          if kw.strip() and kw in (content or "")]
        # Deduplicate and limit to top 10
        skill_indicators = list(dict.fromkeys(raw_indicators))[:10]

        rationale = _build_rationale(fpath, content or "", score, skill_indicators)

        key_files.append({
            "path": fpath,
            "rationale": rationale,
            "skill_indicators": skill_indicators,
            "complexity_score": score,
        })

    return key_files


# ── Scanner Role ──────────────────────────────────────────────────────────

class SkillScannerRole(AgentRole):
    """Scan a project directory and produce a deep artifact analysis.

    Produces:
    - ``project_scan.json``: surface-level file listing (backward-compatible)
    - ``artifact_analysis.json``: artifact type, content samples, structure, key_files

    The artifact analysis is the foundation for skill inference — downstream
    detectors use it to understand what kind of artifact they're analyzing
    and adapt their skill detection accordingly.
    """

    agent_name: str = "skill_scanner"
    max_steps: int = 15

    # ── Tools ───────────────────────────────────────────────────────────

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        return ToolRegistry.to_dicts(ToolRegistry.skill_scanner_tools())

    # ── Task prompt ─────────────────────────────────────────────────────

    def build_task(self, backend: str, project_dir: str = ".",
                   working_dir: str = ".", **context: Any) -> str:
        """Build the deep scan prompt (v2 — deeper sampling, key_files identification)."""
        # Resolve to absolute path so the agent finds the target regardless
        # of which directory it runs commands from (its CWD is working_dir).
        if os.path.isabs(project_dir):
            target_path = project_dir
        else:
            target_path = os.path.abspath(project_dir)

        common = (
            f"You are an artifact analyst. Your job is to examine a project, "
            f"article, or document and determine what kind of thing it is, "
            f"what it contains, and how it is structured.\n\n"
            f"## Target\n{target_path}\n"
            f"Working directory for writes: {working_dir}\n\n"
            f"## Phase 1 — Surface Scan\n"
            f"Run `find {target_path} -type f -not -path '*/.git/*' "
            f"-not -path '*/__pycache__/*' -not -path '*/node_modules/*' "
            f"-not -path '*/.venv/*' -not -path '*/venv/*' "
            f"-not -path '*/dist/*' -not -path '*/build/*' "
            f"-not -path '*/.tox/*' | sort | head -1000` to enumerate files.\n"
            f"Run `ls -laR {target_path} | head -500` for the directory tree.\n"
            f"Categorize files and write `project_scan.json`.\n\n"
            f"## Phase 2 — Artifact Classification\n"
            f"Read key files to understand the artifact's nature. "
            f"Classify it as one of:\n"
            f"- software_project: source code, tests, build configs\n"
            f"- research_paper: LaTeX, PDF, citations, academic structure\n"
            f"- blog_article: markdown posts, frontmatter, web content\n"
            f"- documentation: structured docs, API references, guides\n"
            f"- dataset: data files, schemas, processing scripts\n"
            f"- design_document: architecture specs, proposals, diagrams\n"
            f"- presentation: slides, talk materials\n"
            f"- configuration: dotfiles, infra-as-code, settings\n"
            f"- unknown: doesn't match any clear pattern\n\n"
            f"## Phase 3 — Deep Reading (Improved v2)\n"
            f"Read 8-12 representative files to understand content.\n"
            f"**Sampling priority — read skill-demonstrative files FIRST:**\n"
            f"- For software projects: prioritize complex source files "
            f"(files with many functions, classes, type annotations, "
            f"error handling, async code, design patterns) over README/config.\n"
            f"- For articles/papers: prioritize body paragraphs and "
            f"substantive content sections over headers/frontmatter.\n"
            f"- For documentation: prioritize API reference and guide "
            f"content over index/landing pages.\n"
            f"- README, config files, and build files are SECONDARY — "
            f"sample them only after skill-demonstrative files.\n"
            f"**Sample ~4000 chars per file** (increased from ~2000) — "
            f"enough for deeper context understanding.\n"
            f"**Avoid sampling** generated/minified files, lock files, "
            f"vendor code, and compiled artifacts.\n"
            f"Read from at least 4 different directories for diversity.\n\n"
            f"## Phase 4 — Structure Analysis\n"
            f"Map the artifact's organization:\n"
            f"- Top-level components (directories, sections, chapters)\n"
            f"- How components relate to each other\n"
            f"- Key entry points (main files, index, abstract)\n\n"
            f"## Phase 5 — Key Files Identification (v2)\n"
            f"After deep reading, identify the **10-15 most "
            f"skill-demonstrative files**. These are files that BEST "
            f"showcase the artifact's skills, complexity, and craftsmanship.\n"
            f"For each key file, provide:\n"
            f"- `path`: relative file path\n"
            f"- `rationale`: 1-2 sentence explanation of WHY this file "
            f"demonstrates skills (e.g., 'Contains complex class hierarchy "
            f"with type annotations and error handling — demonstrates "
            f"Type System Proficiency and Design Patterns')\n"
            f"- `skill_indicators`: list of structural keywords/patterns "
            f"found (class, def, async, try/except, decorators, generics, etc.)\n"
            f"- `complexity_score`: 0.0-1.0 estimate based on keyword "
            f"density, file size, and pattern diversity\n\n"
            f"## Output\n"
            f"Write `artifact_analysis.json` with this schema:\n"
            f"```json\n"
            f"{{\n"
            f'  "project_dir": "{target_path}",\n'
            f'  "artifact_type": {{\n'
            f'    "type": "software_project|research_paper|blog_article|...",\n'
            f'    "confidence": "high|medium|low",\n'
            f'    "description": "..."\n'
            f'  }},\n'
            f'  "surface_scan": {{ /* summary from project_scan.json */ }},\n'
            f'  "content_samples": {{\n'
            f'    "path/to/file.py": "content preview...",\n'
            f'    ...\n'
            f'  }},\n'
            f'  "structure": {{\n'
            f'    "top_level_components": ["src", "tests", "docs"],\n'
            f'    "sub_components": [\n'
            f'      {{"path": "src/models", "file_count": 5, "description": "..."}}\n'
            f'    ]\n'
            f'  }},\n'
            f'  "key_files": [\n'
            f'    {{\n'
            f'      "path": "src/models/user.py",\n'
            f'      "rationale": "Complex class hierarchy with type annotations and error handling — demonstrates Type System Proficiency and Design Patterns",\n'
            f'      "skill_indicators": ["class", "TypeVar", "except", "@dataclass"],\n'
            f'      "complexity_score": 0.85\n'
            f'    }},\n'
            f'    ...\n'
            f'  ],\n'
            f'  "metadata": {{\n'
            f'    "total_files": <int>,\n'
            f'    "languages_detected": ["Python", "JavaScript"],\n'
            f'    "has_tests": true|false,\n'
            f'    "has_docs": true|false,\n'
            f'    "has_build_config": true|false\n'
            f'  }},\n'
            f'  "scan_timestamp": "<ISO timestamp>"\n'
            f"}}\n"
            f"```\n\n"
            f"IMPORTANT: Read actual files before classifying — don't guess "
            f"from filenames alone. Write both project_scan.json AND "
            f"artifact_analysis.json, then output TASK_COMPLETE."
        )

        if backend == "claude_cli":
            backend_note = (
                "\n\nUse your own knowledge — do NOT search the web. "
                "Read files, classify the artifact, write both JSON files. "
                "Output TASK_COMPLETE when done."
            )
        else:
            backend_note = (
                "\n\nRead files, classify the artifact, write both JSON files. "
                "Output TASK_COMPLETE when done."
            )

        return common + backend_note

    # ── Parse result ────────────────────────────────────────────────────

    def parse_result(self, result: AgentResult, working_dir: str,
                     project_dir: str = ".", **context: Any) -> dict[str, Any]:
        """Extract artifact analysis from agent response or disk, with fallback."""
        analysis: dict[str, Any] = {}

        # 1. Try extracting from agent response
        for msg in reversed(result.messages):
            content = msg.content if hasattr(msg, "content") else str(msg)
            json_str = extract_json_object(content)
            if json_str:
                try:
                    parsed = json.loads(json_str)
                    if "artifact_type" in parsed or "content_samples" in parsed:
                        analysis = parsed
                        break
                except json.JSONDecodeError:
                    continue

        # 2. Try reading artifact_analysis.json from disk
        if not analysis:
            path = os.path.join(working_dir, "artifact_analysis.json")
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        parsed = json.load(f)
                    if isinstance(parsed, dict) and "artifact_type" in parsed:
                        analysis = parsed
                except (json.JSONDecodeError, OSError):
                    pass

        # 3. Fallback: run deterministic scan + classification
        if not analysis:
            analysis = self._fallback_deep_scanner(project_dir, working_dir)
            # Write to disk so downstream agents can read it
            path = os.path.join(working_dir, "artifact_analysis.json")
            try:
                with open(path, "w") as f:
                    json.dump(analysis, f, indent=2, default=str)
            except OSError:
                pass

        return analysis

    # ── Fallback: deterministic deep scan ───────────────────────────────

    @staticmethod
    def _fallback_deep_scanner(project_dir: str = ".",
                                working_dir: str = ".") -> dict[str, Any]:
        """Run a full artifact analysis without LLM.

        Uses find/ls for surface scan, file reading for content sampling,
        pattern matching for artifact classification, and complexity
        scoring for key file identification.
        """
        from datetime import datetime, timezone

        # Run existing surface scan first (writes project_scan.json)
        surface = SkillScannerRole._fallback_surface_scan(project_dir, working_dir)

        # Write project_scan.json for backward compatibility
        scan_path = os.path.join(working_dir, "project_scan.json")
        try:
            with open(scan_path, "w") as f:
                json.dump(surface, f, indent=2, default=str)
        except OSError:
            pass

        file_list: list[str] = []
        for cat_files in surface.get("file_categories", {}).values():
            file_list.extend(cat_files)

        if not file_list:
            # Try to get files directly if categories are empty
            file_list = surface.get("_raw_file_list", [])

        # Content sampling (v2: deeper, skill-demonstrative-first)
        content_samples = _sample_content(project_dir, working_dir, file_list)

        # Structure analysis
        structure = _analyze_structure(file_list, project_dir)

        # Artifact classification
        artifact_type = _classify_artifact(file_list, content_samples, project_dir)

        # Key files identification (v2)
        key_files = _identify_key_files(file_list, content_samples, project_dir)

        # Metadata
        languages: set[str] = set()
        ext_to_lang: dict[str, str] = {
            ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
            ".jsx": "JavaScript", ".tsx": "TypeScript", ".go": "Go",
            ".rs": "Rust", ".java": "Java", ".c": "C", ".cpp": "C++",
            ".rb": "Ruby", ".swift": "Swift", ".kt": "Kotlin",
            ".scala": "Scala", ".php": "PHP", ".cs": "C#",
            ".tex": "LaTeX", ".md": "Markdown", ".rst": "reStructuredText",
        }
        for f in file_list:
            ext = os.path.splitext(f)[1].lower()
            lang = ext_to_lang.get(ext)
            if lang:
                languages.add(lang)

        has_tests = len(surface.get("file_categories", {}).get("test", [])) > 0
        has_docs = len(surface.get("file_categories", {}).get("docs", [])) > 0
        has_build = (
            len(surface.get("file_categories", {}).get("build", [])) > 0 or
            len(surface.get("file_categories", {}).get("ci", [])) > 0
        )

        return {
            "project_dir": project_dir,
            "artifact_type": artifact_type,
            "surface_scan": {
                "total_files": surface.get("total_files", 0),
                "total_dirs": surface.get("total_dirs", 0),
                "file_categories": surface.get("file_categories", {}),
                "top_level_dirs": surface.get("top_level_dirs", []),
            },
            "content_samples": content_samples,
            "structure": structure,
            "key_files": key_files,
            "metadata": {
                "total_files": surface.get("total_files", 0),
                "languages_detected": sorted(languages),
                "has_tests": has_tests,
                "has_docs": has_docs,
                "has_build_config": has_build,
            },
            "scan_timestamp": datetime.now(timezone.utc).isoformat(),
            "_fallback": True,
        }

    # ── Preserved: surface-level scan (backward-compatible) ─────────────

    @staticmethod
    def _fallback_surface_scan(project_dir: str = ".",
                                working_dir: str = ".") -> dict[str, Any]:
        """Run find/ls for surface-level file enumeration. Same as v1."""
        from datetime import datetime, timezone

        # project_dir is relative to cwd, not working_dir
        if os.path.isabs(project_dir):
            scan_dir = project_dir
        else:
            scan_dir = os.path.abspath(project_dir)
        find_cmd = (
            f"find '{scan_dir}' -type f "
            f"-not -path '*/.git/*' "
            f"-not -path '*/__pycache__/*' "
            f"-not -path '*/node_modules/*' "
            f"-not -path '*/.venv/*' "
            f"-not -path '*/venv/*' "
            f"-not -path '*/dist/*' "
            f"-not -path '*/build/*' "
            f"-not -path '*/.tox/*' "
            f"| sort | head -1000"
        )
        try:
            find_result = subprocess.run(
                find_cmd, shell=True, capture_output=True, text=True,
                timeout=30,
            )
            file_list = [
                f.strip() for f in find_result.stdout.strip().split("\n")
                if f.strip()
            ]
        except (subprocess.TimeoutExpired, OSError):
            file_list = []

        ls_cmd = f"ls -laR '{scan_dir}' | head -500"
        try:
            ls_result = subprocess.run(
                ls_cmd, shell=True, capture_output=True, text=True,
                timeout=30,
            )
            dir_tree = ls_result.stdout.strip()
        except (subprocess.TimeoutExpired, OSError):
            dir_tree = ""

        top_level_dirs: list[str] = []
        if os.path.isdir(scan_dir):
            try:
                entries = os.listdir(scan_dir)
                top_level_dirs = sorted(
                    e for e in entries if os.path.isdir(os.path.join(scan_dir, e))
                )
            except OSError:
                pass

        # Categorize files
        categories: dict[str, list[str]] = {
            "source": [], "config": [], "docs": [],
            "test": [], "build": [], "ci": [], "other": [],
        }

        source_exts = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs",
                       ".java", ".c", ".cpp", ".h", ".hpp", ".rb", ".php",
                       ".swift", ".kt", ".scala", ".cs", ".vb", ".fs"}
        config_exts = {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg",
                       ".conf", ".env", ".properties", ".xml"}
        doc_exts = {".md", ".rst", ".txt", ".adoc", ".tex"}

        for fpath in file_list:
            basename = os.path.basename(fpath)
            ext = os.path.splitext(fpath)[1].lower()

            is_test = (
                basename.startswith("test_") or
                basename.endswith("_test.py") or
                basename.endswith("_test.js") or
                basename.endswith("_test.ts") or
                ".test." in basename or
                basename.startswith("spec.") or
                "__tests__" in fpath or
                "/test/" in fpath or "/tests/" in fpath or "/spec/" in fpath
            )
            if is_test:
                categories["test"].append(fpath)
                continue

            if any(ci in fpath for ci in [
                ".github/workflows", ".gitlab-ci.yml", "Jenkinsfile",
                ".travis.yml", ".circleci", "azure-pipelines",
                "bitbucket-pipelines",
            ]):
                categories["ci"].append(fpath)
                continue

            if basename in ("Dockerfile", "Makefile", "CMakeLists.txt",
                            "Rakefile", "GNUmakefile") or \
               basename.startswith("Dockerfile") or \
               basename.startswith("docker-compose"):
                categories["build"].append(fpath)
                continue

            if ext in doc_exts or basename.lower().startswith((
                    "readme", "changelog", "contributing", "license",
                    "authors", "code_of_conduct", "security", "governance")):
                categories["docs"].append(fpath)
                continue

            if ext in config_exts or basename in (
                    ".editorconfig", ".gitignore", ".prettierrc",
                    ".eslintrc", ".babelrc", ".npmrc"):
                categories["config"].append(fpath)
                continue

            if ext in source_exts:
                categories["source"].append(fpath)
                continue

            categories["other"].append(fpath)

        key_files: dict[str, str] = {}
        for fpath in file_list:
            bn = os.path.basename(fpath).lower()
            if bn.startswith("readme") and "readme" not in key_files:
                key_files["readme"] = fpath
            elif bn.startswith("license") and "license" not in key_files:
                key_files["license"] = fpath
            elif bn.startswith("contributing") and "contributing" not in key_files:
                key_files["contributing"] = fpath
            elif bn.startswith("changelog") and "changelog" not in key_files:
                key_files["changelog"] = fpath

        try:
            dir_count_result = subprocess.run(
                f"find '{scan_dir}' -type d -not -path '*/.git/*' "
                f"-not -path '*/__pycache__/*' -not -path '*/node_modules/*' | wc -l",
                shell=True, capture_output=True, text=True,
                timeout=15,
            )
            total_dirs = int(dir_count_result.stdout.strip() or 0)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            total_dirs = 0

        return {
            "project_dir": project_dir,
            "total_files": len(file_list),
            "total_dirs": total_dirs,
            "file_categories": categories,
            "key_files": key_files,
            "top_level_dirs": top_level_dirs,
            "directory_tree_preview": dir_tree[:2000],
            "scan_timestamp": datetime.now(timezone.utc).isoformat(),
            "_raw_file_list": file_list,
            "_fallback": True,
        }

    # Legacy entry point (called by old _scanner_node if only surface scan needed)
    @staticmethod
    def _fallback_scanner(project_dir: str = ".",
                          working_dir: str = ".") -> dict[str, Any]:
        """Legacy alias — runs surface scan only. Kept for backward compat."""
        return SkillScannerRole._fallback_surface_scan(project_dir, working_dir)
