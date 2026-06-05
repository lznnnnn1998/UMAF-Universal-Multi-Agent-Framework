"""Skill-dimension detector roles for the Skill Pipeline v2.

Four AgentRole subclasses that analyze an artifact for human skills across
universal dimensions (not language-specific domains):

- DomainExpertiseDetectorRole: what specialized knowledge is demonstrated?
- TechnicalCraftDetectorRole: how skilled is the creator at the medium?
- MethodologyDetectorRole: what tools, workflows, and processes are evident?
- RigorDetectorRole: how thorough, careful, and complete is the work?

Each detector reads ``artifact_analysis.json`` (produced by the scanner v2)
and writes a domain-specific JSON report consumed by SkillAggregatorRole.
"""

import json
import os
from typing import Any

from agent import AgentResult, AgentRole
from tools import ToolRegistry
from utils import extract_json_object, _PROFICIENCY_SCORES


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _load_artifact_analysis(working_dir: str) -> dict[str, Any] | None:
    """Load artifact_analysis.json from the working directory."""
    path = os.path.join(working_dir, "artifact_analysis.json")
    if not os.path.exists(path):
        # Fall back to project_scan.json for backward compat
        scan_path = os.path.join(working_dir, "project_scan.json")
        if os.path.exists(scan_path):
            try:
                with open(scan_path) as f:
                    scan = json.load(f)
                return {"surface_scan": scan, "artifact_type":
                        {"type": "unknown", "confidence": "low"}}
            except (json.JSONDecodeError, OSError):
                pass
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _get_all_files(analysis: dict[str, Any]) -> list[str]:
    """Extract the full file list from an artifact analysis."""
    surface = analysis.get("surface_scan", {})
    cats = surface.get("file_categories", {})
    files: list[str] = []
    for cat_files in cats.values():
        if isinstance(cat_files, list):
            files.extend(cat_files)
    return files


def _get_content_text(analysis: dict[str, Any]) -> str:
    """Concatenate all content samples into a single text for analysis."""
    samples = analysis.get("content_samples", {})
    return "\n\n".join(str(v) for v in samples.values())


def _get_artifact_type(analysis: dict[str, Any]) -> str:
    """Get the artifact type string."""
    at = analysis.get("artifact_type", {})
    return at.get("type", "unknown")


# ═══════════════════════════════════════════════════════════════════════════
# Base detector
# ═══════════════════════════════════════════════════════════════════════════

class _BaseDetectorRole(AgentRole):
    """Shared base for the four skill-dimension detectors."""

    max_steps: int = 12
    output_file: str = ""
    domain: str = ""

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        return ToolRegistry.to_dicts(ToolRegistry.skill_detector_tools())

    def parse_result(self, result: AgentResult, working_dir: str,
                     project_dir: str = ".", **context: Any) -> dict[str, Any]:
        """Extract domain report: agent messages → disk file → fallback.

        Validates that values are actual results, not copies of the prompt
        template (which uses ``<PLACEHOLDER>`` markers and pipe-delimited
        example values).
        """
        report: dict[str, Any] = {}

        for msg in reversed(result.messages):
            content = msg.content if hasattr(msg, "content") else str(msg)
            json_str = extract_json_object(content)
            if json_str:
                try:
                    parsed = json.loads(json_str)
                    if "domain" in parsed or "skills" in parsed or \
                       "inferred_skills" in parsed:
                        if self._is_valid_report(parsed):
                            report = parsed
                            break
                except json.JSONDecodeError:
                    continue

        # Second pass: scan Write tool-call parameters for embedded JSON reports.
        # When using claude_cli backend, the LLM writes output via the Write tool
        # rather than emitting raw JSON in its response. The report JSON is
        # embedded as an escaped string in the tool-call "content" parameter.
        if not report and self.output_file:
            for msg in reversed(result.messages):
                content = msg.content if hasattr(msg, "content") else str(msg)
                if "Write" not in content or self.output_file not in content:
                    continue
                json_str = extract_json_object(content)
                if not json_str:
                    continue
                try:
                    params = json.loads(json_str)
                except json.JSONDecodeError:
                    continue
                # Check if this is a Write call targeting our output file
                fp = params.get("file_path", "")
                inner = params.get("content", "")
                if not isinstance(inner, str) or not inner.strip():
                    continue
                if os.path.basename(fp) != self.output_file:
                    continue
                try:
                    parsed = json.loads(inner)
                    if isinstance(parsed, dict) and self._is_valid_report(parsed):
                        report = parsed
                        break
                except json.JSONDecodeError:
                    continue

        if not report and self.output_file:
            path = os.path.join(working_dir, self.output_file)
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        parsed = json.load(f)
                    if isinstance(parsed, dict) and self._is_valid_report(parsed):
                        report = parsed
                except (json.JSONDecodeError, OSError):
                    pass

        if not report:
            report = self._fallback_detect(project_dir, working_dir)

        return report

    @staticmethod
    def _is_valid_report(report: dict[str, Any]) -> bool:
        """Reject reports that are copies of the prompt template.

        Returns False only when values are clearly template placeholders
        (pipe-delimited hints, angle brackets, or literal ``...``).
        Legitimate skill/tool names like "Testing Strategy" or "Git"
        are NOT rejected — only copies of the schema itself.
        """
        for skill in report.get("inferred_skills", []):
            prof = skill.get("proficiency", "")
            conf = skill.get("confidence", "")
            name = skill.get("name", "")
            evidence = skill.get("evidence", {})
            # Pipe-delimited schema hint = template copy (e.g. "advanced|intermediate|beginner|expert")
            if prof and "|" in str(prof):
                return False
            if conf and "|" in str(conf):
                return False
            # Angle-bracket placeholder (e.g. "<SKILL_NAME>")
            if name and "<" in name:
                return False
            # Literal "..." evidence with no real data
            ev_str = json.dumps(evidence) if isinstance(evidence, dict) else str(evidence)
            if ev_str in ('{"description": "..."}', '{"indicators_matched": "..."}',
                          '{"indicators": ["..."]}'):
                return False
        for tool in report.get("detected_tools", []):
            tname = tool.get("name", "")
            tprof = tool.get("proficiency", "")
            tev = tool.get("evidence", [])
            # Angle-bracket placeholder in tool name
            if tname and "<" in tname:
                if tprof and "|" in str(tprof):
                    return False
            # Pipe-delimited proficiency in tool
            if tprof and "|" in str(tprof):
                return False
            # Literal "..." in tool evidence (but not real ellipsis in text)
            if tev and len(tev) == 1 and str(tev[0]).strip() == "...":
                return False
        return True

    def _fallback_detect(self, project_dir: str,
                         working_dir: str) -> dict[str, Any]:
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
# Detector 1 — Domain Expertise
# ═══════════════════════════════════════════════════════════════════════════

# Domain signal words — specialized terminology that indicates deep knowledge
_DOMAIN_SIGNALS: dict[str, list[str]] = {
    "Machine Learning": [
        "neural network", "gradient descent", "backpropagation",
        "transformer", "attention mechanism", "embedding", "fine-tun",
        "cross-validation", "overfitting", "regularization", "dropout",
        "batch normalization", "learning rate", "loss function",
        "BERT", "GPT", "LLaMA", "Diffusion", "reinforcement learning",
    ],
    "Distributed Systems": [
        "consensus", "Paxos", "Raft", "distributed lock", "sharding",
        "replication", "CAP theorem", "eventual consistency",
        "vector clock", "gossip protocol", "leader election",
        "quorum", "two-phase commit", "Saga pattern",
    ],
    "Security": [
        "XSS", "CSRF", "SQL injection", "zero-day", "penetration test",
        "threat model", "OWASP", "cryptographic", "public key",
        "certificate pinning", "sandbox", "privilege escalation",
    ],
    "Compiler Design": [
        "lexer", "parser", "AST", "abstract syntax tree", "LLVM",
        "intermediate representation", "code generation", "type checker",
        "garbage collector", "register allocation", "SSA form",
    ],
    "Database Systems": [
        "B-tree", "LSM tree", "write-ahead log", "MVCC",
        "query optimizer", "index scan", "transaction isolation",
        "serializable", "deadlock detection", "connection pool",
    ],
    "Game Development": [
        "game loop", "entity component system", "collision detection",
        "physics engine", "ray tracing", "shader", "frame buffer",
        "sprite", "tilemap", "pathfinding", "A* algorithm",
    ],
    "Finance": [
        "portfolio optimization", "risk parity", "Black-Scholes",
        "Monte Carlo simulation", "VaR", "derivative pricing",
        "quantitative", "alpha", "backtesting", "order book",
    ],
    "Scientific Computing": [
        "partial differential equation", "finite element",
        "numerical integration", "Monte Carlo", "computational fluid",
        "molecular dynamics", "quantum", "eigenvalue", "sparse matrix",
    ],
    "Natural Language Processing": [
        "tokenization", "named entity recognition", "part-of-speech",
        "dependency parsing", "semantic role", "coreference resolution",
        "text summarization", "machine translation", "BLEU score",
    ],
}


def _detect_domain_expertise(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Deterministic domain expertise detection from content signals."""
    all_text = _get_content_text(analysis).lower()
    all_files = _get_all_files(analysis)
    all_files_str = " ".join(all_files).lower()
    combined = all_text + " " + all_files_str

    skills: list[dict[str, Any]] = []
    for domain, signals in _DOMAIN_SIGNALS.items():
        matches = [s for s in signals if s.lower() in combined]
        if matches:
            prof = "advanced" if len(matches) >= 5 else \
                   "intermediate" if len(matches) >= 2 else "beginner"
            skills.append({
                "name": domain,
                "proficiency": prof,
                "confidence": "high" if len(matches) >= 4 else
                              "medium" if len(matches) >= 2 else "low",
                "evidence": {"signal_matches": matches[:8]},
            })

    # Sort by proficiency score
    skills.sort(key=lambda s: (_PROFICIENCY_SCORES.get(s["proficiency"], 0), len(
        s.get("evidence", {}).get("signal_matches", []))), reverse=True)

    return skills[:10]


class DomainExpertiseDetectorRole(_BaseDetectorRole):
    """Detect specialized domain knowledge demonstrated in the artifact.

    Asks: "What does the creator know deeply about?"
    """

    agent_name: str = "domain_expertise_detector"
    output_file: str = "domain_expertise_report.json"
    domain: str = "Domain Expertise"

    def build_task(self, backend: str, working_dir: str = ".",
                   project_dir: str = ".",
                   artifact_analysis: dict[str, Any] | None = None,
                   **context: Any) -> str:
        """Build the domain expertise detection prompt."""
        analysis = artifact_analysis or _load_artifact_analysis(working_dir)
        artifact_type = _get_artifact_type(analysis or {})
        content_text = _get_content_text(analysis or {})[:4000]
        files_preview = "\n".join(_get_all_files(analysis or {})[:30])

        common = (
            f"You are a domain expertise detector. Your job is to read an "
            f"artifact and determine what specialized knowledge its creator "
            f"demonstrates.\n\n"
            f"## Artifact\n"
            f"**Type**: {artifact_type}\n"
            f"Working directory: {working_dir}\n\n"
            f"### Files\n{files_preview}\n\n"
            f"### Content Samples\n{content_text[:4000]}\n\n"
            f"## Task\n"
            f"1. Read `artifact_analysis.json` from {working_dir} for full details.\n"
            f"2. Identify the specialized domains the creator demonstrates "
            f"expertise in. For each domain, provide evidence from the "
            f"artifact content.\n"
            f"3. For software: look for ML, security, compilers, distributed "
            f"systems, databases, game dev, etc.\n"
            f"4. For articles/papers: look for subject matter expertise "
            f"(economics, biology, history, philosophy, etc.).\n"
            f"5. For each skill, assess proficiency and confidence.\n\n"
            f"## Output: `{self.output_file}`\n"
            f"```json\n"
            f"{{\n"
            f'  "domain": "{self.domain}",\n'
            f'  "inferred_skills": [\n'
            f'    {{\n'
            f'      "name": "<SKILL_NAME>",\n'
            f'      "proficiency": "beginner|intermediate|advanced|expert",\n'
            f'      "confidence": "low|medium|high",\n'
            f'      "evidence": {{\n'
            f'        "signal_matches": ["<SPECIFIC_TERM_FOUND_IN_ARTIFACT>"],\n'
            f'        "context": "<WHERE_AND_HOW_THIS_SKILL_APPEARS>"\n'
            f'      }}\n'
            f'    }}\n'
            f'  ],\n'
            f'  "detected_tools": [\n'
            f'    {{\"name\": \"<TOOL_NAME>\", \"category\": \"<CATEGORY>\",\n'
            f'     \"proficiency\": \"<PROFICIENCY>\"}}\n'
            f'  ]\n'
            f"}}\n"
            f"```\n\n"
            f"IMPORTANT: Replace ALL <PLACEHOLDER> values with ACTUAL findings "
            f"from the artifact. Do NOT copy placeholder text. If you find "
            f"nothing, use empty arrays []. Output TASK_COMPLETE when done."
        )

        if backend == "claude_cli":
            backend_note = (
                "\n\nUse your own knowledge — do NOT search the web. "
                "Read artifact_analysis.json, detect domain expertise, "
                f"write {self.output_file}. Output TASK_COMPLETE."
            )
        else:
            backend_note = (
                f"\n\nRead artifact_analysis.json, detect domain expertise, "
                f"write {self.output_file}, output TASK_COMPLETE."
            )

        return common + backend_note

    def _fallback_detect(self, project_dir: str = ".",
                         working_dir: str = ".") -> dict[str, Any]:
        analysis = _load_artifact_analysis(working_dir) or {}
        skills = _detect_domain_expertise(analysis)

        # Also detect tools from file extensions
        all_files = _get_all_files(analysis)
        tools = _detect_common_tools(all_files)

        return {
            "domain": self.domain,
            "inferred_skills": skills,
            "detected_tools": tools,
            "_fallback": True,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Detector 2 — Technical Craft
# ═══════════════════════════════════════════════════════════════════════════

# Code quality heuristics (language-agnostic)
def _detect_code_craft(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Deterministic code craft detection from source content."""
    all_text = _get_content_text(analysis)
    all_files = _get_all_files(analysis)
    artifact_type = _get_artifact_type(analysis)

    skills: list[dict[str, Any]] = []

    if artifact_type not in ("software_project",):
        # For non-code artifacts, detect writing/creation craft
        return _detect_writing_craft(all_text)

    # Code craft signals
    signals: dict[str, dict[str, Any]] = {
        "Design Patterns": {
            "indicators": ["class ", "factory", "strategy", "observer",
                          "decorator", "singleton", "builder", "adapter",
                          "dependency injection", "abstract class"],
            "threshold_advanced": 4, "threshold_intermediate": 1,
        },
        "Error Handling Maturity": {
            "indicators": ["except ", "raise ", "try:", "finally:",
                          "error", "logging.", "logger.", "retry",
                          "fallback", "circuit breaker"],
            "threshold_advanced": 5, "threshold_intermediate": 2,
            "penalty_indicators": ["except Exception", "except:", "pass"],
        },
        "Type System Proficiency": {
            "indicators": ["-> ", ": ", "type ", "TypeVar", "Generic[",
                          "Protocol", "TypedDict", "dataclass", "@overload",
                          "Union[", "Optional[", "| None", "interface "],
            "threshold_advanced": 5, "threshold_intermediate": 2,
        },
        "Code Organization": {
            "indicators": ["from ", "import ", "export ", "module",
                          "__init__", "package", "namespace", "class ",
                          "__all__"],
            "threshold_advanced": 6, "threshold_intermediate": 2,
        },
        "Performance Awareness": {
            "indicators": ["cache", "lazy", "async ", "await ", "thread",
                          "process pool", "connection pool", "batch",
                          "pipeline", "profiler", "benchmark"],
            "threshold_advanced": 3, "threshold_intermediate": 1,
        },
        "Security Awareness": {
            "indicators": ["validate", "sanitize", "escape", "hash",
                          "encrypt", "decrypt", "auth", "token", "oauth",
                          "csrf", "xss", "sql injection", "rate limit"],
            "threshold_advanced": 3, "threshold_intermediate": 1,
        },
    }

    text_lower = all_text.lower()
    for skill_name, sig in signals.items():
        indicators = sig["indicators"]
        matches = [ind for ind in indicators if ind.lower() in text_lower]
        match_count = len(matches)

        # Penalties reduce proficiency
        penalty = 0
        if "penalty_indicators" in sig:
            penalty = sum(1 for p in sig["penalty_indicators"]
                         if p.lower() in text_lower)

        adjusted = match_count - penalty
        prof = "expert" if adjusted >= sig["threshold_advanced"] + 3 else \
               "advanced" if adjusted >= sig["threshold_advanced"] else \
               "intermediate" if adjusted >= sig["threshold_intermediate"] else \
               "beginner"

        if match_count > 0:
            skills.append({
                "name": skill_name,
                "proficiency": prof,
                "confidence": "high" if match_count >= 5 else
                              "medium" if match_count >= 2 else "low",
                "evidence": {"indicators_matched": matches[:8]},
            })

    prof_order = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    skills.sort(key=lambda s: prof_order.get(s["proficiency"], 0), reverse=True)
    return skills


def _detect_writing_craft(all_text: str) -> list[dict[str, Any]]:
    """Detect writing/communication craft (non-code artifacts)."""
    text = all_text.lower()
    skills: list[dict[str, Any]] = []

    craft_signals: dict[str, dict[str, Any]] = {
        "Argumentation": {
            "indicators": ["therefore", "however", "moreover", "consequently",
                          "in contrast", "on the other hand", "this suggests",
                          "evidence", "counter", "argue", "claim"],
            "threshold_advanced": 5, "threshold_intermediate": 2,
        },
        "Technical Writing": {
            "indicators": ["## ", "### ", "```", "table", "figure",
                          "diagram", "example", "note:", "warning:",
                          "appendix", "reference", "citation"],
            "threshold_advanced": 5, "threshold_intermediate": 2,
        },
        "Narrative Structure": {
            "indicators": ["introduction", "conclusion", "summary",
                          "background", "context", "overview", "in this",
                          "chapter", "section", "part i"],
            "threshold_advanced": 4, "threshold_intermediate": 2,
        },
        "Clarity": {
            "indicators": ["for example", "in other words", "specifically",
                          "namely", "that is", "to clarify", "in short",
                          "simply put", "this means"],
            "threshold_advanced": 4, "threshold_intermediate": 1,
        },
    }

    for skill_name, sig in craft_signals.items():
        matches = [ind for ind in sig["indicators"] if ind.lower() in text]
        match_count = len(matches)
        prof = "advanced" if match_count >= sig["threshold_advanced"] else \
               "intermediate" if match_count >= sig["threshold_intermediate"] else \
               "beginner"
        if match_count > 0:
            skills.append({
                "name": skill_name,
                "proficiency": prof,
                "confidence": "high" if match_count >= 5 else
                              "medium" if match_count >= 2 else "low",
                "evidence": {"indicators_matched": matches[:8]},
            })

    return skills


class TechnicalCraftDetectorRole(_BaseDetectorRole):
    """Detect technical creation skills demonstrated in the artifact.

    Asks: "How skilled is the creator at the medium itself?"
    """

    agent_name: str = "technical_craft_detector"
    output_file: str = "technical_craft_report.json"
    domain: str = "Technical Craft"

    def build_task(self, backend: str, working_dir: str = ".",
                   project_dir: str = ".",
                   artifact_analysis: dict[str, Any] | None = None,
                   **context: Any) -> str:
        """Build the technical craft detection prompt."""
        analysis = artifact_analysis or _load_artifact_analysis(working_dir)
        artifact_type = _get_artifact_type(analysis or {})
        content_text = _get_content_text(analysis or {})[:4000]
        files_preview = "\n".join(_get_all_files(analysis or {})[:30])

        common = (
            f"You are a technical craft evaluator. Your job is to assess how "
            f"skillfully the creator handled the medium itself.\n\n"
            f"## Artifact\n"
            f"**Type**: {artifact_type}\n"
            f"Working directory: {working_dir}\n\n"
            f"### Files\n{files_preview}\n\n"
            f"### Content Samples\n{content_text[:4000]}\n\n"
            f"## Task\n"
            f"1. Read `artifact_analysis.json` from {working_dir}.\n"
            f"2. Evaluate the creator's skill at the medium:\n"
            f"   - **For software**: design patterns, error handling, type "
            f"system usage, code organization, performance awareness, security.\n"
            f"   - **For articles/papers**: argumentation quality, technical "
            f"writing, narrative structure, clarity.\n"
            f"   - **For presentations**: visual design, pacing, storytelling.\n"
            f"3. Assess proficiency based on depth, consistency, and "
            f"sophistication of usage.\n\n"
            f"## Output: `{self.output_file}`\n"
            f"```json\n"
            f"{{\n"
            f'  "domain": "{self.domain}",\n'
            f'  "inferred_skills": [\n'
            f'    {{\n'
            f'      "name": "<SKILL_NAME>",\n'
            f'      "proficiency": "beginner|intermediate|advanced|expert",\n'
            f'      "confidence": "low|medium|high",\n'
            f'      "evidence": {{\n'
            f'        "indicators_matched": ["<SPECIFIC_PATTERN_FOUND>"],\n'
            f'        "context": "<WHERE_AND_HOW>"\n'
            f'      }}\n'
            f'    }}\n'
            f'  ],\n'
            f'  "detected_tools": [\n'
            f'    {{\"name\": \"<TOOL_NAME>\", \"category\": \"<CATEGORY>\",\n'
            f'     \"proficiency\": \"<PROFICIENCY>\"}}\n'
            f'  ]\n'
            f"}}\n"
            f"```\n\n"
            f"IMPORTANT: Replace ALL <PLACEHOLDER> values with ACTUAL findings "
            f"from the artifact. If nothing is found, use empty arrays []. "
            f"Output TASK_COMPLETE when done."
        )

        if backend == "claude_cli":
            backend_note = (
                "\n\nUse your own knowledge. Read artifact_analysis.json, "
                f"evaluate craft, write {self.output_file}. Output TASK_COMPLETE."
            )
        else:
            backend_note = (
                f"\n\nRead artifact_analysis.json, evaluate craft, "
                f"write {self.output_file}, output TASK_COMPLETE."
            )

        return common + backend_note

    def _fallback_detect(self, project_dir: str = ".",
                         working_dir: str = ".") -> dict[str, Any]:
        analysis = _load_artifact_analysis(working_dir) or {}
        skills = _detect_code_craft(analysis)

        return {
            "domain": self.domain,
            "inferred_skills": skills,
            "detected_tools": [],
            "_fallback": True,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Detector 3 — Methodology & Tooling
# ═══════════════════════════════════════════════════════════════════════════

_TOOL_INDICATORS: dict[str, list[str]] = {
    # Version control
    "Git": [".git", ".gitignore", ".gitattributes"],
    "GitHub": [".github/"],
    # Languages & runtimes
    "Python": [".py", "pyproject.toml", "setup.py", "setup.cfg", "Pipfile"],
    "JavaScript": ["package.json", ".js", "node_modules"],
    "TypeScript": ["tsconfig.json", ".ts", ".tsx"],
    "Go": ["go.mod", "go.sum", ".go"],
    "Rust": ["Cargo.toml", ".rs"],
    "Java": ["pom.xml", "build.gradle", ".java"],
    # Testing
    "pytest": ["pytest", "conftest.py", "test_"],
    "Jest": ["jest.config", ".test.js", ".test.ts", "__tests__"],
    "unittest": ["unittest", "TestCase"],
    # CI/CD
    "GitHub Actions": [".github/workflows"],
    "GitLab CI": [".gitlab-ci.yml"],
    "Jenkins": ["Jenkinsfile"],
    # Containers
    "Docker": ["Dockerfile", "docker-compose"],
    "Kubernetes": [".yaml", "kustomization", "helm"],
    # Docs & writing
    "LaTeX": [".tex", ".bib"],
    "Markdown": [".md"],
    "Sphinx": ["conf.py", "index.rst"],
    "MkDocs": ["mkdocs.yml"],
    "Jupyter": [".ipynb"],
    # Data
    "Pandas": ["pandas", "DataFrame"],
    "NumPy": ["numpy", "ndarray"],
    "PyTorch": ["torch", "nn.Module"],
    "scikit-learn": ["sklearn", "fit(", "predict("],
    # Web
    "React": ["react", "jsx", "tsx"],
    "Vue": ["vue", ".vue"],
    "Django": ["manage.py", "settings.py", "urls.py"],
    "Flask": ["flask", "app.py"],
    "FastAPI": ["fastapi", "uvicorn"],
    # Build
    "Webpack": ["webpack.config"],
    "Vite": ["vite.config"],
    "ESLint": [".eslintrc", "eslint.config"],
    "Prettier": [".prettierrc"],
    # Infra
    "Terraform": [".tf"],
    "Ansible": ["ansible", "playbook"],
}


def _detect_tools(all_files: list[str], all_text: str) -> list[dict[str, Any]]:
    """Deterministic tool detection from files and content."""
    files_str = " ".join(all_files)
    combined = (files_str + " " + all_text).lower()
    tools: list[dict[str, Any]] = []

    for tool, indicators in _TOOL_INDICATORS.items():
        matches = [ind for ind in indicators if ind.lower() in combined]
        if matches:
            # Categorize
            if tool in ("Python", "JavaScript", "TypeScript", "Go", "Rust", "Java"):
                cat = "Languages & Runtimes"
            elif tool in ("pytest", "Jest", "unittest"):
                cat = "Testing"
            elif tool in ("GitHub Actions", "GitLab CI", "Jenkins"):
                cat = "CI/CD"
            elif tool in ("Docker", "Kubernetes"):
                cat = "Containers & Orchestration"
            elif tool in ("LaTeX", "Markdown", "Sphinx", "MkDocs", "Jupyter"):
                cat = "Documentation Tools"
            elif tool in ("React", "Vue", "Django", "Flask", "FastAPI"):
                cat = "Web Frameworks"
            elif tool in ("Pandas", "NumPy", "PyTorch", "scikit-learn"):
                cat = "Data Science & ML"
            elif tool in ("Terraform", "Ansible"):
                cat = "Infrastructure as Code"
            elif tool in ("Webpack", "Vite", "ESLint", "Prettier"):
                cat = "Build & Tooling"
            else:
                cat = "Other Tools"

            prof = "advanced" if len(matches) >= 3 else \
                   "intermediate" if len(matches) >= 2 else "beginner"

            tools.append({
                "name": tool, "category": cat, "proficiency": prof,
                "evidence": matches[:5],
            })

    return sorted(tools, key=lambda t: len(t.get("evidence", [])), reverse=True)


def _detect_common_tools(all_files: list[str]) -> list[dict[str, Any]]:
    """Public helper — detect tools from file list only."""
    return _detect_tools(all_files, "")


def _detect_methodology_skills(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Deterministic methodology skill detection."""
    all_files = _get_all_files(analysis)
    all_text = _get_content_text(analysis)
    all_files_str = " ".join(all_files).lower()
    combined = (all_files_str + " " + all_text).lower()

    skills: list[dict[str, Any]] = []

    method_signals: dict[str, dict[str, Any]] = {
        "Git Workflow Maturity": {
            "signals": ["feat:", "fix:", "chore:", "refactor:", "docs:",
                       "test:", "conventional commit", "branch",
                       "merge request", "pull request"],
            "file_signals": [".git/"],
        },
        "CI/CD Sophistication": {
            "signals": ["pipeline", "deploy", "stage", "job:", "workflow",
                       "ci.yml", "build-and-test", "release"],
            "file_signals": [".github/workflows", ".gitlab-ci.yml", "Jenkinsfile",
                           ".circleci", "azure-pipelines"],
        },
        "Dependency Management": {
            "signals": ["version", "lock", "pinned", "requirements",
                       "dependencies", "devDependencies"],
            "file_signals": ["pyproject.toml", "package.json", "poetry.lock",
                           "Pipfile.lock", "yarn.lock", "package-lock.json",
                           "Cargo.lock", "go.sum"],
        },
        "Environment Management": {
            "signals": ["Dockerfile", "docker-compose", "devcontainer",
                       "virtualenv", ".venv", "conda", "nix", "nvm"],
            "file_signals": ["Dockerfile", "docker-compose.yml",
                           ".devcontainer", ".python-version", ".nvmrc"],
        },
        "Incremental Development": {
            "signals": ["TODO", "FIXME", "HACK", "WIP", "v0.", "v1.",
                       "changelog", "version history", "migration"],
            "file_signals": ["CHANGELOG.md", "MIGRATION.md", "VERSION"],
        },
    }

    for skill_name, sig in method_signals.items():
        text_matches = [s for s in sig["signals"] if s.lower() in combined]
        file_matches = [s for s in sig["file_signals"] if s.lower() in all_files_str]
        total = len(text_matches) + len(file_matches) * 2  # file matches count double

        prof = "advanced" if total >= 5 else \
               "intermediate" if total >= 2 else "beginner"
        conf = "high" if total >= 5 else "medium" if total >= 3 else "low"

        if total > 0:
            skills.append({
                "name": skill_name,
                "proficiency": prof,
                "confidence": conf,
                "evidence": {
                    "text_signals": text_matches[:6],
                    "file_signals": file_matches[:6],
                },
            })

    prof_order = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    skills.sort(key=lambda s: prof_order.get(s["proficiency"], 0), reverse=True)
    return skills


class MethodologyDetectorRole(_BaseDetectorRole):
    """Detect tools, workflows, and processes used to create the artifact.

    Asks: "What tools, workflows, and processes are evident?"
    """

    agent_name: str = "methodology_detector"
    output_file: str = "methodology_report.json"
    domain: str = "Methodology & Tooling"

    def build_task(self, backend: str, working_dir: str = ".",
                   project_dir: str = ".",
                   artifact_analysis: dict[str, Any] | None = None,
                   **context: Any) -> str:
        """Build the methodology detection prompt."""
        analysis = artifact_analysis or _load_artifact_analysis(working_dir)
        artifact_type = _get_artifact_type(analysis or {})
        all_files = _get_all_files(analysis or {})
        files_preview = "\n".join(all_files[:40])

        common = (
            f"You are a methodology and tooling detector. Your job is to "
            f"identify what tools, workflows, and processes the creator used.\n\n"
            f"## Artifact\n"
            f"**Type**: {artifact_type}\n"
            f"Working directory: {working_dir}\n\n"
            f"### Files ({len(all_files)} total)\n{files_preview}\n\n"
            f"## Task\n"
            f"1. Read `artifact_analysis.json` from {working_dir}.\n"
            f"2. Detect tools from file indicators (config files, extensions).\n"
            f"3. Infer methodology skills:\n"
            f"   - Git workflow maturity (conventional commits, branching)\n"
            f"   - CI/CD sophistication (pipeline complexity)\n"
            f"   - Dependency management (lock files, version pinning)\n"
            f"   - Environment management (Docker, venv, containers)\n"
            f"   - Release management (versioning, changelogs)\n"
            f"   - Incremental development (small changes, refactoring)\n"
            f"4. For non-code artifacts, look for text-specific tools "
            f"(LaTeX, Markdown, reference managers, CMS, SEO).\n\n"
            f"## Output: `{self.output_file}`\n"
            f"```json\n"
            f"{{\n"
            f'  "domain": "{self.domain}",\n'
            f'  "detected_tools": [\n'
            f'    {{\"name\": "<TOOL_NAME>", "category": "<CATEGORY>",\n'
            f'     "proficiency": "advanced", "evidence": ["commit messages"]}}\n'
            f'  ],\n'
            f'  "inferred_skills": [\n'
            f'    {{\n'
            f'      "name": "<SKILL_NAME>",\n'
            f'      "proficiency": "advanced|intermediate|beginner|expert",\n'
            f'      "confidence": "high|medium|low",\n'
            f'      "evidence": {{"description": "..."}}\n'
            f'    }}\n'
            f'  ]\n'
            f"}}\n"
            f"```\n\n"
            f"IMPORTANT: Replace ALL <PLACEHOLDER> values with ACTUAL findings from the artifact. If nothing is found, use empty arrays []. Output TASK_COMPLETE when done."
        )

        if backend == "claude_cli":
            backend_note = (
                "\n\nUse your own knowledge. Read artifact_analysis.json, "
                f"detect tools and methodology, write {self.output_file}. "
                "Output TASK_COMPLETE."
            )
        else:
            backend_note = (
                f"\n\nRead artifact_analysis.json, detect tools and "
                f"methodology, write {self.output_file}, output TASK_COMPLETE."
            )

        return common + backend_note

    def _fallback_detect(self, project_dir: str = ".",
                         working_dir: str = ".") -> dict[str, Any]:
        analysis = _load_artifact_analysis(working_dir) or {}
        all_files = _get_all_files(analysis)
        all_text = _get_content_text(analysis)

        tools = _detect_tools(all_files, all_text)
        skills = _detect_methodology_skills(analysis)

        return {
            "domain": self.domain,
            "detected_tools": tools,
            "inferred_skills": skills,
            "_fallback": True,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Detector 4 — Depth & Rigor
# ═══════════════════════════════════════════════════════════════════════════

def _detect_rigor_skills(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Deterministic rigor/depth detection."""
    all_files = _get_all_files(analysis)
    all_text = _get_content_text(analysis)
    all_files_str = " ".join(all_files).lower()
    combined = (all_files_str + " " + all_text).lower()
    artifact_type = _get_artifact_type(analysis)

    skills: list[dict[str, Any]] = []

    # Count file types
    total = len(all_files) if all_files else 1
    test_files = [f for f in all_files if
                  "test_" in f or "_test" in f or "/test" in f or "/tests/" in f
                  or "__tests__" in f or f.endswith(".test.js")
                  or f.endswith(".test.ts") or f.endswith("_test.py")
                  or f.endswith("_test.go")]
    doc_files = [f for f in all_files if f.endswith((".md", ".rst", ".adoc"))
                 or os.path.basename(f).lower().startswith(("readme", "contributing",
                 "changelog", "license"))]

    # Testing Strategy
    test_ratio = len(test_files) / max(total, 1)
    if test_files:
        prof = "advanced" if test_ratio > 0.2 else \
               "intermediate" if test_ratio > 0.05 else "beginner"
        skills.append({
            "name": "Testing Strategy",
            "proficiency": prof,
            "confidence": "high" if test_ratio > 0.15 else
                          "medium" if test_ratio > 0.05 else "low",
            "evidence": {
                "test_count": len(test_files),
                "test_ratio": round(test_ratio, 3),
                "sample_files": test_files[:5],
            },
        })

    # Test Coverage signals in content
    coverage_signals = ["assert", "expect(", "should ", "test(", "it(",
                        "describe(", "def test_", "class Test",
                        "parameterize", "fixture", "mock", "stub",
                        "edge case", "corner case", "boundary"]
    coverage_matches = [s for s in coverage_signals if s.lower() in combined]
    if coverage_matches:
        prof = "advanced" if len(coverage_matches) >= 8 else \
               "intermediate" if len(coverage_matches) >= 4 else "beginner"
        skills.append({
            "name": "Test Coverage Thoroughness",
            "proficiency": prof,
            "confidence": "high" if len(coverage_matches) >= 8 else
                          "medium" if len(coverage_matches) >= 4 else "low",
            "evidence": {"indicators": coverage_matches[:10]},
        })

    # Documentation Quality
    doc_ratio = len(doc_files) / max(total, 1)
    doc_content_signals = ["readme", "getting started", "installation",
                          "usage", "api", "example", "tutorial", "guide",
                          "reference", "faq", "troubleshooting", "contributing"]
    doc_matches = [s for s in doc_content_signals if s in combined]
    doc_score = len(doc_matches) + len(doc_files) * 2

    if doc_files or doc_matches:
        prof = "advanced" if doc_score >= 10 else \
               "intermediate" if doc_score >= 4 else "beginner"
        skills.append({
            "name": "Documentation Quality",
            "proficiency": prof,
            "confidence": "high" if doc_score >= 8 else
                          "medium" if doc_score >= 4 else "low",
            "evidence": {
                "doc_files": doc_files[:5],
                "content_signals": doc_matches[:8],
            },
        })

    # Code Quality Enforcement
    quality_files = ["eslint", "prettier", ".ruff", "pyproject.toml",
                    "mypy", "flake8", "pylint", "pre-commit",
                    ".editorconfig", "husky", "lint-staged"]
    quality_matches = [q for q in quality_files if q in all_files_str]
    if quality_matches:
        prof = "advanced" if len(quality_matches) >= 3 else \
               "intermediate" if len(quality_matches) >= 1 else "beginner"
        skills.append({
            "name": "Code Quality Enforcement",
            "proficiency": prof,
            "confidence": "high" if len(quality_matches) >= 3 else "medium",
            "evidence": {"configs_found": quality_matches},
        })

    # For non-code: completeness and rigor signals
    if artifact_type not in ("software_project",):
        rigor_signals = ["citation", "reference", "source", "footnote",
                        "bibliography", "appendix", "methodology",
                        "limitation", "future work", "acknowledgment",
                        "data available", "code available", "reproducible"]
        rigor_matches = [r for r in rigor_signals if r in combined]
        if rigor_matches:
            prof = "advanced" if len(rigor_matches) >= 7 else \
                   "intermediate" if len(rigor_matches) >= 3 else "beginner"
            skills.append({
                "name": "Academic Rigor",
                "proficiency": prof,
                "confidence": "high" if len(rigor_matches) >= 6 else
                              "medium" if len(rigor_matches) >= 3 else "low",
                "evidence": {"rigor_signals": rigor_matches[:10]},
            })

    prof_order = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    skills.sort(key=lambda s: prof_order.get(s["proficiency"], 0), reverse=True)
    return skills


class RigorDetectorRole(_BaseDetectorRole):
    """Detect thoroughness, care, and completeness in the artifact.

    Asks: "How thorough, careful, and complete is the work?"
    """

    agent_name: str = "rigor_detector"
    output_file: str = "rigor_report.json"
    domain: str = "Depth & Rigor"

    def build_task(self, backend: str, working_dir: str = ".",
                   project_dir: str = ".",
                   artifact_analysis: dict[str, Any] | None = None,
                   **context: Any) -> str:
        """Build the depth/rigor detection prompt."""
        analysis = artifact_analysis or _load_artifact_analysis(working_dir)
        artifact_type = _get_artifact_type(analysis or {})
        content_text = _get_content_text(analysis or {})[:4000]
        files_preview = "\n".join(_get_all_files(analysis or {})[:30])
        metadata = (analysis or {}).get("metadata", {})

        has_tests = metadata.get("has_tests", False)
        has_docs = metadata.get("has_docs", False)

        common = (
            f"You are a depth and rigor evaluator. Your job is to assess how "
            f"thorough, careful, and complete the work is.\n\n"
            f"## Artifact\n"
            f"**Type**: {artifact_type}\n"
            f"**Has tests**: {has_tests}  |  **Has docs**: {has_docs}\n"
            f"Working directory: {working_dir}\n\n"
            f"### Files\n{files_preview}\n\n"
            f"### Content Samples\n{content_text[:4000]}\n\n"
            f"## Task\n"
            f"1. Read `artifact_analysis.json` from {working_dir}.\n"
            f"2. Evaluate rigor and thoroughness:\n"
            f"   - **For software**: testing strategy, test coverage, "
            f"edge case handling, documentation quality, linting/quality "
            f"enforcement, error handling patterns.\n"
            f"   - **For articles/papers**: citations, methodology section, "
            f"data availability, limitations, editing quality.\n"
            f"   - **For datasets**: schema documentation, data validation, "
            f"completeness notes, processing scripts.\n"
            f"3. What's missing? What would a more thorough version include?\n\n"
            f"## Output: `{self.output_file}`\n"
            f"```json\n"
            f"{{\n"
            f'  "domain": "{self.domain}",\n'
            f'  "inferred_skills": [\n'
            f'    {{\n'
            f'      "name": "<SKILL_NAME>",\n'
            f'      "proficiency": "advanced|intermediate|beginner|expert",\n'
            f'      "confidence": "high|medium|low",\n'
            f'      "evidence": {{\n'
            f'        "test_count": 25,\n'
            f'        "description": "Test files cover all major modules"\n'
            f'      }}\n'
            f'    }}\n'
            f'  ]\n'
            f"}}\n"
            f"```\n\n"
            f"IMPORTANT: Replace ALL <PLACEHOLDER> values with ACTUAL findings from the artifact. If nothing is found, use empty arrays []. Output TASK_COMPLETE when done."
        )

        if backend == "claude_cli":
            backend_note = (
                "\n\nUse your own knowledge. Read artifact_analysis.json, "
                f"evaluate rigor, write {self.output_file}. Output TASK_COMPLETE."
            )
        else:
            backend_note = (
                f"\n\nRead artifact_analysis.json, evaluate rigor, "
                f"write {self.output_file}, output TASK_COMPLETE."
            )

        return common + backend_note

    def _fallback_detect(self, project_dir: str = ".",
                         working_dir: str = ".") -> dict[str, Any]:
        analysis = _load_artifact_analysis(working_dir) or {}
        skills = _detect_rigor_skills(analysis)

        return {
            "domain": self.domain,
            "inferred_skills": skills,
            "detected_tools": [],
            "_fallback": True,
        }
