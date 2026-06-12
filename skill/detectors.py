"""Skill-dimension detector roles for the Skill Pipeline v2.

Four AgentRole subclasses that analyze an artifact for human skills across
universal dimensions (not language-specific domains):

- DomainExpertiseDetectorRole: what specialized knowledge is demonstrated?
- TechnicalCraftDetectorRole: how skilled is the creator at the medium?
- MethodologyDetectorRole: what tools, workflows, and processes are evident?
- RigorDetectorRole: how thorough, careful, and complete is the work?

Each detector reads ``artifact_analysis.json`` (produced by the scanner v2)
and writes a domain-specific JSON report consumed by SkillAggregatorRole.

v2 improvements:
- Evidence-based proficiency assessment replacing count-based heuristics
- Richer domain signal coverage with multi-word matching and negative signals
- Modern tool detection with version detection and project-vs-ecosystem distinction
- evidence_refs: specific file-path evidence for every detected skill entry
"""

from __future__ import annotations

import json
import os
import re
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


def _get_content_samples(analysis: dict[str, Any]) -> dict[str, str]:
    """Get content samples as a dict of file_path -> content."""
    return analysis.get("content_samples", {})


def _get_artifact_type(analysis: dict[str, Any]) -> str:
    """Get the artifact type string."""
    at = analysis.get("artifact_type", {})
    return at.get("type", "unknown")


# ═══════════════════════════════════════════════════════════════════════════
# Qualitative proficiency assessment (v2)
# ═══════════════════════════════════════════════════════════════════════════


def _assess_proficiency(
    signal_matches: list[str],
    file_distribution: dict[str, list[str]],
    *,
    integration_bonus: int = 0,
    negative_penalty: int = 0,
    total_files: int = 1,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Assess proficiency qualitatively based on depth, consistency, and integration.

    Replaces the old count-based model (``len(matches) >= N -> proficiency``) with
    a multi-dimensional assessment:

    - **Depth**: signal specificity weight (multi-word phrases, rare terms score higher)
    - **Consistency**: how many distinct files contain the signals
    - **Integration**: bonus for interrelated signals appearing together
    - **Negative penalty**: subtract for false-positive indicators

    Args:
        signal_matches: list of matched signal strings.
        file_distribution: dict mapping file_path -> list of signals found in that file.
        integration_bonus: extra score for related-signal co-occurrence.
        negative_penalty: score reduction from negative signals.
        total_files: total file count (for distribution ratio).

    Returns:
        Tuple of (proficiency, confidence, evidence_refs).
        evidence_refs is a list of dicts with ``file``, ``signals``, and ``description``.
    """
    # Depth: weight by signal specificity
    depth_score = 0.0
    for sig in signal_matches:
        words = sig.split()
        if len(words) >= 3:
            depth_score += 3.0   # multi-word phrase — very specific
        elif len(words) == 2:
            depth_score += 2.0   # two-word term — moderately specific
        else:
            depth_score += 1.0   # single word — less specific

    # Consistency: distribution across files
    files_with_signals = len(file_distribution)
    consistency_ratio = files_with_signals / max(total_files, 1)
    consistency_score = min(files_with_signals, 5) * 1.5
    if consistency_ratio > 0.3:
        consistency_score += 3.0  # pervasive
    elif consistency_ratio > 0.1:
        consistency_score += 1.5

    # Integration bonus (from co-occurring related skills)
    integration_score = float(integration_bonus) * 1.5

    # Negative penalty
    penalty = float(negative_penalty) * 2.0

    total_score = depth_score + consistency_score + integration_score - penalty

    # Map to proficiency label
    if total_score >= 14.0:
        proficiency = "expert"
    elif total_score >= 9.0:
        proficiency = "advanced"
    elif total_score >= 4.0:
        proficiency = "intermediate"
    elif total_score >= 1.0:
        proficiency = "beginner"
    else:
        proficiency = "beginner"

    # Confidence based on signal count + distribution breadth
    if len(signal_matches) >= 5 and files_with_signals >= 3:
        confidence = "high"
    elif len(signal_matches) >= 3 and files_with_signals >= 2:
        confidence = "medium"
    elif signal_matches:
        confidence = "low"
    else:
        confidence = "low"

    # Build evidence_refs
    evidence_refs: list[dict[str, Any]] = []
    for fpath, sigs in file_distribution.items():
        evidence_refs.append({
            "file": fpath,
            "signals": sigs[:5],
            "description": f"Found {len(sigs)} signal(s) in {os.path.basename(fpath)}",
        })

    return proficiency, confidence, evidence_refs


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
# Detector 1 — Domain Expertise (v2)
# ═══════════════════════════════════════════════════════════════════════════

# Domain signal words — specialized terminology that indicates deep knowledge.
# v2: expanded from 9 to 19 domains with multi-word phrase matching.
_DOMAIN_SIGNALS: dict[str, list[str]] = {
    "Machine Learning": [
        "neural network", "gradient descent", "backpropagation",
        "transformer", "attention mechanism", "embedding", "fine-tun",
        "cross-validation", "overfitting", "regularization", "dropout",
        "batch normalization", "learning rate", "loss function",
        "BERT", "GPT", "LLaMA", "Diffusion", "reinforcement learning",
        "convolutional neural network", "recurrent neural network",
        "generative adversarial network", "self-attention", "layer normalization",
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
    # ═════ v2: 10 new domains below ═════
    "Computer Vision": [
        "object detection", "image segmentation", "convolutional neural network",
        "feature extraction", "optical flow", "image classification",
        "bounding box", "semantic segmentation", "instance segmentation",
        "YOLO", "ResNet", "VGG", "ImageNet", "data augmentation",
        "transfer learning", "region proposal network", "non-maximum suppression",
    ],
    "Reinforcement Learning": [
        "Q-learning", "policy gradient", "Markov decision process",
        "reward function", "exploration exploitation", "deep Q-network",
        "actor critic", "PPO", "DDPG", "TD learning",
        "Monte Carlo tree search", "experience replay", "environment",
        "state space", "action space", "value function",
    ],
    "Networking": [
        "TCP/IP", "HTTP/2", "HTTP/3", "WebSocket", "gRPC",
        "packet", "latency", "throughput", "load balancer",
        "reverse proxy", "DNS", "TLS handshake", "CDN",
        "network protocol", "socket", "bandwidth", "routing",
        "NAT", "firewall", "VPN", "QUIC",
    ],
    "Operating Systems": [
        "kernel", "syscall", "process scheduler", "virtual memory",
        "page fault", "file system", "inode", "context switch",
        "interrupt handler", "device driver", "system call",
        "memory management unit", "TLB", "kernel panic", "deadlock",
        "race condition", "semaphore", "mutex", "spinlock",
    ],
    "Embedded Systems": [
        "microcontroller", "RTOS", "firmware", "interrupt service routine",
        "GPIO", "I2C", "SPI", "UART", "bare metal",
        "cross-compile", "toolchain", "bootloader", "watchdog timer",
        "memory-mapped I/O", "ARM Cortex", "ESP32", "Arduino",
    ],
    "DevOps": [
        "CI/CD pipeline", "infrastructure as code", "configuration management",
        "continuous deployment", "blue-green deployment", "canary release",
        "monitoring", "alerting", "incident response", "SLO",
        "service mesh", "observability", "GitOps", "immutable infrastructure",
        "Chaos engineering", "site reliability engineering",
    ],
    "Data Engineering": [
        "ETL pipeline", "data warehouse", "data lake", "Apache Spark",
        "Apache Kafka", "stream processing", "batch processing",
        "data ingestion", "data governance", "schema registry",
        "data lineage", "OLAP", "columnar storage", "Parquet",
        "data pipeline", "data mesh", "CDC", "change data capture",
    ],
    "Frontend Development": [
        "component lifecycle", "virtual DOM", "responsive design",
        "CSS Grid", "Flexbox", "state management", "client-side rendering",
        "server-side rendering", "static site generation", "hydration",
        "web components", "CSS-in-JS", "bundle splitting", "lazy loading",
        "progressive web app", "web vitals", "accessibility", "ARIA",
    ],
    "Mobile Development": [
        "iOS", "Android", "React Native", "Flutter", "SwiftUI",
        "UIKit", "Jetpack Compose", "app lifecycle", "push notification",
        "deep linking", "app store", "TestFlight", "code signing",
        "provisioning profile", "background task", "Core Data",
    ],
    "Blockchain": [
        "smart contract", "Solidity", "Ethereum", "consensus mechanism",
        "proof of work", "proof of stake", "DeFi", "NFT",
        "ERC-20", "ERC-721", "Web3", "distributed ledger",
        "gas fee", "wallet", "private key", "DAO",
        "Layer 2", "rollup", "ZK proof", "zero-knowledge proof",
    ],
}

# v2: Negative signals — terms that should NOT count toward domain expertise
# because they are common in other contexts. Each entry maps a domain name to
# patterns that, if found alone, suggest the domain match is a false positive.
_DOMAIN_NEGATIVE_SIGNALS: dict[str, list[str]] = {
    "Game Development": [
        "game theory", "game over", "game of life", "language game",
    ],
    "Operating Systems": [
        "operating system",  # generic mention, not deep OS knowledge
    ],
    "Blockchain": [
        "blockchain technology",  # buzzword usage without depth
    ],
    "Finance": [
        "financial", "financing",  # too generic
    ],
    "Frontend Development": [
        "frontend",  # alone is too generic
    ],
    "Mobile Development": [
        "mobile first", "mobile-friendly",  # responsive design terms, not mobile dev
    ],
}

# v2: Signal specificity weights — multi-word phrases are more indicative of
# deep knowledge than single-keyword matches.
_SIGNAL_SPECIFICITY: dict[str, float] = {}


def _build_signal_specificity() -> None:
    """Pre-compute specificity weights from signal word count (idempotent)."""
    if _SIGNAL_SPECIFICITY:
        return
    for domain, signals in _DOMAIN_SIGNALS.items():
        for sig in signals:
            words = sig.split()
            if len(words) >= 3:
                _SIGNAL_SPECIFICITY[sig.lower()] = 3.0
            elif len(words) == 2:
                _SIGNAL_SPECIFICITY[sig.lower()] = 2.0
            else:
                _SIGNAL_SPECIFICITY[sig.lower()] = 1.0


def _find_signal_in_content(
    signal: str, content_samples: dict[str, str]
) -> list[str]:
    """Find which files contain a given signal string (case-insensitive).

    Args:
        signal: The signal string to search for.
        content_samples: Dict of file_path -> content text.

    Returns:
        List of file paths where the signal was found.
    """
    sig_lower = signal.lower()
    found_in: list[str] = []
    for fpath, content in content_samples.items():
        if sig_lower in content.lower():
            found_in.append(fpath)
    return found_in


def _check_negative_signals(
    domain: str, content_samples: dict[str, str]
) -> int:
    """Count how many negative signals for a domain are present.

    Returns the penalty count (higher = more false-positive risk).
    """
    negative_list = _DOMAIN_NEGATIVE_SIGNALS.get(domain, [])
    penalty = 0
    all_text = " ".join(content_samples.values()).lower()
    for neg in negative_list:
        if neg.lower() in all_text:
            penalty += 1
    return penalty


def _detect_domain_expertise(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Deterministic domain expertise detection from content signals (v2).

    v2 improvements:
    - Qualitative proficiency assessment based on depth, consistency, integration
    - Multi-word phrase matching with specificity weighting
    - Negative signal filtering to reduce false positives
    - evidence_refs with specific file paths for every skill
    """
    # Initialize specificity weights (idempotent)
    _build_signal_specificity()

    all_text = _get_content_text(analysis).lower()
    all_files = _get_all_files(analysis)
    all_files_str = " ".join(all_files).lower()
    combined = all_text + " " + all_files_str
    content_samples = _get_content_samples(analysis)
    total_files = len(content_samples) if content_samples else max(len(all_files), 1)

    skills: list[dict[str, Any]] = []
    for domain, signals in _DOMAIN_SIGNALS.items():
        matches: list[str] = []

        # Match signals against combined content, preferring multi-word phrases
        # Multi-word phrases checked first to avoid double-counting
        # with single-word sub-matches
        for s in signals:
            sig_lower = s.lower()
            if sig_lower in combined:
                matches.append(s)

        if not matches:
            continue

        # Build file distribution for qualitative assessment
        file_distribution: dict[str, list[str]] = {}
        for sig in matches:
            sig_files = _find_signal_in_content(sig, content_samples)
            if not sig_files:
                # Signal found in filenames only, attribute to file list
                file_distribution.setdefault("(file names)", []).append(sig)
                continue
            for fp in sig_files:
                file_distribution.setdefault(fp, []).append(sig)

        # Check negative signals
        negative_penalty = _check_negative_signals(domain, content_samples)

        # Integration bonus: domains that share signals with related domains
        # indicate deeper, interconnected knowledge
        integration_bonus = 0
        # e.g., ML + Scientific Computing or NLP + ML
        related_pairs: dict[str, list[str]] = {
            "Machine Learning": ["Scientific Computing", "Natural Language Processing",
                                  "Computer Vision", "Reinforcement Learning"],
            "Computer Vision": ["Machine Learning", "Reinforcement Learning"],
            "Reinforcement Learning": ["Machine Learning", "Robotics"],
            "Data Engineering": ["Distributed Systems", "DevOps"],
            "DevOps": ["Networking", "Distributed Systems", "Data Engineering"],
            "Frontend Development": ["Mobile Development"],
            "Operating Systems": ["Embedded Systems", "Networking"],
        }
        for related in related_pairs.get(domain, []):
            if related.lower() in combined:
                integration_bonus += 1

        # Qualitative proficiency assessment
        proficiency, confidence, evidence_refs = _assess_proficiency(
            matches, file_distribution,
            integration_bonus=integration_bonus,
            negative_penalty=negative_penalty,
            total_files=total_files,
        )

        if not evidence_refs:
            # Minimal evidence when no per-file samples are available
            evidence_refs = [{
                "file": "(file list)",
                "signals": matches[:5],
                "description": f"Matched {len(matches)} domain signal(s) in file names and content",
            }]

        skills.append({
            "name": domain,
            "proficiency": proficiency,
            "confidence": confidence,
            "evidence": {"signal_matches": matches[:8]},
            "evidence_refs": evidence_refs,
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
            f"expertise in. For each domain, provide specific evidence "
            f"with file paths from the artifact content.\n"
            f"3. For software: look for ML, security, compilers, distributed "
            f"systems, databases, game dev, computer vision, networking, etc.\n"
            f"4. For articles/papers: look for subject matter expertise "
            f"(economics, biology, history, philosophy, etc.).\n"
            f"5. Assess proficiency qualitatively — consider depth of usage "
            f"(superficial vs sophisticated), consistency (one-off vs pervasive), "
            f"and integration (isolated vs interconnected with other skills).\n"
            f"6. Include evidence_refs: specific file paths with descriptions "
            f"of what demonstrates each skill.\n\n"
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
            f'      }},\n'
            f'      "evidence_refs": [\n'
            f'        {{"file": "<FILE_PATH>", "signals": ["<SIGNAL>"], '
            f'"description": "<WHAT_DEMONSTRATES_THE_SKILL>"}}\n'
            f'      ]\n'
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
# Detector 2 — Technical Craft (v2)
# ═══════════════════════════════════════════════════════════════════════════

def _detect_code_craft(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Deterministic code craft detection from source content (v2).

    v2 improvements:
    - Qualitative proficiency: depth of pattern usage, consistency across files,
      and integration with other craft skills
    - evidence_refs with specific file paths
    - Sophisticated vs basic indicator distinction
    """
    all_text = _get_content_text(analysis)
    all_files = _get_all_files(analysis)
    content_samples = _get_content_samples(analysis)
    artifact_type = _get_artifact_type(analysis)
    total_files = len(content_samples) if content_samples else max(len(all_files), 1)

    skills: list[dict[str, Any]] = []

    if artifact_type not in ("software_project",):
        # For non-code artifacts, detect writing/creation craft
        return _detect_writing_craft(analysis)

    # Code craft signals with sophistication tiers (v2)
    # basic indicators show awareness; advanced indicators show deep skill
    signals: dict[str, dict[str, Any]] = {
        "Design Patterns": {
            "indicators": ["class ", "factory", "strategy", "observer",
                          "decorator", "singleton", "builder", "adapter",
                          "dependency injection", "abstract class"],
            # v2: sophisticated patterns — multi-pattern compositions
            "advanced_indicators": [
                "abstract factory", "factory method", "template method",
                "chain of responsibility", "command pattern",
                "dependency injection", "inversion of control",
                "repository pattern",
            ],
            "penalty_indicators": [],  # basic 'class' alone is not a pattern
        },
        "Error Handling Maturity": {
            "indicators": ["except ", "raise ", "try:", "finally:",
                          "error", "logging.", "logger.", "retry",
                          "fallback", "circuit breaker"],
            "advanced_indicators": [
                "circuit breaker", "exponential backoff", "retry with",
                "custom exception", "exception hierarchy",
                "structured logging", "error boundary",
                "graceful degradation",
            ],
            "penalty_indicators": ["except Exception", "except:", "pass"],
        },
        "Type System Proficiency": {
            "indicators": ["-> ", ": ", "type ", "TypeVar", "Generic[",
                          "Protocol", "TypedDict", "dataclass", "@overload",
                          "Union[", "Optional[", "| None", "interface "],
            "advanced_indicators": [
                "TypeGuard", "ParamSpec", "Concatenate", "Self type",
                "variance", "covariant", "contravariant",
                "recursive type", "generic constraint",
            ],
            "penalty_indicators": [],
        },
        "Code Organization": {
            "indicators": ["from ", "import ", "export ", "module",
                          "__init__", "package", "namespace", "class ",
                          "__all__"],
            "advanced_indicators": [
                "layered architecture", "hexagonal architecture",
                "clean architecture", "domain-driven design",
                "bounded context", "aggregate root",
                "interface segregation", "dependency inversion",
            ],
            "penalty_indicators": [],
        },
        "Performance Awareness": {
            "indicators": ["cache", "lazy", "async ", "await ", "thread",
                          "process pool", "connection pool", "batch",
                          "pipeline", "profiler", "benchmark"],
            "advanced_indicators": [
                "memory profiler", "CPU profiling", "flame graph",
                "zero-copy", "SIMD", "vectorization",
                "lock-free", "wait-free", "cache coherence",
                "NUMA-aware", "JIT compilation",
            ],
            "penalty_indicators": [],
        },
        "Security Awareness": {
            "indicators": ["validate", "sanitize", "escape", "hash",
                          "encrypt", "decrypt", "auth", "token", "oauth",
                          "csrf", "xss", "sql injection", "rate limit"],
            "advanced_indicators": [
                "OWASP", "threat model", "penetration test",
                "zero trust", "principle of least privilege",
                "secure by default", "defense in depth",
                "CVE", "vulnerability disclosure",
            ],
            "penalty_indicators": [],
        },
    }

    text_lower = all_text.lower()
    content_keys_lower = {k.lower(): v.lower() for k, v in content_samples.items()}

    for skill_name, sig in signals.items():
        indicators: list[str] = sig["indicators"]
        advanced_indicators: list[str] = sig.get("advanced_indicators", [])
        penalty_indicators: list[str] = sig.get("penalty_indicators", [])

        # Match indicators and advanced indicators
        basic_matches = [ind for ind in indicators if ind.lower() in text_lower]
        adv_matches = [ind for ind in advanced_indicators if ind.lower() in text_lower]
        all_matches = basic_matches + adv_matches

        if not all_matches:
            continue

        # Build file distribution
        file_distribution: dict[str, list[str]] = {}
        for ind in all_matches:
            for fpath_key, content in content_keys_lower.items():
                if ind.lower() in content:
                    file_distribution.setdefault(fpath_key, []).append(ind)
            # If not found in any specific file, attribute to general content
            if ind not in sum(file_distribution.values(), []):
                file_distribution.setdefault("(content)", []).append(ind)

        # Penalties reduce proficiency
        penalty = sum(1 for p in penalty_indicators if p.lower() in text_lower)

        # Integration bonus: related craft skills appearing together
        integration_bonus = 0
        # Design patterns + type system = deep craft
        if skill_name == "Design Patterns" and any(
            s in text_lower for s in ["generic", "protocol", "typeguard"]
        ):
            integration_bonus += 2
        # Error handling + security = resilient code
        if skill_name == "Error Handling Maturity" and any(
            s in text_lower for s in ["validate", "sanitize", "auth"]
        ):
            integration_bonus += 1
        # Performance + code organization = well-architected perf
        if skill_name == "Performance Awareness" and any(
            s in text_lower for s in ["layered", "hexagonal", "clean architecture"]
        ):
            integration_bonus += 1

        # Advanced indicators count extra toward depth
        # Prepend adv_matches as "sophisticated" versions so _assess_proficiency weights them higher
        depth_adjusted_matches = [f"[adv] {m}" for m in adv_matches] + basic_matches

        proficiency, confidence, evidence_refs = _assess_proficiency(
            depth_adjusted_matches, file_distribution,
            integration_bonus=integration_bonus,
            negative_penalty=penalty,
            total_files=total_files,
        )

        if not evidence_refs:
            evidence_refs = [{
                "file": "(content)",
                "signals": all_matches[:5],
                "description": f"Matched {len(all_matches)} craft indicator(s)",
            }]

        skills.append({
            "name": skill_name,
            "proficiency": proficiency,
            "confidence": confidence,
            "evidence": {
                "indicators_matched": basic_matches[:8],
                "advanced_indicators": adv_matches[:5],
            },
            "evidence_refs": evidence_refs,
        })

    prof_order = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    skills.sort(key=lambda s: prof_order.get(s["proficiency"], 0), reverse=True)
    return skills


def _detect_writing_craft(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Detect writing/communication craft (non-code artifacts) (v2)."""
    all_text = _get_content_text(analysis)
    content_samples = _get_content_samples(analysis)
    all_files = _get_all_files(analysis)
    total_files = len(content_samples) if content_samples else max(len(all_files), 1)
    text = all_text.lower()

    skills: list[dict[str, Any]] = []

    craft_signals: dict[str, dict[str, Any]] = {
        "Argumentation": {
            "indicators": ["therefore", "however", "moreover", "consequently",
                          "in contrast", "on the other hand", "this suggests",
                          "evidence", "counter", "argue", "claim"],
            "advanced_indicators": ["logical fallacy", "syllogism", "deductive",
                                     "inductive", "abductive", "counterargument",
                                     "rebuttal", "premise", "warrant"],
        },
        "Technical Writing": {
            "indicators": ["## ", "### ", "```", "table", "figure",
                          "diagram", "example", "note:", "warning:",
                          "appendix", "reference", "citation"],
            "advanced_indicators": ["cross-reference", "glossary", "index",
                                     "table of contents", "footnote", "endnote",
                                     "inline code", "syntax highlighting"],
        },
        "Narrative Structure": {
            "indicators": ["introduction", "conclusion", "summary",
                          "background", "context", "overview", "in this",
                          "chapter", "section", "part i"],
            "advanced_indicators": ["thesis statement", "topic sentence",
                                     "transition", "foreshadowing",
                                     "narrative arc", "climax",
                                     "denouement", "exposition"],
        },
        "Clarity": {
            "indicators": ["for example", "in other words", "specifically",
                          "namely", "that is", "to clarify", "in short",
                          "simply put", "this means"],
            "advanced_indicators": ["plain language", "jargon-free",
                                     "concrete example", "analogy",
                                     "step-by-step", "visual aid",
                                     "executive summary"],
        },
    }

    for skill_name, sig in craft_signals.items():
        indicators = sig["indicators"]
        adv_indicators = sig.get("advanced_indicators", [])
        basic_matches = [ind for ind in indicators if ind.lower() in text]
        adv_matches = [ind for ind in adv_indicators if ind.lower() in text]
        all_matches = basic_matches + adv_matches

        if not all_matches:
            continue

        # Build file distribution
        file_distribution: dict[str, list[str]] = {}
        for ind in all_matches:
            for fpath, content in content_samples.items():
                if ind.lower() in content.lower():
                    file_distribution.setdefault(fpath, []).append(ind)
            if ind not in sum(file_distribution.values(), []):
                file_distribution.setdefault("(content)", []).append(ind)

        depth_adjusted = [f"[adv] {m}" for m in adv_matches] + basic_matches
        proficiency, confidence, evidence_refs = _assess_proficiency(
            depth_adjusted, file_distribution,
            total_files=total_files,
        )

        if not evidence_refs:
            evidence_refs = [{
                "file": "(content)",
                "signals": all_matches[:5],
                "description": f"Matched {len(all_matches)} writing craft indicator(s)",
            }]

        skills.append({
            "name": skill_name,
            "proficiency": proficiency,
            "confidence": confidence,
            "evidence": {
                "indicators_matched": basic_matches[:8],
                "advanced_indicators": adv_matches[:5],
            },
            "evidence_refs": evidence_refs,
        })

    prof_order = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    skills.sort(key=lambda s: prof_order.get(s["proficiency"], 0), reverse=True)
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
            f"2. Evaluate the creator's skill at the medium qualitatively:\n"
            f"   - **Depth**: superficial reference vs sophisticated application\n"
            f"   - **Consistency**: one-off mention vs pervasive usage across files\n"
            f"   - **Integration**: isolated pattern vs interconnected with other skills\n"
            f"   - **For software**: design patterns, error handling, type "
            f"system usage, code organization, performance awareness, security.\n"
            f"   - **For articles/papers**: argumentation quality, technical "
            f"writing, narrative structure, clarity.\n"
            f"   - **For presentations**: visual design, pacing, storytelling.\n"
            f"3. Include evidence_refs with specific file paths and descriptions.\n\n"
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
            f'      }},\n'
            f'      "evidence_refs": [\n'
            f'        {{"file": "<FILE_PATH>", "signals": ["<SIGNAL>"], '
            f'"description": "<WHAT_DEMONSTRATES_THE_SKILL>"}}\n'
            f'      ]\n'
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
# Detector 3 — Methodology & Tooling (v2)
# ═══════════════════════════════════════════════════════════════════════════

# v2: Expanded from ~30 to 55+ tools with modern ecosystem coverage.
# Config file indicators support version detection.
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
    "Vitest": ["vitest.config", "vitest"],
    "Playwright": ["playwright.config", "@playwright/test"],
    # CI/CD
    "GitHub Actions": [".github/workflows"],
    "GitLab CI": [".gitlab-ci.yml"],
    "Jenkins": ["Jenkinsfile"],
    "CircleCI": [".circleci"],
    # Containers
    "Docker": ["Dockerfile", "docker-compose"],
    "Kubernetes": [".yaml", "kustomization", "helm"],
    # Docs & writing
    "LaTeX": [".tex", ".bib"],
    "Markdown": [".md"],
    "Sphinx": ["conf.py", "index.rst"],
    "MkDocs": ["mkdocs.yml"],
    "Jupyter": [".ipynb"],
    "Storybook": [".stories.", "storybook"],
    # Data
    "Pandas": ["pandas", "DataFrame"],
    "NumPy": ["numpy", "ndarray"],
    "PyTorch": ["torch", "nn.Module"],
    "scikit-learn": ["sklearn", "fit(", "predict("],
    # Web frameworks
    "React": ["react", "jsx", "tsx"],
    "Vue": ["vue", ".vue"],
    "Django": ["manage.py", "settings.py", "urls.py"],
    "Flask": ["flask", "app.py"],
    "FastAPI": ["fastapi", "uvicorn"],
    "Svelte": [".svelte"],
    "SolidJS": ["solid-js", "createSignal"],
    "Qwik": ["@builder.io/qwik", "qwik"],
    "Astro": ["astro.config", ".astro"],
    "htmx": ["htmx", "hx-get", "hx-post"],
    "Alpine.js": ["alpinejs", "x-data", "x-show"],
    "Next.js": ["next.config", "nextjs"],
    "Nuxt": ["nuxt.config"],
    "Remix": ["remix.config", "@remix-run"],
    # API / data layer
    "tRPC": ["@trpc", "trpc"],
    "Prisma": ["schema.prisma", "prisma"],
    "Drizzle": ["drizzle.config", "drizzle"],
    "TanStack Query": ["@tanstack/react-query", "useQuery", "useMutation"],
    "GraphQL": [".graphql", "apollo", "relay"],
    # State management
    "Zustand": ["zustand", "create(", "useStore"],
    "Redux": ["redux", "createSlice", "configureStore"],
    # Build & tooling
    "Webpack": ["webpack.config"],
    "Vite": ["vite.config"],
    "Turbopack": ["turbopack"],
    "ESLint": [".eslintrc", "eslint.config"],
    "Prettier": [".prettierrc"],
    "Biome": ["biome.json"],
    # Python tooling (v2)
    "uv": ["[tool.uv]", "uv.lock"],
    "Ruff": ["[tool.ruff]", "ruff.toml"],
    "Poetry": ["poetry.lock", "[tool.poetry]"],
    # Package managers
    "pnpm": ["pnpm-lock.yaml", "pnpm-workspace"],
    "Bun": ["bun.lockb", "bunfig.toml"],
    "Yarn": ["yarn.lock", ".yarnrc"],
    # CSS / styling
    "TailwindCSS": ["tailwind.config", "@tailwind", "@apply"],
    "shadcn/ui": ["@radix-ui", "components/ui", "shadcn"],
    # Testing / mocking
    "MSW": ["msw", "setupServer", "setupWorker"],
    # Infra
    "Terraform": [".tf"],
    "Ansible": ["ansible", "playbook"],
}

# v2: Category mapping for tool classification
_TOOL_CATEGORY_MAP: dict[str, str] = {
    "Git": "Version Control",
    "GitHub": "Version Control",
    "Python": "Languages & Runtimes",
    "JavaScript": "Languages & Runtimes",
    "TypeScript": "Languages & Runtimes",
    "Go": "Languages & Runtimes",
    "Rust": "Languages & Runtimes",
    "Java": "Languages & Runtimes",
    "pytest": "Testing",
    "Jest": "Testing",
    "unittest": "Testing",
    "Vitest": "Testing",
    "Playwright": "Testing",
    "MSW": "Testing",
    "GitHub Actions": "CI/CD",
    "GitLab CI": "CI/CD",
    "Jenkins": "CI/CD",
    "CircleCI": "CI/CD",
    "Docker": "Containers & Orchestration",
    "Kubernetes": "Containers & Orchestration",
    "LaTeX": "Documentation Tools",
    "Markdown": "Documentation Tools",
    "Sphinx": "Documentation Tools",
    "MkDocs": "Documentation Tools",
    "Jupyter": "Documentation Tools",
    "Storybook": "Documentation Tools",
    "React": "Web Frameworks",
    "Vue": "Web Frameworks",
    "Django": "Web Frameworks",
    "Flask": "Web Frameworks",
    "FastAPI": "Web Frameworks",
    "Svelte": "Web Frameworks",
    "SolidJS": "Web Frameworks",
    "Qwik": "Web Frameworks",
    "Astro": "Web Frameworks",
    "htmx": "Web Frameworks",
    "Alpine.js": "Web Frameworks",
    "Next.js": "Web Frameworks",
    "Nuxt": "Web Frameworks",
    "Remix": "Web Frameworks",
    "tRPC": "API & Data Layer",
    "Prisma": "API & Data Layer",
    "Drizzle": "API & Data Layer",
    "GraphQL": "API & Data Layer",
    "TanStack Query": "State Management",
    "Zustand": "State Management",
    "Redux": "State Management",
    "Pandas": "Data Science & ML",
    "NumPy": "Data Science & ML",
    "PyTorch": "Data Science & ML",
    "scikit-learn": "Data Science & ML",
    "Webpack": "Build & Tooling",
    "Vite": "Build & Tooling",
    "Turbopack": "Build & Tooling",
    "ESLint": "Build & Tooling",
    "Prettier": "Build & Tooling",
    "Biome": "Build & Tooling",
    "uv": "Package Management",
    "Ruff": "Build & Tooling",
    "Poetry": "Package Management",
    "pnpm": "Package Management",
    "Bun": "Package Management",
    "Yarn": "Package Management",
    "TailwindCSS": "CSS & Styling",
    "shadcn/ui": "CSS & Styling",
    "Terraform": "Infrastructure as Code",
    "Ansible": "Infrastructure as Code",
}

# v2: Tool indicators that are version-bearing config files.
# When detected, we attempt to extract version info.
_VERSION_CONFIG_FILES: dict[str, str] = {
    "pyproject.toml": "Python/Poetry/uv",
    "package.json": "JavaScript/TypeScript",
    "Cargo.toml": "Rust",
    "go.mod": "Go",
    "pom.xml": "Java",
    "build.gradle": "Java",
}

# v2: Ecosystem-level tool indicators — tools implied by ecosystem
# that may not be directly used in the artifact.
_ECOSYSTEM_LEVEL_INDICATORS: set[str] = {
    "Markdown",  # ubiquitous, not a distinguishing tool
    "JavaScript",  # implied by .js files, not necessarily intentional choice
    "Python",  # implied by .py files
    "TypeScript",  # implied by .ts files
}


def _detect_tool_version(
    tool_name: str, indicator: str, all_files: list[str],
    working_dir: str = ".",
) -> str | None:
    """Attempt to detect a tool's version from config files.

    Args:
        tool_name: The name of the tool.
        indicator: The indicator that matched.
        all_files: List of all project files.
        working_dir: Base directory for file reading.

    Returns:
        Version string if found, or None.
    """
    # Check if any version-bearing config file exists that's relevant to this tool
    config_files_to_check: list[str] = []
    for cf in _VERSION_CONFIG_FILES:
        if any(cf in f for f in all_files):
            config_files_to_check.append(cf)

    if not config_files_to_check:
        return None

    for cf in config_files_to_check:
        # Find the actual file path
        candidates = [f for f in all_files if cf in f]
        if not candidates:
            continue
        fpath = os.path.join(working_dir, candidates[0])
        try:
            if cf == "pyproject.toml":
                import re
                with open(fpath) as f:
                    content = f.read()
                # Look for tool version in pyproject.toml
                for line in content.splitlines():
                    # [tool.uv], [tool.ruff], [tool.poetry], etc.
                    m = re.search(r'^\[tool\.(\w+)\]', line)
                    if m and m.group(1).lower() == tool_name.lower():
                        return "detected in pyproject.toml"
                # Check project version as fallback
                m = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
                if m:
                    return m.group(1)
            elif cf == "package.json":
                with open(fpath) as f:
                    pkg = json.load(f)
                # Check dependencies for version
                deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
                dep_key = tool_name.lower()
                for key, ver in deps.items():
                    if key.lower() == dep_key:
                        return str(ver).lstrip("^~")
                # Check package.json version itself
                if "version" in pkg:
                    return pkg["version"]
        except (json.JSONDecodeError, OSError, ValueError):
            continue

    return None


def _detect_tools(
    all_files: list[str], all_text: str, working_dir: str = ".",
) -> list[dict[str, Any]]:
    """Deterministic tool detection from files and content (v2).

    v2 improvements:
    - 55+ modern tools including JS/TS ecosystem and Python tooling
    - Version detection from config files (pyproject.toml, package.json)
    - ``scope`` field: ``project_level`` (directly used) vs ``ecosystem_level`` (implied)
    - Category map centralized for consistency
    """
    files_str = " ".join(all_files)
    combined = (files_str + " " + all_text).lower()
    tools: list[dict[str, Any]] = []

    for tool, indicators in _TOOL_INDICATORS.items():
        matches = [ind for ind in indicators if ind.lower() in combined]
        if not matches:
            continue

        # Determine scope: project_level (config files found) vs ecosystem_level (implied)
        scope = "ecosystem_level" if tool in _ECOSYSTEM_LEVEL_INDICATORS else "project_level"
        # Override: if config-file indicators were found, it's definitely project_level
        config_indicators = [m for m in matches if m.endswith((".toml", ".json", ".yml", ".yaml",
                                ".lock", ".config", "Dockerfile"))]
        if config_indicators:
            scope = "project_level"

        # Categorize using centralized map
        cat = _TOOL_CATEGORY_MAP.get(tool, "Other Tools")

        # v2: Qualitative proficiency — not just count of matches
        # Config file matches indicate deeper usage than text matches
        file_matches = [m for m in matches if m.endswith(tuple(
            [".toml", ".json", ".yml", ".yaml", ".lock", ".config",
             "Dockerfile", ".gitignore", ".gitattributes", "Jenkinsfile",
             ".github/", ".circleci", "conftest.py", "jest.config",
             "vitest.config", "playwright.config", "astro.config",
             "next.config", "nuxt.config", "remix.config",
             "webpack.config", "vite.config", "schema.prisma",
             "drizzle.config", "tailwind.config", "biome.json",
             ".eslintrc", ".prettierrc", ".editorconfig",
        ]))]
        text_matches = [m for m in matches if m not in file_matches]

        # Depth: config files = deeper integration, text content = lighter usage
        depth_score = len(file_matches) * 2 + len(text_matches) * 1

        if depth_score >= 8:
            proficiency = "expert"
        elif depth_score >= 5:
            proficiency = "advanced"
        elif depth_score >= 2:
            proficiency = "intermediate"
        else:
            proficiency = "beginner"

        # Attempt version detection
        version: str | None = None
        if config_indicators:
            # Try each indicator as a potential version source
            for ind in matches:
                ver = _detect_tool_version(tool, ind, all_files, working_dir)
                if ver:
                    version = ver
                    break

        tool_entry: dict[str, Any] = {
            "name": tool,
            "category": cat,
            "proficiency": proficiency,
            "scope": scope,
            "evidence": matches[:5],
        }
        if version:
            tool_entry["version"] = version

        tools.append(tool_entry)

    return sorted(tools, key=lambda t: len(t.get("evidence", [])), reverse=True)


def _detect_common_tools(all_files: list[str]) -> list[dict[str, Any]]:
    """Public helper — detect tools from file list only."""
    return _detect_tools(all_files, "")


def _detect_methodology_skills(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Deterministic methodology skill detection (v2).

    v2 improvements:
    - Qualitative proficiency: sophistication of methodology implementation,
      not just presence of signals
    - Multi-stage/deep indicators distinguish sophisticated from basic
    - evidence_refs with specific file paths
    """
    all_files = _get_all_files(analysis)
    all_text = _get_content_text(analysis)
    content_samples = _get_content_samples(analysis)
    all_files_str = " ".join(all_files).lower()
    combined = (all_files_str + " " + all_text).lower()
    total_files = len(content_samples) if content_samples else max(len(all_files), 1)

    skills: list[dict[str, Any]] = []

    method_signals: dict[str, dict[str, Any]] = {
        "Git Workflow Maturity": {
            "signals": ["feat:", "fix:", "chore:", "refactor:", "docs:",
                       "test:", "conventional commit", "branch",
                       "merge request", "pull request"],
            "file_signals": [".git/"],
            # v2: deep signals indicate sophisticated Git workflow
            "deep_signals": [
                "signed-off-by", "co-authored-by", "breaking change",
                "semantic versioning", "commitlint", "commitizen",
                "git flow", "github flow", "trunk-based",
            ],
            "deep_file_signals": [
                ".gitmessage", ".commitlintrc", ".czrc",
                "CODEOWNERS", ".git-blame-ignore-revs",
            ],
        },
        "CI/CD Sophistication": {
            "signals": ["pipeline", "deploy", "stage", "job:", "workflow",
                       "ci.yml", "build-and-test", "release"],
            "file_signals": [".github/workflows", ".gitlab-ci.yml", "Jenkinsfile",
                           ".circleci", "azure-pipelines"],
            "deep_signals": [
                "staging environment", "production deployment",
                "approval gate", "canary deploy", "blue-green",
                "rollback", "smoke test", "integration test",
                "matrix build", "cache", "artifact",
            ],
            "deep_file_signals": [
                "docker-compose.ci.yml", "docker-compose.prod.yml",
            ],
        },
        "Dependency Management": {
            "signals": ["version", "lock", "pinned", "requirements",
                       "dependencies", "devDependencies"],
            "file_signals": ["pyproject.toml", "package.json", "poetry.lock",
                           "Pipfile.lock", "yarn.lock", "package-lock.json",
                           "Cargo.lock", "go.sum"],
            "deep_signals": [
                "devDependencies", "peerDependencies",
                "optionalDependencies", "resolutions",
                "overrides", "renovate", "dependabot",
                "dependency graph", "SBOM",
            ],
            "deep_file_signals": [
                ".github/dependabot.yml", "renovate.json",
            ],
        },
        "Environment Management": {
            "signals": ["Dockerfile", "docker-compose", "devcontainer",
                       "virtualenv", ".venv", "conda", "nix", "nvm"],
            "file_signals": ["Dockerfile", "docker-compose.yml",
                           ".devcontainer", ".python-version", ".nvmrc"],
            "deep_signals": [
                "multi-stage build", "healthcheck", "docker-compose.override",
                "docker-compose.prod", "NixOS", "flakes",
                "devcontainer.json", "docker-compose.dev.yml",
            ],
            "deep_file_signals": [
                "docker-compose.override.yml",
                "docker-compose.dev.yml",
                "docker-compose.prod.yml",
                "flake.nix", "shell.nix",
            ],
        },
        "Incremental Development": {
            "signals": ["TODO", "FIXME", "HACK", "WIP", "v0.", "v1.",
                       "changelog", "version history", "migration"],
            "file_signals": ["CHANGELOG.md", "MIGRATION.md", "VERSION"],
            "deep_signals": [
                "breaking change", "deprecation warning",
                "upgrade guide", "backward compatibility",
                "migration guide", "release notes",
            ],
            "deep_file_signals": ["UPGRADE.md", "RELEASES.md"],
        },
    }

    for skill_name, sig in method_signals.items():
        text_matches = [s for s in sig["signals"] if s.lower() in combined]
        file_matches = [s for s in sig["file_signals"] if s.lower() in all_files_str]
        deep_text = [s for s in sig.get("deep_signals", []) if s.lower() in combined]
        deep_file = [s for s in sig.get("deep_file_signals", []) if s.lower() in all_files_str]

        all_text_signals = text_matches + deep_text
        all_file_signals = file_matches + deep_file

        if not (all_text_signals or all_file_signals):
            continue

        # Build file distribution for evidence_refs
        file_distribution: dict[str, list[str]] = {}
        for sig_text in text_matches + deep_text:
            for fpath, content in content_samples.items():
                if sig_text.lower() in content.lower():
                    file_distribution.setdefault(fpath, []).append(sig_text)
        for sig_file in file_matches + deep_file:
            file_distribution.setdefault(sig_file, []).append(sig_file)

        # v2: Qualitative scoring
        # Deep signals (multi-stage CI, multi-container env, dependabot) = sophisticated
        # File signals (config files) = concrete evidence, weight more
        # Text signals (mentions) = awareness but may be superficial
        depth_score = (
            len(deep_text) * 3.0 +
            len(deep_file) * 4.0 +
            len(text_matches) * 1.0 +
            len(file_matches) * 2.0
        )

        # Integration bonus: related methodology skills reinforcing each other
        integration_bonus = 0
        if skill_name == "CI/CD Sophistication" and any(
            s in combined for s in ["docker", "container", "kubernetes"]
        ):
            integration_bonus += 1  # CI/CD + containers = mature infra
        if skill_name == "Dependency Management" and any(
            s in combined for s in ["renovate", "dependabot", "security"]
        ):
            integration_bonus += 1  # automated dep management
        if skill_name == "Environment Management" and any(
            s in combined for s in ["devcontainer", "multi-stage"]
        ):
            integration_bonus += 1  # sophisticated env setup

        # Map to proficiency
        if depth_score >= 12.0:
            proficiency = "expert"
        elif depth_score >= 7.0:
            proficiency = "advanced"
        elif depth_score >= 3.0:
            proficiency = "intermediate"
        elif depth_score >= 1.0:
            proficiency = "beginner"
        else:
            proficiency = "beginner"

        if integration_bonus >= 2:
            # Boost proficiency for highly integrated methodology
            prof_order = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
            levels = ["beginner", "intermediate", "advanced", "expert"]
            current_idx = levels.index(proficiency)
            boosted_idx = min(current_idx + 1, len(levels) - 1)
            proficiency = levels[boosted_idx]

        # Confidence
        total_signals = len(all_text_signals) + len(all_file_signals)
        has_deep = bool(deep_text or deep_file)
        if total_signals >= 6 and has_deep:
            confidence = "high"
        elif total_signals >= 3:
            confidence = "medium"
        else:
            confidence = "low"

        # Build evidence_refs
        evidence_refs: list[dict[str, Any]] = []
        for fpath, sigs in file_distribution.items():
            evidence_refs.append({
                "file": fpath,
                "signals": sigs[:5],
                "description": f"Found {len(sigs)} methodology signal(s)",
            })
        if not evidence_refs:
            evidence_refs = [{
                "file": "(file list)",
                "signals": all_text_signals[:5],
                "description": f"Matched {len(all_text_signals)} text signal(s) and "
                               f"{len(all_file_signals)} file signal(s)",
            }]

        skills.append({
            "name": skill_name,
            "proficiency": proficiency,
            "confidence": confidence,
            "evidence": {
                "text_signals": text_matches[:6],
                "file_signals": file_matches[:6],
                "deep_signals": deep_text[:4],
                "deep_file_signals": deep_file[:4],
            },
            "evidence_refs": evidence_refs,
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
            f"   For each tool, distinguish project_level (directly used) vs "
            f"ecosystem_level (implied by language/ecosystem).\n"
            f"3. Infer methodology skills qualitatively:\n"
            f"   - Git workflow maturity: basic commits vs conventional commits, "
            f"signed-off, git-flow\n"
            f"   - CI/CD sophistication: single stage vs multi-stage with "
            f"staging/production deployment\n"
            f"   - Dependency management: manual vs locked+pinned with "
            f"dependabot/renovate\n"
            f"   - Environment management: single Dockerfile vs multi-container "
            f"with health checks\n"
            f"   - Release management: tags only vs changelogs+versioning+migration guides\n"
            f"   - Incremental development: monolithic vs iterative with "
            f"deprecation warnings\n"
            f"4. Include evidence_refs with specific file paths.\n"
            f"5. For non-code artifacts, look for text-specific tools "
            f"(LaTeX, Markdown, reference managers, CMS, SEO).\n\n"
            f"## Output: `{self.output_file}`\n"
            f"```json\n"
            f"{{\n"
            f'  "domain": "{self.domain}",\n'
            f'  "detected_tools": [\n'
            f'    {{\"name\": \"<TOOL_NAME>\", \"category\": \"<CATEGORY>\",\n'
            f'     "proficiency": "advanced", "scope": "project_level",\n'
            f'     "evidence": ["commit messages"]}}\n'
            f'  ],\n'
            f'  "inferred_skills": [\n'
            f'    {{\n'
            f'      "name": "<SKILL_NAME>",\n'
            f'      "proficiency": "advanced|intermediate|beginner|expert",\n'
            f'      "confidence": "high|medium|low",\n'
            f'      "evidence": {{"description": "..."}},\n'
            f'      "evidence_refs": [\n'
            f'        {{"file": "<FILE_PATH>", "signals": ["<SIGNAL>"], '
            f'"description": "<WHAT_DEMONSTRATES_THE_SKILL>"}}\n'
            f'      ]\n'
            f'    }}\n'
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

        tools = _detect_tools(all_files, all_text, working_dir)
        skills = _detect_methodology_skills(analysis)

        return {
            "domain": self.domain,
            "detected_tools": tools,
            "inferred_skills": skills,
            "_fallback": True,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Detector 4 — Depth & Rigor (v2)
# ═══════════════════════════════════════════════════════════════════════════

def _detect_rigor_skills(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Deterministic rigor/depth detection (v2).

    v2 improvements:
    - Qualitative proficiency: testing pyramid completeness, documentation
      hierarchy depth, quality enforcement layers
    - evidence_refs with specific file paths
    - Deep signals distinguish sophisticated rigor from basic
    """
    all_files = _get_all_files(analysis)
    all_text = _get_content_text(analysis)
    content_samples = _get_content_samples(analysis)
    all_files_str = " ".join(all_files).lower()
    combined = (all_files_str + " " + all_text).lower()
    artifact_type = _get_artifact_type(analysis)
    total_files = len(content_samples) if content_samples else max(len(all_files), 1)

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

    # ═══ Testing Strategy (v2: testing pyramid completeness) ═══
    test_ratio = len(test_files) / max(total, 1)

    # v2: distinguish test types for pyramid assessment
    unit_test_files = [
        f for f in test_files
        if "integration" not in f.lower() and "e2e" not in f.lower()
        and "end_to_end" not in f.lower() and "smoke" not in f.lower()
    ]
    integration_test_files = [
        f for f in test_files
        if "integration" in f.lower() or "integ" in f.lower()
    ]
    e2e_test_files = [
        f for f in test_files
        if "e2e" in f.lower() or "end_to_end" in f.lower()
        or "playwright" in f.lower() or "cypress" in f.lower()
    ]

    if test_files:
        # v2: qualitative assessment based on testing pyramid
        has_unit = len(unit_test_files) > 0
        has_integration = len(integration_test_files) > 0
        has_e2e = len(e2e_test_files) > 0
        pyramid_levels = sum([has_unit, has_integration, has_e2e])

        # Testing infrastructure depth signals
        deep_test_signals = [
            "parameterize", "fixture", "conftest", "mock", "stub",
            "faker", "property-based", "snapshot test", "regression test",
            "performance test", "load test", "stress test", "chaos test",
        ]
        deep_matches = [s for s in deep_test_signals if s.lower() in combined]

        if pyramid_levels >= 3:
            proficiency = "expert"
        elif pyramid_levels >= 2 or (has_unit and len(deep_matches) >= 4):
            proficiency = "advanced"
        elif has_unit or test_ratio > 0.05:
            proficiency = "intermediate"
        else:
            proficiency = "beginner"

        confidence = "high" if test_ratio > 0.15 else \
                     "medium" if test_ratio > 0.05 else "low"

        # Build evidence_refs with test file breakdown
        evidence_refs: list[dict[str, Any]] = []
        if unit_test_files:
            evidence_refs.append({
                "file": unit_test_files[0],
                "signals": ["unit test"],
                "description": f"{len(unit_test_files)} unit test file(s) found",
            })
        if integration_test_files:
            evidence_refs.append({
                "file": integration_test_files[0],
                "signals": ["integration test"],
                "description": f"{len(integration_test_files)} integration test file(s)",
            })
        if e2e_test_files:
            evidence_refs.append({
                "file": e2e_test_files[0],
                "signals": ["e2e test"],
                "description": f"{len(e2e_test_files)} E2E test file(s)",
            })
        if deep_matches:
            evidence_refs.append({
                "file": "(content)",
                "signals": deep_matches[:5],
                "description": f"Advanced testing infrastructure: {', '.join(deep_matches[:3])}",
            })

        skills.append({
            "name": "Testing Strategy",
            "proficiency": proficiency,
            "confidence": confidence,
            "evidence": {
                "test_count": len(test_files),
                "test_ratio": round(test_ratio, 3),
                "unit_tests": len(unit_test_files),
                "integration_tests": len(integration_test_files),
                "e2e_tests": len(e2e_test_files),
                "pyramid_levels": pyramid_levels,
                "sample_files": test_files[:5],
            },
            "evidence_refs": evidence_refs,
        })

    # ═══ Test Coverage Thoroughness (v2: qualitative) ═══
    coverage_signals = ["assert", "expect(", "should ", "test(", "it(",
                        "describe(", "def test_", "class Test",
                        "parameterize", "fixture", "mock", "stub",
                        "edge case", "corner case", "boundary"]
    basic_cov = [s for s in coverage_signals[:10] if s.lower() in combined]
    advanced_cov = ["parameterize", "fixture", "mock", "stub",
                     "edge case", "corner case", "boundary",
                     "property-based", "snapshot", "mutation test"]
    adv_cov_matches = [s for s in advanced_cov if s.lower() in combined]
    all_cov_matches = [s for s in coverage_signals if s.lower() in combined]

    if all_cov_matches:
        # File distribution for coverage
        cov_dist: dict[str, list[str]] = {}
        for sig in all_cov_matches:
            for fpath, content in content_samples.items():
                if sig.lower() in content.lower():
                    cov_dist.setdefault(fpath, []).append(sig)

        coverage_depth = len(all_cov_matches) + len(adv_cov_matches) * 2
        if coverage_depth >= 20:
            coverage_prof = "expert"
        elif coverage_depth >= 12:
            coverage_prof = "advanced"
        elif coverage_depth >= 5:
            coverage_prof = "intermediate"
        else:
            coverage_prof = "beginner"

        cov_evidence_refs: list[dict[str, Any]] = [
            {"file": fp, "signals": sigs[:5],
             "description": f"{len(sigs)} coverage signal(s)"}
            for fp, sigs in cov_dist.items()
        ] if cov_dist else [{
            "file": "(content)",
            "signals": all_cov_matches[:5],
            "description": f"{len(all_cov_matches)} coverage signal(s)",
        }]

        skills.append({
            "name": "Test Coverage Thoroughness",
            "proficiency": coverage_prof,
            "confidence": "high" if coverage_depth >= 15 else
                          "medium" if coverage_depth >= 6 else "low",
            "evidence": {
                "indicators": all_cov_matches[:10],
                "advanced_indicators": adv_cov_matches[:5],
            },
            "evidence_refs": cov_evidence_refs,
        })

    # ═══ Documentation Quality (v2: documentation hierarchy depth) ═══
    doc_ratio = len(doc_files) / max(total, 1)

    # v2: Distinguish documentation hierarchy levels
    has_readme = any("readme" in f.lower() for f in doc_files)
    has_api_docs = any(
        ind in all_files_str
        for ind in ["api", "sphinx", "mkdocs", "jsdoc", "typedoc", "pydoc"]
    )
    has_contributing = any("contributing" in f.lower() for f in doc_files) or \
                       "contributing" in combined
    has_architecture = any(
        ind in combined
        for ind in ["architecture", "design doc", "ADR", "design decision",
                     "technical spec", "RFC"]
    )
    has_changelog = any("changelog" in f.lower() for f in doc_files) or \
                    "changelog" in combined

    doc_hierarchy_levels = sum([has_readme, has_api_docs, has_contributing,
                                 has_architecture, has_changelog])

    doc_content_signals = ["readme", "getting started", "installation",
                          "usage", "api", "example", "tutorial", "guide",
                          "reference", "faq", "troubleshooting", "contributing"]
    doc_matches = [s for s in doc_content_signals if s in combined]

    # v2: Deep documentation signals
    deep_doc_signals = [
        "architecture decision record", "ADR", "style guide",
        "code of conduct", "security policy", "governance",
        "roadmap", "backlog", "release notes",
    ]
    deep_doc_matches = [s for s in deep_doc_signals if s in combined]

    if doc_files or doc_matches:
        doc_score = len(doc_matches) + len(doc_files) * 2 + len(deep_doc_matches) * 3

        if doc_hierarchy_levels >= 4:
            doc_prof = "expert"
        elif doc_hierarchy_levels >= 3:
            doc_prof = "advanced"
        elif doc_hierarchy_levels >= 2:
            doc_prof = "intermediate"
        else:
            doc_prof = "beginner"

        doc_evidence_refs: list[dict[str, Any]] = []
        if doc_files:
            doc_evidence_refs.append({
                "file": doc_files[0],
                "signals": doc_files[:5],
                "description": f"{len(doc_files)} documentation file(s): "
                               f"readme={has_readme}, api={has_api_docs}, "
                               f"contrib={has_contributing}, arch={has_architecture}",
            })

        skills.append({
            "name": "Documentation Quality",
            "proficiency": doc_prof,
            "confidence": "high" if doc_score >= 12 else
                          "medium" if doc_score >= 5 else "low",
            "evidence": {
                "doc_files": doc_files[:5],
                "content_signals": doc_matches[:8],
                "hierarchy_levels": doc_hierarchy_levels,
            },
            "evidence_refs": doc_evidence_refs or [{
                "file": "(content)",
                "signals": doc_matches[:5],
                "description": f"Documentation hierarchy: {doc_hierarchy_levels} levels",
            }],
        })

    # ═══ Code Quality Enforcement (v2: quality enforcement layers) ═══
    quality_files = ["eslint", "prettier", ".ruff", "pyproject.toml",
                    "mypy", "flake8", "pylint", "pre-commit",
                    ".editorconfig", "husky", "lint-staged"]
    quality_matches = [q for q in quality_files if q in all_files_str]

    # v2: Distinguish enforcement layers
    has_linter = any(q in all_files_str for q in
                     ["eslint", ".ruff", "flake8", "pylint", "biome"])
    has_formatter = any(q in all_files_str for q in
                        ["prettier", ".ruff", "biome", "black"])
    has_type_checker = any(q in all_files_str for q in
                           ["mypy", "tsc", "pyright", "pyre"])
    has_precommit = "pre-commit" in all_files_str or "husky" in all_files_str
    has_ci_quality = any(q in combined for q in
                         ["lint", "typecheck", "format-check", "quality gate"])

    enforcement_layers = sum([has_linter, has_formatter, has_type_checker,
                               has_precommit, has_ci_quality])

    if quality_matches:
        if enforcement_layers >= 4:
            qa_prof = "expert"
        elif enforcement_layers >= 3:
            qa_prof = "advanced"
        elif enforcement_layers >= 2:
            qa_prof = "intermediate"
        else:
            qa_prof = "beginner"

        qa_evidence_refs = [{
            "file": "(config)",
            "signals": quality_matches,
            "description": f"Quality enforcement layers ({enforcement_layers}): "
                           f"linter={has_linter}, formatter={has_formatter}, "
                           f"type={has_type_checker}, precommit={has_precommit}, "
                           f"ci={has_ci_quality}",
        }]

        skills.append({
            "name": "Code Quality Enforcement",
            "proficiency": qa_prof,
            "confidence": "high" if enforcement_layers >= 3 else "medium",
            "evidence": {
                "configs_found": quality_matches,
                "enforcement_layers": enforcement_layers,
            },
            "evidence_refs": qa_evidence_refs,
        })

    # ═══ Academic Rigor (v2: for non-code artifacts) ═══
    if artifact_type not in ("software_project",):
        rigor_signals = ["citation", "reference", "source", "footnote",
                        "bibliography", "appendix", "methodology",
                        "limitation", "future work", "acknowledgment",
                        "data available", "code available", "reproducible"]

        # v2: Deep academic rigor signals
        deep_rigor_signals = [
            "peer review", "systematic review", "meta-analysis",
            "randomized controlled", "double-blind", "informed consent",
            "IRB", "institutional review", "conflict of interest",
            "supplementary material", "preregistered",
            "replication study",
        ]

        basic_rigor = [r for r in rigor_signals if r in combined]
        deep_rigor = [r for r in deep_rigor_signals if r in combined]
        all_rigor = basic_rigor + deep_rigor

        if all_rigor:
            rigor_depth = len(basic_rigor) + len(deep_rigor) * 3

            if rigor_depth >= 18:
                rigor_prof = "expert"
            elif rigor_depth >= 10:
                rigor_prof = "advanced"
            elif rigor_depth >= 5:
                rigor_prof = "intermediate"
            else:
                rigor_prof = "beginner"

            rigor_evidence_refs = [{
                "file": "(content)",
                "signals": all_rigor[:8],
                "description": f"{len(basic_rigor)} basic + {len(deep_rigor)} deep "
                               f"rigor signal(s)",
            }]

            skills.append({
                "name": "Academic Rigor",
                "proficiency": rigor_prof,
                "confidence": "high" if rigor_depth >= 12 else
                              "medium" if rigor_depth >= 6 else "low",
                "evidence": {
                    "rigor_signals": basic_rigor[:10],
                    "deep_rigor_signals": deep_rigor[:5],
                },
                "evidence_refs": rigor_evidence_refs,
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
            f"2. Evaluate rigor and thoroughness qualitatively:\n"
            f"   - **For software**: testing pyramid completeness (unit + integration "
            f"+ E2E), test coverage quality, edge case handling, "
            f"documentation hierarchy (README + API + architecture + contributing), "
            f"quality enforcement layers (linter + formatter + type checker + "
            f"pre-commit + CI).\n"
            f"   - **For articles/papers**: citations, methodology section, "
            f"data availability, limitations, editing quality, peer review.\n"
            f"   - **For datasets**: schema documentation, data validation, "
            f"completeness notes, processing scripts.\n"
            f"3. What's missing? What would a more thorough version include?\n"
            f"4. Include evidence_refs with specific file paths.\n\n"
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
            f'      }},\n'
            f'      "evidence_refs": [\n'
            f'        {{"file": "<FILE_PATH>", "signals": ["<SIGNAL>"], '
            f'"description": "<WHAT_DEMONSTRATES_THE_SKILL>"}}\n'
            f'      ]\n'
            f'    }}\n'
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
