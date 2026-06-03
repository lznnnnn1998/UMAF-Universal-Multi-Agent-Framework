import json
import os
import re
from typing import Any

from agent import AgentResult, AgentRole, BaseDecomposerRole
from tools import TOOL_MAP, ToolRegistry


# ═══════════════════════════════════════════════════════════════════════════
# CoderPP Decomposer
# ═══════════════════════════════════════════════════════════════════════════

_ENV_SETUP_PHASE = """## Phase 2: Environment Setup
After finalizing the module decomposition, you MUST set up and DOCUMENT the environment.
All worker agents will rely on this environment — it must be consistent and well-documented.

Run these exact commands and record ALL output in an `ENVIRONMENT.md` file:

1. **Python path**: `which python3` (or `which python`) — record the full path
2. **Python version**: `python3 --version` — record the exact version
3. **Conda environment**: `echo "$CONDA_DEFAULT_ENV"` and `conda info --envs 2>/dev/null || echo "conda not available"` — record the active conda env name
4. **Pip location**: `which pip3` or `python3 -m pip --version` — record pip path
5. **Installed packages**: `python3 -m pip list 2>/dev/null | head -40` — snapshot key packages already available
6. **Working directory**: `pwd` — record the absolute working directory path

Then write `ENVIRONMENT.md` with ALL of this information in a clear format:
```markdown
# Project Environment

## Python
- Path: /path/to/python3
- Version: Python 3.x.x

## Conda
- Environment: my_env (or "none")
- Conda path: /path/to/conda

## Working Directory
- Path: /absolute/path/to/working_dir

## Required Packages (requirements.txt)
- torch
- numpy
- ...

## Pre-installed Packages (snapshot)
- ...
```

7. **Write requirements.txt**: Based on ALL sub-modules, list required packages (one per line, use `>=` constraints).
8. **Install missing packages**: Run `python3 -m pip install -r requirements.txt`. If some fail, document which ones.
9. Update ENVIRONMENT.md with the installation results.

The `ENVIRONMENT.md` file WILL be read by worker agents so they know exactly which python, which conda env, and which packages to use. This is critical — inconsistent environments are the #1 cause of worker failures."""


class CoderPPDecomposerRole(BaseDecomposerRole):
    """Decompose a coding task into sub-modules and set up the project environment."""

    agent_name = "coderpp_head"
    max_steps = 20

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        return ToolRegistry.to_dicts(ToolRegistry.coderpp_decomposer_tools(backend))

    # -- Template overrides ---------------------------------------------------

    def _role_prompt(self, input_spec: str, **context: Any) -> str:
        is_tex = (
            input_spec[:200].count("\\section") >= 2
            or "research proposal" in input_spec[:500].lower()
        )
        if is_tex:
            return (
                "You are a software architect translating a research proposal "
                "into code modules. This is a LaTeX research proposal describing "
                "specific techniques, methods, and algorithms. Extract the KEY "
                "TECHNIQUES/METHODS described and decompose them into code modules "
                "that IMPLEMENT those techniques. Focus on the actual algorithms "
                "and methods (Flash Attention, GQA, PagedAttention, Ring Attention, "
                "etc.) — NOT on parsing LaTeX, extracting text, or processing "
                "documents."
            )
        return (
            "You are a software architect. Analyze this coding requirement and "
            "decompose it into self-contained sub-modules that can be implemented "
            "independently."
        )

    def _sizing_guide(self) -> str:
        return (
            "- **Simple** (single script, few features): 2-3 modules\n"
            "- **Moderate** (multi-file, several features): 4-5 modules\n"
            "- **Complex** (full app, many interacting parts): up to 20 modules"
        )

    def _sub_unit_requirements(self) -> str:
        return (
            "- Be specific and self-contained with clear interfaces\n"
            "- Each module should be independently testable\n"
            "- List the exact files each module should produce\n"
            "- Declare dependencies on other modules (by module_name)\n"
            "- Cover different functional areas of the overall requirement"
        )

    @staticmethod
    def _json_template() -> str:
        return """[
  {
    "id": 1,
    "module_name": "short_snake_case_name",
    "description": "What this module does, its responsibility, key algorithms or patterns to use.",
    "files_to_create": ["module_name.py", "test_module_name.py"],
    "dependencies": []
  },
  ...
]"""

    def _extra_phases(self) -> str:
        return _ENV_SETUP_PHASE

    def _disk_fallback_paths(self, working_dir: str) -> list[str]:
        return ["decomposition.json", "module_decomposition.json"]

    def _backend_instructions(self, backend: str) -> str:
        if backend == "claude_cli":
            return (
                "Use your own knowledge to design the module architecture — do "
                "NOT search the web. If a .tex file was provided in the requirement, "
                "read it first to extract implementation ideas from the 'Future Work' "
                "and 'Optimizations' sections. Complete BOTH phases: output the JSON "
                "decomposition array first, then set up the environment with "
                "requirements.txt and pip install. End with TASK_COMPLETE."
            )
        return (
            "If the requirement references a .tex file, read it first to extract "
            "implementation-relevant sections (especially 'Future Work' and "
            "'Optimizations'). Complete BOTH phases: output the JSON decomposition "
            "array first, then set up the environment with requirements.txt and pip "
            "install. End with TASK_COMPLETE."
        )

    @staticmethod
    def _fallback_decompose(input_spec: str) -> list[dict[str, Any]]:
        return _fallback_decompose(input_spec)


def _fallback_decompose(spec: str) -> list[dict[str, Any]]:
    """Fallback decomposition.

    For LaTeX proposals, extracts \\section titles as module ideas.
    Otherwise splits on keywords.
    """
    # Detect LaTeX: extract section titles as modules
    if '\\section' in spec or '\\documentclass' in spec:
        sections = re.findall(r'\\section\{([^}]+)\}', spec)
        if sections:
            templates: list[dict[str, Any]] = []
            for i, sec in enumerate(sections[:20]):
                raw = re.sub(r'[^a-z0-9_]', '_', sec.lower())[:50]
                safe_name = raw.strip('_') or f"module_{i+1:02d}"
                templates.append({
                    "id": i + 1,
                    "module_name": safe_name[:40],
                    "description": f"Implement: {sec}",
                    "files_to_create": [f"{safe_name[:40]}.py", f"test_{safe_name[:40]}.py"],
                    "dependencies": [],
                })
            return templates[:20]

    # Non-LaTeX: keyword-based decomposition
    keywords = [s.strip() for s in re.split(r',| and | with |;|\n', spec) if len(s.strip()) >= 3]
    if not keywords:
        keywords = ["core_logic", "cli_interface", "utilities"]

    templates: list[dict[str, Any]] = []
    for i, kw in enumerate(keywords[:20]):
        safe_name = re.sub(r'[^a-z0-9_]', '_', kw.lower().strip().rstrip('.'))[:40]
        templates.append({
            "id": i + 1,
            "module_name": safe_name or f"module_{i+1:02d}",
            "description": f"Implement '{kw}' functionality. Handle all related operations with clean interfaces.",
            "files_to_create": [f"{safe_name}.py", f"test_{safe_name}.py"],
            "dependencies": [],
        })

    # Always add a main entry point
    if templates:
        templates.append({
            "id": len(templates) + 1,
            "module_name": "main",
            "description": "Entry point that integrates all modules. Parse CLI args, wire components together, handle top-level error handling.",
            "files_to_create": ["main.py"],
            "dependencies": [t["module_name"] for t in templates],
        })

    if len(templates) < 2:
        templates.append({
            "id": len(templates) + 1,
            "module_name": "utils",
            "description": "Shared utility functions and constants used across other modules.",
            "files_to_create": ["utils.py", "test_utils.py"],
            "dependencies": [],
        })

    return templates[:20]


def decompose_to_modules(
    input_spec: str, working_dir: str, backend: str = "deepseek",
) -> list[dict[str, Any]]:
    """Decompose a coding task into sub-module specs.

    If input_spec ends with .tex, the file is read first to extract
    implementation-relevant sections. Otherwise it is treated as a user prompt.

    Returns a list of dicts with keys: id, module_name, description,
    files_to_create, dependencies.
    """
    role = CoderPPDecomposerRole()
    return role.execute(working_dir=working_dir, backend=backend, input_spec=input_spec)


# ═══════════════════════════════════════════════════════════════════════════
# Observer (kept as-is — not a decomposer)
# ═══════════════════════════════════════════════════════════════════════════

class ObserverRole(AgentRole):
    """Read all worker outputs and write a progress observation report."""

    agent_name = "coderpp_observer"
    max_steps = 8

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        return ToolRegistry.to_dicts(ToolRegistry.coderpp_decomposer_tools(backend))

    def build_task(self, backend: str, worker_outputs: list[dict[str, Any]] | None = None,
                   sub_tasks: list[dict[str, Any]] | None = None, **context: Any) -> str:
        assert worker_outputs is not None
        lines = []
        for wo in worker_outputs:
            name = wo.get("module_name", "?")
            files = wo.get("files", [])
            log_file = wo.get("log_file", "")
            summary = wo.get("summary", "")[:300]
            lines.append(f"### {name}")
            lines.append(f"  Files: {', '.join(files) if files else '(none)'}")
            if log_file:
                lines.append(f"  Log: {log_file}")
            if summary:
                lines.append(f"  Summary: {summary}")
            lines.append("")
        worker_summary = "\n".join(lines)

        return f"""You are the head agent observing your workers' progress. Read each worker's output files and write a brief observation report.

## Workers
{worker_summary}

## Instructions
1. Read each worker's main module file (`modules/<name>/module.py`) to assess code quality and completeness.
2. Read each worker's log file (`modules/<name>/log.md`) if it exists.
3. Read each worker's test file to check test coverage.
4. Write `OBSERVATIONS.md` at the working directory root with:
   - **Per-Worker Assessment**: For each worker, note: what was implemented, code quality (good/adequate/incomplete), whether tests exist and pass, any obvious issues.
   - **Cross-Cutting Concerns**: Inconsistencies between modules, missing integrations, duplicated logic.
   - **Overall Status**: Summary of worker progress (e.g., "4/5 workers produced complete code with tests, 1 worker missing tests").
   - **Recommendations for Reviewer**: What the reviewer should pay attention to.
5. Output TASK_COMPLETE when done.

Be concise. This is a quick progress check, not a full review."""

    def parse_result(self, result: AgentResult, working_dir: str, **context: Any) -> str:
        obs_path = os.path.join(working_dir, "OBSERVATIONS.md")
        if os.path.exists(obs_path):
            return obs_path
        return ""


def observe_workers(
    worker_outputs: list[dict[str, Any]],
    sub_tasks: list[dict[str, Any]],
    working_dir: str,
    backend: str = "deepseek",
) -> str:
    """Run the head agent as an observer to spy on worker progress.

    Reads all worker outputs and writes OBSERVATIONS.md with per-worker
    assessment, cross-cutting concerns, and recommendations for the reviewer.

    Returns the path to OBSERVATIONS.md, or empty string on failure.
    """
    role = ObserverRole()
    return role.execute(
        working_dir=working_dir,
        backend=backend,
        worker_outputs=worker_outputs,
        sub_tasks=sub_tasks,
    )
