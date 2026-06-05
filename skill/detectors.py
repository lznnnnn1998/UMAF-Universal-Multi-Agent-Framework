"""Domain detector roles for the Skill Summarizer Pipeline.

Four AgentRole subclasses that each analyze a project for skills in a
specific domain:
- PythonDetectorRole: Python version, packages, frameworks, testing, linting
- JSDetectorRole: Node.js version, packages, frameworks, testing, TypeScript
- InfraDetectorRole: Docker, Kubernetes, CI/CD, cloud, IaC
- ConfigDocsDetectorRole: Config formats, documentation, API specs, tooling

Each detector reads ``project_scan.json`` (produced by SkillScannerRole)
and writes a domain-specific JSON report consumed by SkillAggregatorRole.
"""

import json
import os
import re
import subprocess
import sys
from typing import Any

# Ensure repo root is on sys.path so we can import agent.py and tools.py
# __file__ is .../coderpp_output/modules/skill_agent_roles/skill/detectors.py
# Repo root is 5 dirs up: .../universal_multi_agent_framework/
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agent import AgentResult, AgentRole  # noqa: E402
from tools import ToolRegistry  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Helpers shared across detectors
# ═══════════════════════════════════════════════════════════════════════════

def _load_project_scan(working_dir: str) -> dict[str, Any] | None:
    """Load project_scan.json from the working directory."""
    path = os.path.join(working_dir, "project_scan.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


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


def _run_cmd(cmd: str, cwd: str = ".", timeout: int = 30) -> tuple[str, int]:
    """Run a shell command and return (output, returncode)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=cwd,
        )
        return (result.stdout + "\n" + result.stderr).strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "Timeout", -1
    except OSError as e:
        return f"Error: {e}", -2


def _detect_framework(frameworks: list[tuple[str, list[str]]],
                      all_files: list[str]) -> list[dict[str, Any]]:
    """Match a list of (framework_name, [indicator_files]) against the
    project files and return detected frameworks."""
    detected: list[dict[str, Any]] = []
    for fw_name, indicators in frameworks:
        matched = []
        for ind in indicators:
            for fpath in all_files:
                if ind in fpath or os.path.basename(fpath) == ind:
                    if ind not in matched:
                        matched.append(ind)
        if matched:
            detected.append({
                "name": fw_name,
                "indicators": matched,
                "confidence": "high" if len(matched) >= 2 else "medium",
            })
    return detected


def _format_scan_summary(project_scan: dict[str, Any] | None) -> str:
    """Build an inline summary of the project scan for detector prompts.

    When the scanner has already run, this summary is embedded directly in
    the prompt so detectors don't need to read ``project_scan.json`` from
    disk.  Falls back to instructing the agent to read the file if no scan
    data is provided.
    """
    if not project_scan or not project_scan.get("file_categories"):
        return (
            "\n## Project Scan\n"
            "No pre-computed project scan is available. "
            "Read `project_scan.json` from the working directory to "
            "understand the project structure before proceeding.\n"
        )

    cats = project_scan.get("file_categories", {})
    total = project_scan.get("total_files", len(sum(cats.values(), [])))
    dirs = project_scan.get("total_dirs", "?")

    lines = [
        "\n## Project Scan (pre-computed — NO need to read from disk)",
        f"**Total files**: {total}  |  **Directories**: {dirs}",
        "",
        "### File Categories",
    ]
    for cat_name, label in [
        ("source", "Source"), ("test", "Test"), ("config", "Config"),
        ("docs", "Docs"), ("build", "Build/CI"), ("ci", "CI"),
        ("other", "Other"),
    ]:
        files = cats.get(cat_name, [])
        if files:
            preview = ", ".join(f"`{f}`" for f in files[:8])
            extra = f" ... (+{len(files) - 8} more)" if len(files) > 8 else ""
            lines.append(f"- **{label}** ({len(files)}): {preview}{extra}")

    # Key directories
    dir_list = project_scan.get("directories", [])
    if dir_list:
        lines.append(f"\n### Key Directories\n{', '.join(f'`{d}`' for d in dir_list[:12])}")

    lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Base detector — reduces boilerplate across the four domain detectors
# ═══════════════════════════════════════════════════════════════════════════

class _BaseDetectorRole(AgentRole):
    """Shared base for the four domain detector roles.

    Provides common parse_result logic (agent messages → disk → fallback)
    so subclasses only need to define tools, prompts, and fallback logic.
    """

    max_steps: int = 12
    output_file: str = ""
    domain: str = ""

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        return ToolRegistry.to_dicts(ToolRegistry.skill_detector_tools())

    def parse_result(self, result: AgentResult, working_dir: str,
                     project_dir: str = ".", **context: Any) -> dict[str, Any]:
        """Extract domain report: agent messages → disk file → fallback."""
        report: dict[str, Any] = {}

        # 1. Agent response
        for msg in reversed(result.messages):
            content = msg.content if hasattr(msg, "content") else str(msg)
            json_str = _extract_json_object(content)
            if json_str:
                try:
                    parsed = json.loads(json_str)
                    if "domain" in parsed or "skills" in parsed:
                        report = parsed
                        break
                except json.JSONDecodeError:
                    continue

        # 2. Disk file
        if not report and self.output_file:
            path = os.path.join(working_dir, self.output_file)
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        parsed = json.load(f)
                    if isinstance(parsed, dict):
                        report = parsed
                except (json.JSONDecodeError, OSError):
                    pass

        # 3. Fallback
        if not report:
            report = self._fallback_detect(project_dir, working_dir)

        return report

    def _fallback_detect(self, project_dir: str,
                         working_dir: str) -> dict[str, Any]:
        """Override in subclasses for domain-specific fallback detection."""
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
# PythonDetectorRole
# ═══════════════════════════════════════════════════════════════════════════

_PYTHON_FRAMEWORKS = [
    ("Django", ["manage.py", "django", "wsgi.py", "asgi.py", "urls.py"]),
    ("Flask", ["flask", "app.py", "wsgi.py"]),
    ("FastAPI", ["fastapi", "main.py"]),
    ("Pyramid", ["pyramid", "pserve"]),
    ("Tornado", ["tornado"]),
    ("aiohttp", ["aiohttp"]),
    ("Sanic", ["sanic"]),
    ("Streamlit", ["streamlit"]),
    ("Gradio", ["gradio"]),
    ("SQLAlchemy", ["sqlalchemy", "alembic"]),
    ("Pydantic", ["pydantic"]),
    ("Celery", ["celery"]),
    ("Pytest", ["pytest", "conftest.py", "pytest.ini"]),
    ("unittest", ["unittest", "test_"]),
    ("Sphinx", ["sphinx", "conf.py"]),
    ("Click", ["click"]),
    ("Typer", ["typer"]),
    ("Rich", ["rich"]),
    ("Poetry", ["poetry.lock", "pyproject.toml"]),
    ("pipenv", ["Pipfile"]),
    ("hatch", ["hatch.toml"]),
]

_PYTHON_LINTING = [
    ("ruff", ["ruff.toml", "ruff", ".ruff.toml"]),
    ("black", ["black", "pyproject.toml"]),
    ("flake8", ["flake8", ".flake8", "setup.cfg"]),
    ("pylint", ["pylint", ".pylintrc"]),
    ("mypy", ["mypy", "mypy.ini", ".mypy.ini", "pyproject.toml"]),
    ("isort", ["isort", ".isort.cfg", "pyproject.toml"]),
    ("bandit", ["bandit"]),
    ("pre-commit", [".pre-commit-config.yaml"]),
]

_PYTHON_DATA_SCIENCE = [
    ("numpy", ["numpy"]),
    ("pandas", ["pandas"]),
    ("scikit-learn", ["sklearn", "scikit-learn", "scikit_learn"]),
    ("PyTorch", ["torch", "pytorch"]),
    ("TensorFlow", ["tensorflow", "tf."]),
    ("Keras", ["keras"]),
    ("Jupyter", ["jupyter", ".ipynb", "ipynb"]),
    ("matplotlib", ["matplotlib"]),
    ("seaborn", ["seaborn"]),
    ("plotly", ["plotly"]),
    ("scipy", ["scipy"]),
    ("statsmodels", ["statsmodels"]),
    ("xgboost", ["xgboost"]),
    ("lightgbm", ["lightgbm"]),
    ("transformers", ["transformers", "huggingface"]),
    ("spaCy", ["spacy"]),
    ("NLTK", ["nltk"]),
    ("OpenCV", ["cv2", "opencv"]),
    ("Pillow", ["PIL", "Pillow"]),
    ("Dask", ["dask"]),
    ("Ray", ["ray"]),
    ("MLflow", ["mlflow"]),
    ("Weights & Biases", ["wandb"]),
    ("LangChain", ["langchain"]),
    ("LangGraph", ["langgraph"]),
    ("LlamaIndex", ["llama_index", "llamaindex"]),
]


class PythonDetectorRole(_BaseDetectorRole):
    """Detect Python ecosystem skills: version, packages, frameworks, testing,
    linting, type checking, and data science tools."""

    agent_name: str = "python_detector"
    output_file: str = "python_report.json"
    domain: str = "Python"

    def build_task(self, backend: str, project_dir: str = ".",
                   working_dir: str = ".", project_scan: dict[str, Any] | None = None,
                   **context: Any) -> str:
        """Build the Python detection prompt."""
        scan_summary = _format_scan_summary(project_scan)
        common = (
            f"You are a Python ecosystem analyst. Your job is to analyze a "
            f"project directory and identify all Python-related skills, tools, "
            f"and technologies in use.\n\n"
            f"## Project Directory\n{project_dir}\n"
            f"{scan_summary}"
            f"## Instructions\n"
            f"1. The project structure is shown above. "
            f"Focus your analysis on Python-related files and tools.\n"
            f"2. Check for Python version indicators:\n"
            f"   - Run `python3 --version` or `python --version`\n"
            f"   - Look for `.python-version` file\n"
            f"   - Look for `python_requires` in setup.cfg/pyproject.toml\n"
            f"3. Check for package management:\n"
            f"   - `requirements.txt`, `requirements-dev.txt`\n"
            f"   - `setup.py`, `setup.cfg`\n"
            f"   - `pyproject.toml` (setuptools, Poetry, hatch)\n"
            f"   - `Pipfile`, `Pipfile.lock`\n"
            f"4. Run `pip list 2>/dev/null || python3 -m pip list 2>/dev/null` "
            f"to enumerate installed packages (if pip is available).\n"
            f"5. Analyze the file listing for:\n"
            f"   - **Web frameworks**: Django, Flask, FastAPI, Pyramid, etc.\n"
            f"   - **Testing**: pytest, unittest, nose, tox\n"
            f"   - **Linting & formatting**: ruff, black, flake8, pylint, mypy, "
            f"isort, bandit\n"
            f"   - **Data science**: numpy, pandas, scikit-learn, PyTorch, "
            f"TensorFlow, Jupyter, matplotlib, etc.\n"
            f"   - **AI/ML frameworks**: transformers, spaCy, LangChain, "
            f"LlamaIndex, etc.\n"
            f"6. For each detected skill, assign a **proficiency level**:\n"
            f"   - `expert`: extensive configuration files, large codebase, "
            f"multiple advanced features\n"
            f"   - `advanced`: moderate configuration, multiple files using it\n"
            f"   - `intermediate`: basic usage, some configuration present\n"
            f"   - `beginner`: minimal usage, just listed as a dependency\n\n"
            f"7. Write your findings as JSON to `python_report.json`:\n"
            f"```json\n"
            f"{{\n"
            f'  "domain": "Python",\n'
            f'  "version": {{"detected": "3.11", "source": ".python-version"}},\n'
            f'  "package_manager": ["pip", "poetry"],\n'
            f'  "skills": [\n'
            f'    {{\n'
            f'      "name": "Django",\n'
            f'      "category": "web_framework",\n'
            f'      "proficiency": "advanced",\n'
            f'      "evidence": ["manage.py", "settings.py"],\n'
            f'      "version_hint": ">=4.0"\n'
            f'    }}\n'
            f'  ],\n'
            f'  "installed_packages_preview": ["pkg1==1.0", ...],\n'
            f'  "total_python_files": <int>,\n'
            f'  "total_test_files": <int>\n'
            f"}}\n"
            f"```\n\n"
            f"Working directory: {working_dir}"
        )

        if backend == "claude_cli":
            backend_note = (
                "\n\nUse your own knowledge — do NOT search the web. "
                "Read the project scan, run available commands, and write "
                "python_report.json. Output TASK_COMPLETE when done."
            )
        else:
            backend_note = (
                "\n\nRead the project scan, run available commands, write "
                "python_report.json, then output TASK_COMPLETE."
            )

        return common + backend_note

    def _fallback_detect(self, project_dir: str = ".",
                         working_dir: str = ".") -> dict[str, Any]:
        """Deterministic Python detection without LLM."""
        scan = _load_project_scan(working_dir) or {}
        categories = scan.get("file_categories", {})
        all_files = (
            categories.get("source", []) +
            categories.get("config", []) +
            categories.get("test", []) +
            categories.get("other", [])
        )
        if not all_files:
            # Try direct find
            try:
                r = subprocess.run(
                    f"find {project_dir} -type f -not -path '*/.git/*' "
                    f"-not -path '*/node_modules/*' -not -path '*/__pycache__/*' | head -500",
                    shell=True, capture_output=True, text=True,
                    timeout=15, cwd=working_dir,
                )
                all_files = [f.strip() for f in r.stdout.strip().split("\n") if f.strip()]
            except (subprocess.TimeoutExpired, OSError):
                all_files = []

        py_files = [f for f in all_files if f.endswith(".py")]
        test_files = [f for f in all_files if
                      os.path.basename(f).startswith("test_") or
                      os.path.basename(f).endswith("_test.py") or
                      "test_" in f]

        # Detect Python version
        version_info: dict[str, str] = {}
        version_info["detected"] = "unknown"

        # Check .python-version
        pv_path = os.path.join(working_dir, ".python-version")
        if not os.path.exists(pv_path):
            pv_path = os.path.join(working_dir, project_dir, ".python-version")
        if os.path.exists(pv_path):
            try:
                with open(pv_path) as f:
                    version_info["detected"] = f.read().strip()
                version_info["source"] = ".python-version"
            except OSError:
                pass

        # Try running python3 --version
        if version_info.get("detected") == "unknown":
            out, rc = _run_cmd("python3 --version", cwd=working_dir)
            if rc == 0:
                m = re.search(r"(\d+\.\d+\.?\d*)", out)
                if m:
                    version_info["detected"] = m.group(1)
                    version_info["source"] = "python3 --version"

        # Detect package manager
        pkg_managers: list[str] = []
        for f in all_files:
            bn = os.path.basename(f)
            if bn == "requirements.txt" and "pip" not in pkg_managers:
                pkg_managers.append("pip")
            if bn == "Pipfile" and "pipenv" not in pkg_managers:
                pkg_managers.append("pipenv")
            if bn == "poetry.lock" and "poetry" not in pkg_managers:
                pkg_managers.append("poetry")
            if bn == "pyproject.toml" and "pip" not in pkg_managers:
                # Could be any modern tool, include pip as default
                pkg_managers.append("pip")
            if bn == "setup.py" and "setuptools" not in pkg_managers:
                pkg_managers.append("setuptools")

        # Detect skills
        skills: list[dict[str, Any]] = []

        # Frameworks
        for fw_name, indicators in _PYTHON_FRAMEWORKS:
            matched = []
            for ind in indicators:
                for fpath in all_files:
                    bn = os.path.basename(fpath)
                    if ind.lower() in bn.lower() or ind.lower() in fpath.lower():
                        if ind not in matched:
                            matched.append(ind)
            if matched:
                skills.append({
                    "name": fw_name,
                    "category": "web_framework" if fw_name in
                        ("Django", "Flask", "FastAPI", "Pyramid", "Tornado",
                         "aiohttp", "Sanic", "Streamlit", "Gradio")
                        else "testing" if fw_name in ("Pytest", "unittest")
                        else "linting" if fw_name in ("ruff", "black", "flake8",
                            "pylint", "mypy", "isort", "bandit", "pre-commit")
                        else "data_science" if fw_name in (
                            "numpy", "pandas", "scikit-learn", "PyTorch",
                            "TensorFlow", "Keras", "Jupyter", "matplotlib",
                            "seaborn", "plotly", "scipy", "statsmodels",
                            "xgboost", "lightgbm", "transformers", "spaCy",
                            "NLTK", "OpenCV", "Pillow", "Dask", "Ray",
                            "MLflow", "Weights & Biases", "LangChain",
                            "LangGraph", "LlamaIndex")
                        else "tooling",
                    "proficiency": "advanced" if len(matched) >= 3
                        else "intermediate" if len(matched) >= 2
                        else "beginner",
                    "evidence": matched,
                    "version_hint": "",
                })

        return {
            "domain": "Python",
            "version": version_info,
            "package_manager": pkg_managers if pkg_managers else ["unknown"],
            "skills": skills,
            "installed_packages_preview": [],
            "total_python_files": len(py_files),
            "total_test_files": len(test_files),
            "_fallback": True,
        }


# ═══════════════════════════════════════════════════════════════════════════
# JSDetectorRole
# ═══════════════════════════════════════════════════════════════════════════

_JS_FRAMEWORKS = [
    ("React", ["react", "jsx", "tsx", ".jsx", ".tsx"]),
    ("Vue.js", ["vue", ".vue"]),
    ("Angular", ["angular", "@angular", "angular.json"]),
    ("Next.js", ["next.config", "next-env", "next", ".next"]),
    ("Nuxt", ["nuxt.config", "nuxt"]),
    ("Svelte", ["svelte", ".svelte"]),
    ("Express", ["express"]),
    ("Fastify", ["fastify"]),
    ("NestJS", ["@nestjs", "nest-cli"]),
    ("Gatsby", ["gatsby-config", "gatsby"]),
    ("Remix", ["remix.config", "@remix-run"]),
    ("Astro", ["astro.config", "astro"]),
    ("Electron", ["electron"]),
    ("React Native", ["react-native", "@react-native"]),
    ("jQuery", ["jquery"]),
    ("Redux", ["redux", "@reduxjs"]),
    ("MobX", ["mobx"]),
    ("Zustand", ["zustand"]),
    ("Tailwind CSS", ["tailwind.config", "tailwind"]),
    ("Bootstrap", ["bootstrap"]),
    ("Material UI", ["@mui", "@material-ui", "material-ui"]),
    ("Chakra UI", ["@chakra-ui", "chakra"]),
    ("Ant Design", ["antd", "ant-design"]),
    ("D3.js", ["d3", "d3.js"]),
    ("Three.js", ["three"]),
    ("Chart.js", ["chart.js", "chartjs"]),
    ("Socket.io", ["socket.io", "socket.io-client"]),
]

_JS_TESTING = [
    ("Jest", ["jest.config", "jest", ".test.", "spec.", "__tests__"]),
    ("Mocha", ["mocha", ".mocharc"]),
    ("Vitest", ["vitest.config", "vitest"]),
    ("Cypress", ["cypress", "cypress.config"]),
    ("Playwright", ["playwright.config", "@playwright"]),
    ("Testing Library", ["@testing-library", "testing-library"]),
    ("Storybook", ["storybook", ".storybook"]),
]

_JS_BUILD = [
    ("Webpack", ["webpack.config", "webpack"]),
    ("Vite", ["vite.config", "vite"]),
    ("Rollup", ["rollup.config", "rollup"]),
    ("esbuild", ["esbuild"]),
    ("Parcel", ["parcel", ".parcelrc"]),
    ("Turbopack", ["turbopack"]),
    ("Babel", [".babelrc", "babel.config", "@babel"]),
    ("SWC", [".swcrc", "@swc"]),
    ("ESLint", [".eslintrc", "eslint.config", "eslint"]),
    ("Prettier", [".prettierrc", "prettier.config", "prettier"]),
    ("TypeScript", ["tsconfig.json", "tsconfig", ".ts", ".tsx"]),
    ("pnpm", ["pnpm-lock.yaml", "pnpm-workspace"]),
    ("Yarn", ["yarn.lock", ".yarn"]),
    ("npm", ["package-lock.json", "package.json"]),
    ("Nx", ["nx.json", "@nrwl"]),
    ("Turborepo", ["turbo.json"]),
    ("Lerna", ["lerna.json"]),
]


class JSDetectorRole(_BaseDetectorRole):
    """Detect JavaScript/Node.js ecosystem skills: version, packages,
    frameworks, testing, build tools, and TypeScript."""

    agent_name: str = "js_detector"
    output_file: str = "javascript_report.json"
    domain: str = "JavaScript"

    def build_task(self, backend: str, project_dir: str = ".",
                   working_dir: str = ".", project_scan: dict[str, Any] | None = None,
                   **context: Any) -> str:
        """Build the JS/Node.js detection prompt."""
        scan_summary = _format_scan_summary(project_scan)
        common = (
            f"You are a JavaScript/Node.js ecosystem analyst. Your job is to "
            f"analyze a project directory and identify all JavaScript-related "
            f"skills, tools, and technologies in use.\n\n"
            f"## Project Directory\n{project_dir}\n"
            f"{scan_summary}"
            f"## Instructions\n"
            f"1. The project structure is shown above. "
            f"Focus your analysis on JavaScript/Node.js-related files and tools.\n"
            f"2. Check for Node.js version:\n"
            f"   - Run `node --version`\n"
            f"   - Look for `.nvmrc` or `.node-version`\n"
            f"   - Check `engines` field in package.json\n"
            f"3. Check for package manager:\n"
            f"   - `package-lock.json` (npm)\n"
            f"   - `yarn.lock` + `.yarn/` (Yarn)\n"
            f"   - `pnpm-lock.yaml` (pnpm)\n"
            f"4. Check for TypeScript: look for `tsconfig.json`, `.ts`, `.tsx` files\n"
            f"5. Analyze for:\n"
            f"   - **Frontend frameworks**: React, Vue, Angular, Svelte, Next.js, etc.\n"
            f"   - **Backend frameworks**: Express, Fastify, NestJS, etc.\n"
            f"   - **Testing**: Jest, Mocha, Vitest, Cypress, Playwright\n"
            f"   - **Build tools**: Webpack, Vite, Rollup, esbuild, Babel, SWC\n"
            f"   - **Linting**: ESLint, Prettier\n"
            f"   - **State management, UI libs, data viz**\n"
            f"6. Assign proficiency levels (expert/advanced/intermediate/beginner).\n\n"
            f"7. Write your findings to `javascript_report.json`:\n"
            f"```json\n"
            f"{{\n"
            f'  "domain": "JavaScript",\n'
            f'  "runtime": {{"node_version": "20.x", "source": ".nvmrc"}},\n'
            f'  "package_manager": "npm",\n'
            f'  "typescript": {{"used": true, "config": "tsconfig.json"}},\n'
            f'  "skills": [\n'
            f'    {{\n'
            f'      "name": "React",\n'
            f'      "category": "frontend_framework",\n'
            f'      "proficiency": "advanced",\n'
            f'      "evidence": ["next.config.js", "pages/*.tsx"],\n'
            f'      "version_hint": "18.x"\n'
            f'    }}\n'
            f'  ],\n'
            f'  "total_js_files": <int>,\n'
            f'  "total_ts_files": <int>,\n'
            f'  "total_test_files": <int>\n'
            f"}}\n"
            f"```\n\n"
            f"Working directory: {working_dir}"
        )

        if backend == "claude_cli":
            backend_note = (
                "\n\nUse your own knowledge — do NOT search the web. "
                "Read project_scan.json, run available version commands, "
                "and write javascript_report.json. Output TASK_COMPLETE."
            )
        else:
            backend_note = (
                "\n\nRead project_scan.json, run available version commands, "
                "write javascript_report.json, output TASK_COMPLETE."
            )

        return common + backend_note

    def _fallback_detect(self, project_dir: str = ".",
                         working_dir: str = ".") -> dict[str, Any]:
        """Deterministic JS/Node.js detection without LLM."""
        scan = _load_project_scan(working_dir) or {}
        categories = scan.get("file_categories", {})
        all_files = (
            categories.get("source", []) +
            categories.get("config", []) +
            categories.get("test", []) +
            categories.get("other", [])
        )
        if not all_files:
            try:
                r = subprocess.run(
                    f"find {project_dir} -type f -not -path '*/.git/*' "
                    f"-not -path '*/node_modules/*' | head -500",
                    shell=True, capture_output=True, text=True,
                    timeout=15, cwd=working_dir,
                )
                all_files = [f.strip() for f in r.stdout.strip().split("\n") if f.strip()]
            except (subprocess.TimeoutExpired, OSError):
                all_files = []

        js_files = [f for f in all_files if f.endswith((".js", ".jsx", ".mjs", ".cjs"))]
        ts_files = [f for f in all_files if f.endswith((".ts", ".tsx", ".mts", ".cts"))]
        test_files = [f for f in all_files if ".test." in f or "spec." in f or
                      "__tests__" in f or "/test/" in f or "/tests/" in f]

        # Detect Node.js version
        node_version: dict[str, str] = {"detected": "unknown"}
        for nv_file in [".nvmrc", ".node-version"]:
            p = os.path.join(working_dir, nv_file)
            if not os.path.exists(p):
                p = os.path.join(working_dir, project_dir, nv_file)
            if os.path.exists(p):
                try:
                    with open(p) as f:
                        node_version["detected"] = f.read().strip().lstrip("v")
                    node_version["source"] = nv_file
                    break
                except OSError:
                    pass

        if node_version["detected"] == "unknown":
            out, rc = _run_cmd("node --version", cwd=working_dir)
            if rc == 0:
                m = re.search(r"v?(\d+\.\d+\.\d+)", out)
                if m:
                    node_version["detected"] = m.group(1)
                    node_version["source"] = "node --version"

        # Detect package manager
        pkg_manager = "unknown"
        for f in all_files:
            bn = os.path.basename(f)
            if bn == "pnpm-lock.yaml":
                pkg_manager = "pnpm"
                break
            elif bn == "yarn.lock":
                pkg_manager = "yarn"
                break
            elif bn == "package-lock.json":
                pkg_manager = "npm"
                break
            elif bn == "package.json":
                pkg_manager = "npm"  # default

        # Check TypeScript
        ts_used = len(ts_files) > 0
        ts_config = ""
        for f in all_files:
            bn = os.path.basename(f)
            if bn in ("tsconfig.json", "tsconfig.base.json", "tsconfig.build.json"):
                ts_config = bn
                ts_used = True
                break

        # Detect skills
        skills: list[dict[str, Any]] = []

        all_indicators = _JS_FRAMEWORKS + _JS_TESTING + _JS_BUILD
        for skill_name, indicators in all_indicators:
            matched = []
            for ind in indicators:
                for fpath in all_files:
                    bn = os.path.basename(fpath)
                    if ind.lower() in bn.lower() or ind.lower() in fpath.lower():
                        if ind not in matched:
                            matched.append(ind)
            if matched:
                # Determine category
                fw_names = {fw[0] for fw in _JS_FRAMEWORKS}
                test_names = {t[0] for t in _JS_TESTING}
                if skill_name in fw_names:
                    cat = "frontend_framework" if skill_name not in ("Express", "Fastify", "NestJS") else "backend_framework"
                elif skill_name in test_names:
                    cat = "testing"
                else:
                    cat = "build_tooling"
                skills.append({
                    "name": skill_name,
                    "category": cat,
                    "proficiency": "advanced" if len(matched) >= 3
                        else "intermediate" if len(matched) >= 2
                        else "beginner",
                    "evidence": matched,
                    "version_hint": "",
                })

        return {
            "domain": "JavaScript",
            "runtime": node_version,
            "package_manager": pkg_manager,
            "typescript": {"used": ts_used, "config": ts_config},
            "skills": skills,
            "total_js_files": len(js_files),
            "total_ts_files": len(ts_files),
            "total_test_files": len(test_files),
            "_fallback": True,
        }


# ═══════════════════════════════════════════════════════════════════════════
# InfraDetectorRole
# ═══════════════════════════════════════════════════════════════════════════

_INFRA_INDICATORS = [
    # Docker
    ("Docker", ["Dockerfile", "docker-compose", ".dockerignore", "docker"]),
    ("Docker Compose", ["docker-compose.yml", "docker-compose.yaml"]),
    # Kubernetes
    ("Kubernetes", ["k8s", "kubernetes", "deployment.yaml", "service.yaml",
                    "ingress.yaml", "helm", "kustomization"]),
    ("Helm", ["helm", "Chart.yaml", "values.yaml"]),
    # CI/CD
    ("GitHub Actions", [".github/workflows", "action.yml"]),
    ("GitLab CI", [".gitlab-ci.yml"]),
    ("Jenkins", ["Jenkinsfile"]),
    ("Travis CI", [".travis.yml"]),
    ("CircleCI", [".circleci", "config.yml"]),
    ("Azure Pipelines", ["azure-pipelines.yml", "azure-pipelines"]),
    ("Bitbucket Pipelines", ["bitbucket-pipelines.yml"]),
    # Cloud providers
    ("AWS", ["aws", "serverless.yml", "samconfig.toml", ".aws"]),
    ("GCP", ["gcp", "gcloud", ".gcloudignore", "app.yaml", "cloudbuild"]),
    ("Azure", ["azure", "azurerm", ".azure"]),
    # IaC
    ("Terraform", ["terraform", ".tf", ".tfvars", "terraform.tfstate"]),
    ("Pulumi", ["pulumi", "Pulumi.yaml"]),
    ("CloudFormation", ["cloudformation", ".template", "cfn-"]),
    ("Ansible", ["ansible", "playbook.yml", "playbook.yaml", "ansible.cfg"]),
    ("Chef", ["chef", "Berksfile", "Policyfile"]),
    ("Puppet", ["puppet", "Puppetfile"]),
    ("Vagrant", ["Vagrantfile"]),
    # Monitoring / Observability
    ("Prometheus", ["prometheus", "prometheus.yml", "alertmanager"]),
    ("Grafana", ["grafana", "dashboard.json"]),
    ("OpenTelemetry", ["opentelemetry", "otel"]),
    ("Datadog", ["datadog", "datadog-agent"]),
    ("Sentry", ["sentry"]),
    ("ELK Stack", ["elasticsearch", "logstash", "kibana", "filebeat"]),
    # Service Mesh
    ("Istio", ["istio", "virtual-service", "destination-rule"]),
    ("Linkerd", ["linkerd"]),
    # Infrastructure tools
    ("Nginx", ["nginx.conf", "nginx", "sites-available"]),
    ("Apache", ["httpd.conf", "apache", ".htaccess"]),
    ("HAProxy", ["haproxy", "haproxy.cfg"]),
    ("Redis", ["redis", "redis.conf"]),
    ("PostgreSQL", ["postgresql", "pg_hba.conf", "postgres"]),
    ("MySQL", ["mysql", "my.cnf"]),
    ("MongoDB", ["mongodb", "mongod", "mongod.conf"]),
    ("RabbitMQ", ["rabbitmq", "rabbitmq.conf"]),
    ("Kafka", ["kafka", "server.properties"]),
]


class InfraDetectorRole(_BaseDetectorRole):
    """Detect infrastructure & DevOps skills: Docker, Kubernetes, CI/CD,
    cloud providers, IaC tools, monitoring, databases, and service mesh."""

    agent_name: str = "infra_detector"
    output_file: str = "infrastructure_report.json"
    domain: str = "Infrastructure"

    def build_task(self, backend: str, project_dir: str = ".",
                   working_dir: str = ".", project_scan: dict[str, Any] | None = None,
                   **context: Any) -> str:
        """Build the infrastructure detection prompt."""
        scan_summary = _format_scan_summary(project_scan)
        common = (
            f"You are an infrastructure & DevOps analyst. Your job is to "
            f"analyze a project directory and identify all infrastructure and "
            f"DevOps tools and technologies in use.\n\n"
            f"## Project Directory\n{project_dir}\n"
            f"{scan_summary}"
            f"## Instructions\n"
            f"1. The project structure is shown above. "
            f"Focus your analysis on infrastructure and DevOps files.\n"
            f"2. Analyze the file listing for infrastructure indicators:\n"
            f"   - **Containerization**: Docker, Docker Compose, Podman\n"
            f"   - **Orchestration**: Kubernetes, Helm, Kustomize, OpenShift\n"
            f"   - **CI/CD**: GitHub Actions, GitLab CI, Jenkins, CircleCI, "
            f"Travis CI, Azure Pipelines, Bitbucket Pipelines\n"
            f"   - **Cloud Providers**: AWS (CloudFormation, SAM, CDK), "
            f"GCP (Cloud Build, Deployment Manager), Azure (ARM, Bicep)\n"
            f"   - **IaC**: Terraform, Pulumi, Ansible, Chef, Puppet, Vagrant\n"
            f"   - **Monitoring**: Prometheus, Grafana, OpenTelemetry, Datadog, "
            f"Sentry, ELK Stack\n"
            f"   - **Service Mesh**: Istio, Linkerd, Consul\n"
            f"   - **Web Servers**: Nginx, Apache, HAProxy, Caddy, Traefik\n"
            f"   - **Databases**: PostgreSQL, MySQL, MongoDB, Redis, "
            f"RabbitMQ, Kafka, Elasticsearch\n"
            f"3. For each detected technology, assign a proficiency level.\n\n"
            f"4. Write findings to `infrastructure_report.json`:\n"
            f"```json\n"
            f"{{\n"
            f'  "domain": "Infrastructure",\n'
            f'  "skills": [\n'
            f'    {{\n'
            f'      "name": "Docker",\n'
            f'      "category": "containerization",\n'
            f'      "proficiency": "advanced",\n'
            f'      "evidence": ["Dockerfile", "docker-compose.yml"],\n'
            f'      "version_hint": ""\n'
            f'    }}\n'
            f'  ]\n'
            f"}}\n"
            f"```\n\n"
            f"Working directory: {working_dir}"
        )

        if backend == "claude_cli":
            backend_note = (
                "\n\nUse your own knowledge — do NOT search the web. "
                "Read project_scan.json, analyze files, write "
                "infrastructure_report.json. Output TASK_COMPLETE."
            )
        else:
            backend_note = (
                "\n\nRead project_scan.json, analyze files, write "
                "infrastructure_report.json, output TASK_COMPLETE."
            )

        return common + backend_note

    def _fallback_detect(self, project_dir: str = ".",
                         working_dir: str = ".") -> dict[str, Any]:
        """Deterministic infrastructure detection without LLM."""
        scan = _load_project_scan(working_dir) or {}
        categories = scan.get("file_categories", {})
        all_files = (
            categories.get("source", []) +
            categories.get("config", []) +
            categories.get("build", []) +
            categories.get("ci", []) +
            categories.get("other", [])
        )
        if not all_files:
            try:
                r = subprocess.run(
                    f"find {project_dir} -type f -not -path '*/.git/*' "
                    f"-not -path '*/node_modules/*' -not -path '*/__pycache__/*' "
                    f"| head -500",
                    shell=True, capture_output=True, text=True,
                    timeout=15, cwd=working_dir,
                )
                all_files = [f.strip() for f in r.stdout.strip().split("\n") if f.strip()]
            except (subprocess.TimeoutExpired, OSError):
                all_files = []

        skills: list[dict[str, Any]] = []
        for skill_name, indicators in _INFRA_INDICATORS:
            matched = []
            for ind in indicators:
                for fpath in all_files:
                    bn = os.path.basename(fpath)
                    if ind.lower() in bn.lower() or ind.lower() in fpath.lower():
                        if ind not in matched:
                            matched.append(ind)
            if matched:
                # Determine category
                containerization = {"Docker", "Docker Compose"}
                orchestration = {"Kubernetes", "Helm"}
                cicd = {"GitHub Actions", "GitLab CI", "Jenkins", "Travis CI",
                        "CircleCI", "Azure Pipelines", "Bitbucket Pipelines"}
                cloud = {"AWS", "GCP", "Azure"}
                iac = {"Terraform", "Pulumi", "CloudFormation", "Ansible",
                       "Chef", "Puppet", "Vagrant"}
                monitoring = {"Prometheus", "Grafana", "OpenTelemetry",
                              "Datadog", "Sentry", "ELK Stack"}
                service_mesh = {"Istio", "Linkerd"}
                web_server = {"Nginx", "Apache", "HAProxy"}
                database = {"Redis", "PostgreSQL", "MySQL", "MongoDB",
                            "RabbitMQ", "Kafka"}

                if skill_name in containerization:
                    cat = "containerization"
                elif skill_name in orchestration:
                    cat = "orchestration"
                elif skill_name in cicd:
                    cat = "ci_cd"
                elif skill_name in cloud:
                    cat = "cloud"
                elif skill_name in iac:
                    cat = "infrastructure_as_code"
                elif skill_name in monitoring:
                    cat = "monitoring"
                elif skill_name in service_mesh:
                    cat = "service_mesh"
                elif skill_name in web_server:
                    cat = "web_server"
                elif skill_name in database:
                    cat = "database"
                else:
                    cat = "other"

                skills.append({
                    "name": skill_name,
                    "category": cat,
                    "proficiency": "advanced" if len(matched) >= 3
                        else "intermediate" if len(matched) >= 2
                        else "beginner",
                    "evidence": matched,
                    "version_hint": "",
                })

        return {
            "domain": "Infrastructure",
            "skills": skills,
            "_fallback": True,
        }


# ═══════════════════════════════════════════════════════════════════════════
# ConfigDocsDetectorRole
# ═══════════════════════════════════════════════════════════════════════════

_CONFIG_FORMATS = [
    ("YAML", [".yaml", ".yml"]),
    ("TOML", [".toml", "pyproject.toml", "Cargo.toml"]),
    ("JSON", [".json", "package.json", "tsconfig.json"]),
    ("INI/CFG", [".ini", ".cfg", ".conf", "setup.cfg"]),
    ("ENV", [".env", ".env.example", ".env.template"]),
    ("XML", [".xml", "pom.xml", "web.config"]),
]

_DOC_TYPES = [
    ("README", ["readme.md", "readme.rst", "readme.txt", "README"]),
    ("CONTRIBUTING", ["contributing.md", "contributing.rst"]),
    ("CHANGELOG", ["changelog.md", "changelog.rst", "CHANGES", "HISTORY"]),
    ("LICENSE", ["license", "license.md", "license.txt", "COPYING"]),
    ("CODE_OF_CONDUCT", ["code_of_conduct", "code-of-conduct"]),
    ("SECURITY", ["security.md", "security.txt"]),
    ("Documentation Directory", ["docs/", "doc/", "documentation/"]),
    ("Wiki", ["wiki/", ".wiki/"]),
]

_API_SPECS = [
    ("OpenAPI/Swagger", ["openapi", "swagger", "openapi.json", "openapi.yaml",
                         "swagger.json", "swagger.yaml"]),
    ("GraphQL", ["graphql", ".graphql", ".gql", "schema.graphql"]),
    ("gRPC", [".proto", "grpc", "protobuf"]),
    ("AsyncAPI", ["asyncapi", "asyncapi.yaml", "asyncapi.json"]),
    ("JSON Schema", ["schema.json", ".schema.json"]),
    ("RAML", [".raml"]),
]

_TOOLING = [
    ("EditorConfig", [".editorconfig"]),
    ("Git", [".gitignore", ".gitattributes", ".gitmodules"]),
    ("Pre-commit", [".pre-commit-config.yaml", ".pre-commit-hooks.yaml"]),
    ("Husky", [".husky", "husky"]),
    ("Commitlint", ["commitlint", "commitlint.config"]),
    ("Commitizen", ["cz", "commitizen", ".cz.toml"]),
    ("Semantic Release", [".releaserc", "release.config", "semantic-release"]),
    ("Renovate", ["renovate", "renovate.json", ".renovaterc"]),
    ("Dependabot", ["dependabot", ".github/dependabot"]),
    ("Dev Container", [".devcontainer", "devcontainer.json"]),
    ("Gitpod", [".gitpod.yml", ".gitpod.Dockerfile"]),
    ("Codespaces", [".devcontainer", "devcontainer.json"]),
]


class ConfigDocsDetectorRole(_BaseDetectorRole):
    """Detect configuration formats, documentation, API specs, and tooling
    used in a project."""

    agent_name: str = "configdocs_detector"
    output_file: str = "configdocs_report.json"
    domain: str = "Configuration & Documentation"

    def build_task(self, backend: str, project_dir: str = ".",
                   working_dir: str = ".", project_scan: dict[str, Any] | None = None,
                   **context: Any) -> str:
        """Build the config/docs detection prompt."""
        scan_summary = _format_scan_summary(project_scan)
        common = (
            f"You are a project configuration & documentation analyst. Your "
            f"job is to analyze a project directory and identify all "
            f"configuration formats, documentation types, API specifications, "
            f"and development tooling in use.\n\n"
            f"## Project Directory\n{project_dir}\n"
            f"{scan_summary}"
            f"## Instructions\n"
            f"1. The project structure is shown above. "
            f"Focus your analysis on configuration, documentation, and tooling files.\n"
            f"2. Analyze for:\n"
            f"   - **Config formats**: YAML, TOML, JSON, INI, ENV, XML\n"
            f"   - **Documentation**: README, CONTRIBUTING, CHANGELOG, LICENSE, "
            f"CODE_OF_CONDUCT, SECURITY, docs/ directory, Wiki\n"
            f"   - **API specs**: OpenAPI/Swagger, GraphQL schemas, gRPC protobuf, "
            f"AsyncAPI, JSON Schema, RAML\n"
            f"   - **Tooling**: EditorConfig, Git config, pre-commit hooks, "
            f"Husky, commitlint, commitizen, semantic-release, Renovate, "
            f"Dependabot, Dev Containers, Gitpod, Codespaces\n"
            f"3. For each finding, determine:\n"
            f"   - What format/type it is\n"
            f"   - Which files provide evidence\n"
            f"   - How comprehensive the configuration is\n\n"
            f"4. Write findings to `configdocs_report.json`:\n"
            f"```json\n"
            f"{{\n"
            f'  "domain": "Configuration & Documentation",\n'
            f'  "config_formats": [\n'
            f'    {{"format": "YAML", "file_count": 15, '
            f'"examples": ["docker-compose.yml", ".github/workflows/ci.yml"]}}\n'
            f'  ],\n'
            f'  "documentation": [\n'
            f'    {{"type": "README", "path": "README.md", "size_hint": "5KB", '
            f'"completeness": "comprehensive"}}\n'
            f'  ],\n'
            f'  "api_specs": [\n'
            f'    {{"type": "OpenAPI", "path": "openapi.yaml", "version_hint": "3.0"}}\n'
            f'  ],\n'
            f'  "tooling": [\n'
            f'    {{"tool": "pre-commit", "config": ".pre-commit-config.yaml"}}\n'
            f'  ]\n'
            f"}}\n"
            f"```\n\n"
            f"Working directory: {working_dir}"
        )

        if backend == "claude_cli":
            backend_note = (
                "\n\nUse your own knowledge — do NOT search the web. "
                "Read project_scan.json, analyze files, write "
                "configdocs_report.json. Output TASK_COMPLETE."
            )
        else:
            backend_note = (
                "\n\nRead project_scan.json, analyze files, write "
                "configdocs_report.json, output TASK_COMPLETE."
            )

        return common + backend_note

    def _fallback_detect(self, project_dir: str = ".",
                         working_dir: str = ".") -> dict[str, Any]:
        """Deterministic config/docs detection without LLM."""
        scan = _load_project_scan(working_dir) or {}
        categories = scan.get("file_categories", {})
        all_files = (
            categories.get("source", []) +
            categories.get("config", []) +
            categories.get("docs", []) +
            categories.get("ci", []) +
            categories.get("build", []) +
            categories.get("other", [])
        )
        if not all_files:
            try:
                r = subprocess.run(
                    f"find {project_dir} -type f -not -path '*/.git/*' "
                    f"-not -path '*/node_modules/*' -not -path '*/__pycache__/*' "
                    f"| head -500",
                    shell=True, capture_output=True, text=True,
                    timeout=15, cwd=working_dir,
                )
                all_files = [f.strip() for f in r.stdout.strip().split("\n") if f.strip()]
            except (subprocess.TimeoutExpired, OSError):
                all_files = []

        # Config formats
        config_formats: list[dict[str, Any]] = []
        for fmt_name, extensions in _CONFIG_FORMATS:
            examples = [
                f for f in all_files
                if any(f.endswith(ext) for ext in extensions) or
                   any(ext in os.path.basename(f) for ext in extensions
                       if not ext.startswith("."))
            ][:5]
            if examples:
                config_formats.append({
                    "format": fmt_name,
                    "file_count": len(examples),
                    "examples": examples[:3],
                })

        # Documentation
        documentation: list[dict[str, Any]] = []
        for doc_type, indicators in _DOC_TYPES:
            matched = []
            for ind in indicators:
                for fpath in all_files:
                    bn = os.path.basename(fpath).lower()
                    path_lower = fpath.lower()
                    if ind.lower() in bn or ind.lower() in path_lower:
                        matched.append(fpath)
            if matched:
                size_hint = ""
                try:
                    fpath_full = os.path.join(working_dir, matched[0])
                    if os.path.exists(fpath_full):
                        size = os.path.getsize(fpath_full)
                        size_hint = f"{size // 1024}KB" if size >= 1024 else f"{size}B"
                except OSError:
                    pass
                completeness = "comprehensive" if len(matched) >= 3 else \
                               "moderate" if len(matched) >= 2 else "minimal"
                documentation.append({
                    "type": doc_type,
                    "path": matched[0],
                    "size_hint": size_hint,
                    "completeness": completeness,
                })

        # API specs
        api_specs: list[dict[str, Any]] = []
        for spec_name, indicators in _API_SPECS:
            matched = []
            for ind in indicators:
                for fpath in all_files:
                    bn = os.path.basename(fpath).lower()
                    if ind.lower() in bn or ind.lower() in fpath.lower():
                        matched.append(fpath)
            if matched:
                api_specs.append({
                    "type": spec_name,
                    "path": matched[0],
                    "version_hint": "",
                })

        # Tooling
        tooling: list[dict[str, Any]] = []
        for tool_name, indicators in _TOOLING:
            matched = []
            for ind in indicators:
                for fpath in all_files:
                    bn = os.path.basename(fpath).lower()
                    if ind.lower() in bn or ind.lower() in fpath.lower():
                        matched.append(fpath)
            if matched:
                tooling.append({
                    "tool": tool_name,
                    "config": matched[0],
                })

        return {
            "domain": "Configuration & Documentation",
            "config_formats": config_formats,
            "documentation": documentation,
            "api_specs": api_specs,
            "tooling": tooling,
            "_fallback": True,
        }
