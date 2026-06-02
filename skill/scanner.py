"""SkillScannerRole — scans project directory structure and writes project_scan.json."""

import json
import os
import subprocess
import sys
from typing import Any

# Ensure repo root is on sys.path so we can import agent.py and tools.py
# __file__ is .../coderpp_output/modules/skill_agent_roles/skill/scanner.py
# Repo root is 5 dirs up: .../universal_multi_agent_framework/
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agent import AgentResult, AgentRole  # noqa: E402
from tools import ToolRegistry  # noqa: E402


class SkillScannerRole(AgentRole):
    """Scan a project directory and build a structured directory listing.

    Uses ``find`` and ``ls`` to enumerate files under a project root,
    then writes the results to ``project_scan.json``. The scan output
    is consumed by the four domain detector roles.
    """

    agent_name: str = "skill_scanner"
    max_steps: int = 8

    # ── Tools ───────────────────────────────────────────────────────────

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        """Provide read_file, write_file, and run_command for directory scans."""
        return ToolRegistry.to_dicts([
            ToolRegistry.READ_FILE,
            ToolRegistry.WRITE_FILE,
            ToolRegistry.RUN_COMMAND,
        ])

    # ── Task prompt ─────────────────────────────────────────────────────

    def build_task(self, backend: str, project_dir: str = ".",
                   working_dir: str = ".", **context: Any) -> str:
        """Build the directory scan prompt with backend-aware instructions."""
        common = (
            f"You are a project structure analyst. Your job is to scan a "
            f"project directory and produce a structured listing of all files "
            f"and directories.\n\n"
            f"## Project Directory\n{project_dir}\n\n"
            f"## Task\n"
            f"1. Run `find {project_dir} -type f -not -path '*/.git/*' "
            f"-not -path '*/__pycache__/*' -not -path '*/node_modules/*' "
            f"-not -path '*/.venv/*' -not -path '*/venv/*' "
            f"-not -path '*/dist/*' -not -path '*/build/*' "
            f"-not -path '*/.tox/*' | sort | head -1000` to enumerate files.\n"
            f"2. Run `ls -laR {project_dir} | head -500` to capture the "
            f"directory tree structure.\n"
            f"3. Analyze the results and categorize files into groups:\n"
            f"   - **source**: .py, .js, .ts, .go, .rs, .java files\n"
            f"   - **config**: .yaml, .yml, .json, .toml, .ini, .cfg, .env files\n"
            f"   - **docs**: .md, .rst, .txt, README*, CHANGELOG*, CONTRIBUTING*\n"
            f"   - **test**: test_*.py, *_test.py, *.test.*, spec.*, __tests__/\n"
            f"   - **build**: Dockerfile*, docker-compose*, Makefile, CMakeLists.txt\n"
            f"   - **ci**: .github/workflows/, .gitlab-ci.yml, Jenkinsfile, .travis.yml\n"
            f"   - **other**: everything else\n\n"
            f"4. Write the scan results as JSON to `project_scan.json` with "
            f"this structure:\n"
            f"```json\n"
            f"{{\n"
            f'  "project_dir": "{project_dir}",\n'
            f'  "total_files": <int>,\n'
            f'  "total_dirs": <int>,\n'
            f'  "file_categories": {{\n'
            f'    "source": ["path/to/file1.py", ...],\n'
            f'    "config": ["path/to/config.yaml", ...],\n'
            f'    "docs": ["path/to/README.md", ...],\n'
            f'    "test": ["path/to/test_x.py", ...],\n'
            f'    "build": ["Dockerfile", ...],\n'
            f'    "ci": [".github/workflows/ci.yml", ...],\n'
            f'    "other": ["path/to/other.txt", ...]\n'
            f'  }},\n'
            f'  "key_files": {{\n'
            f'    "readme": "README.md",\n'
            f'    "license": "LICENSE",\n'
            f'    "contributing": "CONTRIBUTING.md",\n'
            f'    "changelog": "CHANGELOG.md"\n'
            f'  }},\n'
            f'  "top_level_dirs": ["src", "tests", "docs", ...],\n'
            f'  "scan_timestamp": "<ISO timestamp>"\n'
            f"}}\n"
            f"```\n\n"
            f"Working directory for file writes: {working_dir}"
        )

        if backend == "claude_cli":
            backend_note = (
                "\n\nUse your own knowledge — do NOT search the web. "
                "Run the find and ls commands, analyze their output, "
                "then write project_scan.json. Output TASK_COMPLETE when done."
            )
        else:
            backend_note = (
                "\n\nRun the find and ls commands, analyze their output, "
                "write project_scan.json, then output TASK_COMPLETE."
            )

        return common + backend_note

    # ── Parse result ────────────────────────────────────────────────────

    def parse_result(self, result: AgentResult, working_dir: str,
                     project_dir: str = ".", **context: Any) -> dict[str, Any]:
        """Extract project scan from agent response or disk, with fallback."""
        scan: dict[str, Any] = {}

        # 1. Try extracting JSON from agent response
        for msg in reversed(result.messages):
            content = msg.content if hasattr(msg, "content") else str(msg)
            json_str = self._extract_json_object(content)
            if json_str:
                try:
                    parsed = json.loads(json_str)
                    if "file_categories" in parsed or "total_files" in parsed:
                        scan = parsed
                        break
                except json.JSONDecodeError:
                    continue

        # 2. Try reading from disk
        if not scan:
            path = os.path.join(working_dir, "project_scan.json")
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        parsed = json.load(f)
                    if isinstance(parsed, dict) and "file_categories" in parsed:
                        scan = parsed
                except (json.JSONDecodeError, OSError):
                    pass

        # 3. Fallback: run find/ls directly
        if not scan:
            scan = self._fallback_scanner(project_dir, working_dir)

        return scan

    # ── Fallback scanner ────────────────────────────────────────────────

    @staticmethod
    def _fallback_scanner(project_dir: str = ".",
                          working_dir: str = ".") -> dict[str, Any]:
        """Run find and ls directly to scan a project directory without LLM.

        This is the primary fallback when the agent fails to produce output.
        It runs the same find/ls commands the agent would have run and
        categorizes files using deterministic rules.
        """
        from datetime import datetime, timezone

        scan_dir = project_dir

        # Run find for file listing
        find_cmd = (
            f"find {scan_dir} -type f "
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
                timeout=30, cwd=working_dir,
            )
            file_list = [
                f.strip() for f in find_result.stdout.strip().split("\n")
                if f.strip()
            ]
        except (subprocess.TimeoutExpired, OSError):
            file_list = []

        # Run ls for directory tree
        ls_cmd = f"ls -laR {scan_dir} | head -500"
        try:
            ls_result = subprocess.run(
                ls_cmd, shell=True, capture_output=True, text=True,
                timeout=30, cwd=working_dir,
            )
            dir_tree = ls_result.stdout.strip()
        except (subprocess.TimeoutExpired, OSError):
            dir_tree = ""

        # Determine top-level directories from ls output
        top_level_dirs: list[str] = []
        if os.path.isdir(os.path.join(working_dir, scan_dir)):
            try:
                entries = os.listdir(os.path.join(working_dir, scan_dir))
                top_level_dirs = sorted(
                    e for e in entries
                    if os.path.isdir(os.path.join(working_dir, scan_dir, e))
                )
            except OSError:
                pass

        # Categorize files
        categories: dict[str, list[str]] = {
            "source": [],
            "config": [],
            "docs": [],
            "test": [],
            "build": [],
            "ci": [],
            "other": [],
        }

        source_exts = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs",
                       ".java", ".c", ".cpp", ".h", ".hpp", ".rb", ".php",
                       ".swift", ".kt", ".scala", ".cs", ".vb", ".fs"}
        config_exts = {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg",
                       ".conf", ".env", ".properties", ".xml"}
        doc_exts = {".md", ".rst", ".txt", ".adoc", ".tex"}
        test_patterns = ("test_", "_test", ".test.", "spec.", "__tests__",
                         "test/", "tests/", "spec/")

        for fpath in file_list:
            basename = os.path.basename(fpath)
            ext = os.path.splitext(fpath)[1].lower()

            # Detect test files first
            is_test = (
                basename.startswith("test_") or
                basename.endswith("_test.py") or
                basename.endswith("_test.js") or
                basename.endswith("_test.ts") or
                ".test." in basename or
                basename.startswith("spec.") or
                "__tests__" in fpath or
                "/test/" in fpath or
                "/tests/" in fpath or
                "/spec/" in fpath
            )
            if is_test:
                categories["test"].append(fpath)
                continue

            # CI/CD detection
            if any(ci in fpath for ci in [
                ".github/workflows", ".gitlab-ci.yml", "Jenkinsfile",
                ".travis.yml", ".circleci", "azure-pipelines",
                "bitbucket-pipelines",
            ]):
                categories["ci"].append(fpath)
                continue

            # Build files
            if basename in ("Dockerfile", "Makefile", "CMakeLists.txt",
                            "Rakefile", "GNUmakefile") or \
               basename.startswith("Dockerfile") or \
               basename.startswith("docker-compose"):
                categories["build"].append(fpath)
                continue

            # Documentation
            if ext in doc_exts or basename.lower().startswith(("readme", "changelog",
                   "contributing", "license", "authors", "code_of_conduct",
                   "security", "governance")):
                categories["docs"].append(fpath)
                continue

            # Config files
            if ext in config_exts or basename in (".editorconfig", ".gitignore",
                   ".prettierrc", ".eslintrc", ".babelrc", ".npmrc"):
                categories["config"].append(fpath)
                continue

            # Source files
            if ext in source_exts:
                categories["source"].append(fpath)
                continue

            # Everything else
            categories["other"].append(fpath)

        # Identify key files
        key_files: dict[str, str] = {}
        for fpath in file_list:
            bn = os.path.basename(fpath).lower()
            if bn.startswith("readme") and "key_files" not in key_files:
                key_files["readme"] = fpath
            elif bn.startswith("license") and "license" not in key_files:
                key_files["license"] = fpath
            elif bn.startswith("contributing") and "contributing" not in key_files:
                key_files["contributing"] = fpath
            elif bn.startswith("changelog") and "changelog" not in key_files:
                key_files["changelog"] = fpath

        # Count directories
        try:
            dir_count_result = subprocess.run(
                f"find {scan_dir} -type d -not -path '*/.git/*' -not -path '*/__pycache__/*' -not -path '*/node_modules/*' | wc -l",
                shell=True, capture_output=True, text=True,
                timeout=15, cwd=working_dir,
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
            "_fallback": True,
        }

    # ── JSON extraction helper ──────────────────────────────────────────

    @staticmethod
    def _extract_json_object(text: str) -> str | None:
        """Extract the first complete JSON object from text using brace
        counting with string-aware parsing."""
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
