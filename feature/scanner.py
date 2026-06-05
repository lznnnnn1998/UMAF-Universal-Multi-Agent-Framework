"""FeatureScannerRole — project understanding agent."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any

from tools import ToolRegistry
from agent import AgentResult, AgentRole
from utils import extract_json_object, safe_read


class FeatureScannerRole(AgentRole):
    """Scans project, identifies conventions, writes project_context.json."""

    agent_name = "feature_scanner"
    max_steps = 15

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        if hasattr(ToolRegistry, "feature_scanner_tools"):
            return ToolRegistry.to_dicts(ToolRegistry.feature_scanner_tools())
        return ToolRegistry.to_dicts([
            ToolRegistry.READ_FILE, ToolRegistry.WRITE_FILE, ToolRegistry.RUN_COMMAND,
        ])

    def build_task(self, backend: str, working_dir: str = ".",
                   project_dir: str = ".", **context: Any) -> str:
        common = (
            f"You are a project analyst. Scan the project directory and produce "
            f"a detailed project_context.json file.\n\n"
            f"## Project Directory: {project_dir}\n"
            f"Working directory for output: {working_dir}\n\n"
            f"## Instructions\n"
            f"1. Use `find` or `ls` to discover the project structure\n"
            f"2. Read key config files (pyproject.toml, setup.cfg, package.json, "
            f"requirements.txt, etc.) to determine language, version, framework, "
            f"and dependencies\n"
            f"3. Read 3-5 representative source files to extract conventions:\n"
            f"   - Naming (functions, classes, variables)\n"
            f"   - Import grouping/style (stdlib first, absolute vs relative)\n"
            f"   - Type annotation usage level\n"
            f"   - Docstring format (google, numpy, sphinx, none)\n"
            f"   - Error handling patterns (try/except, return None, raise)\n"
            f"4. Read 1-2 test files to determine:\n"
            f"   - Test framework (pytest, unittest, jest, etc.)\n"
            f"   - Test file naming (test_*.py, *_test.py)\n"
            f"   - Fixture patterns (conftest.py, setup/teardown)\n"
            f"   - Mock library used\n"
            f"5. Write project_context.json with ALL of the above\n\n"
            f"## Output schema\n"
            f"```json\n"
            f'{{\n'
            f'  "project_dir": ".",\n'
            f'  "language": "python",\n'
            f'  "language_version": "3.11",\n'
            f'  "framework": null,\n'
            f'  "source_directories": ["src"],\n'
            f'  "test_directories": ["tests"],\n'
            f'  "entry_points": ["src/main.py"],\n'
            f'  "conventions": {{\n'
            f'    "naming": {{"functions": "snake_case", "classes": "PascalCase"}},\n'
            f'    "import_style": "absolute",\n'
            f'    "type_annotations": "high",\n'
            f'    "docstring_format": "google",\n'
            f'    "error_handling": "exceptions with logging"\n'
            f'  }},\n'
            f'  "test_patterns": {{\n'
            f'    "framework": "pytest",\n'
            f'    "file_pattern": "test_*.py",\n'
            f'    "fixture_location": "conftest.py",\n'
            f'    "mock_library": "unittest.mock"\n'
            f'  }},\n'
            f'  "tech_stack": {{\n'
            f'    "dependencies": ["click"],\n'
            f'    "dev_dependencies": ["pytest"]\n'
            f'  }},\n'
            f'  "file_manifest": [\n'
            f'    {{"path": "src/main.py", "role": "entry_point"}}\n'
            f'  ],\n'
            f'  "scan_timestamp": "<ISO>"\n'
            f'}}\n'
            f"```\n"
        )
        if backend == "claude_cli":
            common += (
                "\nUse your own knowledge — do NOT search the web. "
                "Read key files and write project_context.json. "
                "Output TASK_COMPLETE when done."
            )
        else:
            common += "\nOutput TASK_COMPLETE when project_context.json is written."
        return common

    def parse_result(self, result: AgentResult, working_dir: str = ".",
                     project_dir: str = ".", **context: Any) -> dict[str, Any]:
        scan: dict[str, Any] = {}
        # 1. Try agent messages
        for msg in reversed(result.messages):
            content = msg.content if hasattr(msg, "content") else str(msg)
            json_str = extract_json_object(content)
            if json_str:
                try:
                    parsed = json.loads(json_str)
                    if "file_manifest" in parsed or "conventions" in parsed:
                        scan = parsed
                        break
                except json.JSONDecodeError:
                    continue
        # 2. Try disk
        if not scan:
            path = os.path.join(working_dir, "project_context.json")
            if os.path.isfile(path):
                try:
                    with open(path) as f:
                        scan = json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass
        # 3. Fallback
        if not scan:
            scan = self._fallback_scanner(project_dir, working_dir)
            if scan:
                out_path = os.path.join(working_dir, "project_context.json")
                try:
                    with open(out_path, "w") as f:
                        json.dump(scan, f, indent=2)
                except OSError:
                    pass
        return scan

    @staticmethod
    def _fallback_scanner(project_dir: str = ".", working_dir: str = ".") -> dict[str, Any]:
        """Deterministic scanner using find/ls — no LLM needed."""
        find_cmd = (
            f"find {project_dir} -type f "
            f"-not -path '*/.git/*' -not -path '*/__pycache__/*' "
            f"-not -path '*/node_modules/*' -not -path '*/.venv/*' "
            f"-not -path '*/venv/*' -not -path '*/dist/*' "
            f"-not -path '*/build/*' -not -path '*/.tox/*' "
            f"| sort | head -2000"
        )
        try:
            result = subprocess.run(
                find_cmd, shell=True, capture_output=True, text=True,
                timeout=30, cwd=working_dir,
            )
            raw = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        except (subprocess.TimeoutExpired, OSError):
            raw = []

        _scan_abs = os.path.abspath(os.path.join(working_dir, project_dir))
        files: list[str] = []
        for f in raw:
            abs_f = os.path.abspath(os.path.join(working_dir, f))
            if abs_f.startswith(_scan_abs):
                rel = os.path.relpath(abs_f, _scan_abs)
            else:
                rel = f
            files.append(rel)

        # Classify files
        config_exts = {".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".env"}
        config_names = {"requirements.txt", "package.json", "Makefile", "Dockerfile",
                        "setup.py", "setup.cfg", "pyproject.toml", "Makefile"}
        src_exts = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java",
                     ".c", ".cpp", ".h", ".hpp", ".rb", ".swift", ".kt"}
        test_dirs = {"tests", "test", "spec", "specs", "__tests__"}

        manifest = []
        src_dirs: set[str] = set()
        test_dirs_found: set[str] = set()
        entry_points: list[str] = []
        config_files: list[str] = []

        for f in files:
            parts = f.replace("\\", "/").split("/")
            ext = os.path.splitext(f)[1].lower()
            role = "other"

            if any(d in parts[:-1] for d in test_dirs):
                role = "test"
                if len(parts) >= 1:
                    test_dirs_found.add(parts[0] if parts[0] in test_dirs else
                                        next((p for p in parts if p in test_dirs), parts[0]))
            elif ext in config_exts or os.path.basename(f) in config_names:
                role = "config"
                config_files.append(f)
            elif ext in src_exts:
                role = "source"
                if len(parts) > 1:
                    src_dirs.add(parts[0])
                if os.path.basename(f).startswith("main.") or f in ("app.py", "index.js"):
                    entry_points.append(f)

            manifest.append({"path": f, "role": role})

        # Detect language
        py_count = sum(1 for f in files if f.endswith(".py"))
        js_count = sum(1 for f in files if f.endswith((".js", ".ts", ".jsx", ".tsx")))
        language = "python" if py_count >= js_count else ("javascript" if js_count > 0 else "unknown")

        # Read a few source files for conventions (best-effort)
        source_files = [m["path"] for m in manifest if m["role"] == "source"]
        conventions: dict[str, Any] = {
            "naming": {"functions": "snake_case", "classes": "PascalCase"},
            "import_style": "absolute",
            "type_annotations": "unknown",
            "docstring_format": "unknown",
            "error_handling": "unknown",
        }
        for sf in source_files[:5]:
            content = safe_read(os.path.join(working_dir, sf))
            if not content:
                continue
            if "def " in content and "_" in content:
                pass  # snake_case confirmed
            if "from __future__ import annotations" in content:
                conventions["type_annotations"] = "high"
            if '"""' in content:
                conventions["docstring_format"] = "google-style"
            if "try:" in content:
                conventions["error_handling"] = "try/except"
            break  # Sample one file for speed

        # Detect test patterns
        test_files = [m["path"] for m in manifest if m["role"] == "test"]
        test_patterns: dict[str, Any] = {
            "framework": "pytest" if any("pytest" in safe_read(
                os.path.join(working_dir, tf)) for tf in test_files[:3]) else "unknown",
            "file_pattern": "test_*.py",
            "fixture_location": "conftest.py" if any(
                "conftest" in tf for tf in test_files) else "inline",
            "mock_library": "unittest.mock",
        }

        return {
            "project_dir": project_dir,
            "total_files": len(files),
            "language": language,
            "source_directories": sorted(src_dirs),
            "test_directories": sorted(test_dirs_found),
            "entry_points": entry_points,
            "conventions": conventions,
            "test_patterns": test_patterns,
            "tech_stack": {"dependencies": [], "dev_dependencies": []},
            "file_manifest": manifest,
            "scan_timestamp": datetime.now(timezone.utc).isoformat(),
            "_fallback": True,
        }
