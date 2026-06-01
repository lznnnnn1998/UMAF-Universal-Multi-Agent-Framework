import os
from typing import Any

from agent import AgentRole
from tools import ToolRegistry


class OrganizerRole(AgentRole):
    """Assemble reviewed modules into a complete, working project."""

    agent_name = "coderpp_organizer"
    max_steps = 15

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        return ToolRegistry.to_dicts(ToolRegistry.organizer_tools())

    def build_task(self, backend: str, reviewed_modules: list[dict[str, Any]] | None = None,
                   input_spec: str = "", **context: Any) -> str:
        assert reviewed_modules is not None
        module_lines = []
        for m in reviewed_modules:
            name = m["module_name"]
            passed = "PASSED" if m.get("passed") else "NEEDS REVIEW"
            files = m.get("files", [])
            file_str = ", ".join(files) if files else "(no files)"
            log = m.get("log_file", "")
            module_lines.append(f"  - **{name}** [{passed}]: {file_str}")
            if log:
                module_lines.append(f"    Review log: `{log}`")
        module_list = "\n".join(module_lines)
        project_dir = "project"

        if backend == "claude_cli":
            return _build_organizer_task_claude_cli(module_list, project_dir, input_spec)
        return _build_organizer_task_deepseek(module_list, project_dir, input_spec)

    def parse_result(self, result, working_dir: str, **context: Any) -> str:
        project_dir = "project"
        full_project_path = os.path.join(working_dir, project_dir)
        if os.path.isdir(full_project_path):
            return project_dir
        return ""


def _build_organizer_task_deepseek(
    module_list: str, project_dir: str, input_spec: str,
) -> str:
    return f"""You are a project integrator. Assemble independently-built modules into a complete, working project.

## Requirement
{input_spec}

## Modules to Integrate
{module_list}

## Output
Project root: `{project_dir}/` (all paths are relative to working directory — do NOT prepend the working directory name)

## Instructions
1. Read ALL module source files listed above to understand each module's API and behavior.
2. Create the project structure under `{project_dir}/`:
   - `README.md`: project description, installation instructions, usage examples, module overview
   - `requirements.txt`: all third-party dependencies used by any module
   - `setup.py` or `pyproject.toml`: package configuration if this is a library
   - `main.py` (or appropriate entry point): integrates all modules, handles CLI/API entry
   - `__init__.py` files as needed for package structure
3. Write integration code that wires modules together. Fix any import path issues or API mismatches between modules.
4. Copy or reference module implementations into the project (use `read_file` to read module sources, then `write_file` to write them into the project structure).
5. Run the full project's tests to verify everything works together:
   ```bash
   cd {project_dir} && python3 -m pytest -v || python3 -m unittest discover -v
   ```
6. Fix any integration issues until tests pass.
7. Write `BUILD_LOG.md` in `{project_dir}/` summarizing:
   - Project structure and module layout
   - Integration decisions and changes made
   - Final test results
   - How to run and use the project
8. Output TASK_COMPLETE when done.

IMPORTANT: The final project must be complete and runnable. Every module should be properly integrated."""


def _build_organizer_task_claude_cli(
    module_list: str, project_dir: str, input_spec: str,
) -> str:
    return f"""You are a project integrator. Assemble independently-built modules into a complete, working project.

## Requirement
{input_spec}

## Modules to Integrate
{module_list}

## Output
Project root: `{project_dir}/` (all paths are relative to working directory — do NOT prepend the working directory name)

## Instructions
1. Use **Read** to read ALL module source files listed above to understand each module's API and behavior.
2. Create the project structure under `{project_dir}/` using **Write**:
   - `README.md`: project description, installation instructions, usage examples, module overview
   - `requirements.txt`: all third-party dependencies used by any module
   - `setup.py` or `pyproject.toml`: package configuration if this is a library
   - `main.py` (or appropriate entry point): integrates all modules, handles CLI/API entry
   - `__init__.py` files as needed for package structure
3. Write integration code that wires modules together. Fix any import path issues or API mismatches between modules.
4. Copy module implementations into the project (read module sources with **Read**, write into project with **Write**).
5. Use **Bash** to run the full project's tests and verify everything works:
   ```bash
   cd {project_dir} && python3 -m pytest -v || python3 -m unittest discover -v
   ```
6. Fix any integration issues until tests pass.
7. Write `BUILD_LOG.md` in `{project_dir}/` summarizing:
   - Project structure and module layout
   - Integration decisions and changes made
   - Final test results
   - How to run and use the project
8. Output TASK_COMPLETE when done.

IMPORTANT: The final project must be complete and runnable. Every module should be properly integrated."""


def assemble_project(
    reviewed_modules: list[dict[str, Any]],
    input_spec: str,
    working_dir: str,
    backend: str = "deepseek",
) -> str:
    """Assemble reviewed modules into a complete, working project.

    Args:
        reviewed_modules: list of dicts from review_module with module_name, files, passed, etc.
        input_spec: original requirement or .tex path.
        working_dir: base working directory.
        backend: LLM backend.

    Returns:
        Path to the assembled project directory (relative to working_dir), or empty string on failure.
    """
    if not reviewed_modules:
        return ""

    role = OrganizerRole()
    return role.execute(
        working_dir=working_dir,
        backend=backend,
        reviewed_modules=reviewed_modules,
        input_spec=input_spec,
    )
