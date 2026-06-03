import os
from typing import Any

from agent import AgentResult, AgentRole
from tools import ToolRegistry


class CoderPPWorkerRole(AgentRole):
    """Implement a single sub-module with code, tests, and a build log."""

    agent_name = "coderpp_worker"
    max_steps = 20

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        return ToolRegistry.to_dicts(ToolRegistry.coderpp_worker_tools())

    @staticmethod
    def _is_cpp_module(sub_task: dict) -> bool:
        """Detect whether a module produces C++ (vs Python) from files_to_create."""
        files = sub_task.get("files_to_create", [])
        if not files:
            return False
        cpp_exts = {".h", ".hpp", ".cpp", ".cxx", ".cc", ".c", ".cu"}
        for f in files:
            if os.path.splitext(f)[1].lower() in cpp_exts:
                return True
        return False

    def build_task(self, backend: str, sub_task: dict | None = None, **context: Any) -> str:
        assert sub_task is not None
        module_name = sub_task["module_name"]
        description = sub_task["description"]
        module_dir = f"modules/{module_name}"
        files_to_create = sub_task.get("files_to_create", [])
        environment = context.get("environment", "")

        is_cpp = self._is_cpp_module(sub_task)

        if backend == "claude_cli":
            if is_cpp:
                return _build_worker_task_cpp_claude_cli(module_name, description, module_dir, files_to_create, environment)
            return _build_worker_task_claude_cli(module_name, description, module_dir, environment)
        if is_cpp:
            return _build_worker_task_cpp_deepseek(module_name, description, module_dir, files_to_create, environment)
        return _build_worker_task_deepseek(module_name, description, module_dir, environment)

    def parse_result(self, result: AgentResult, working_dir: str,
                     sub_task: dict | None = None, **context: Any) -> dict[str, Any]:
        assert sub_task is not None
        module_name = sub_task["module_name"]
        module_dir = f"modules/{module_name}"
        sub_id = sub_task["id"]

        if not result.success:
            return {
                "sub_task_id": sub_id,
                "module_name": module_name,
                "files": [],
                "log_file": "",
                "summary": "Agent did not complete successfully.",
            }

        module_path = os.path.join(working_dir, module_dir)
        created_files: list[str] = []
        log_file = ""
        if os.path.isdir(module_path):
            for root, dirs, files in sorted(os.walk(module_path)):
                dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__" and d != "build")
                rel_root = os.path.relpath(root, working_dir)
                for f in sorted(files):
                    if f.startswith("."):
                        continue
                    full = os.path.join(rel_root, f)
                    if os.path.isfile(os.path.join(working_dir, full)):
                        created_files.append(full)
                        if f.endswith("log.md"):
                            log_file = full

        summary = ""
        for msg in reversed(result.messages):
            content = msg.content if hasattr(msg, "content") else str(msg)
            if len(content) > 100:
                summary = content[:500] + "..." if len(content) > 500 else content
                break

        return {
            "sub_task_id": sub_id,
            "module_name": module_name,
            "files": created_files,
            "log_file": log_file,
            "summary": summary,
        }


def _build_worker_task_deepseek(title: str, description: str, module_dir: str, environment: str = "") -> str:
    env_section = ""
    if environment:
        env_section = f"""## Project Environment
The following environment was set up by the head agent. ALL workers MUST use this same environment.
Use the exact python path and conda environment documented below when running commands.

{environment}

"""
    return f"""You are a software engineer. Implement the following module with complete, working code.

{env_section}## Module
**Name**: {title}
**Description**: {description}

## Output Directory
All file paths are RELATIVE to the working directory. The working directory name MUST NOT appear in any path or command you write.
- `write_file` / `write_lines`: paths resolve relative to working directory automatically (e.g., `{module_dir}/module.py`).
- `run_command`: commands ALREADY run inside the working directory. NEVER `cd <working_dir>` — you are already there.

## Import Path Setup
Modules live in `modules/<name>/` as Python packages with `__init__.py`. If your module imports from sibling modules, run tests with:

    PYTHONPATH=modules python -m pytest {module_dir}/ -v

Document the exact test command in your log.md.

## Instructions
1. Write the main implementation file(s) with complete, working Python code. Use type hints and docstrings.
   - **PREFERRED**: Use `write_lines` for all code files. It takes `path` and `lines` (a JSON array of strings, one per line). Each line is a separate string element in a JSON array — this avoids multi-line string escaping problems.
   - Example: `{{"tool": "write_lines", "args": {{"path": "{module_dir}/module.py", "lines": ["def hello():", "    print('hi')"]}}}}`
   - Only use `write_file` for very short files like `log.md`.
   - **FALLBACK**: If `write_lines` fails repeatedly (you see "No valid tool call found"), DO NOT keep retrying. Immediately switch to using `run_command` with a Python one-liner that writes the file using triple-quoted strings: `python3 -c "content='''...your code...'''; open('{module_dir}/module.py','w').write(content)"`. The triple-quote syntax inside single quotes avoids escaping issues.
2. Write unit tests that thoroughly cover the module's functionality. Use pytest or unittest.
3. Run `PYTHONPATH=modules python -m pytest {module_dir}/ -v` with `run_command` to verify tests pass. Fix any issues.
4. Write a `log.md` file in `{module_dir}/` documenting:
   - **Implementation Summary**: what was built and how it works
   - **Design Decisions**: key architectural choices and trade-offs
   - **Known Issues**: any limitations, edge cases not handled, or assumptions made
   - **Test Results**: which tests pass, coverage notes, any failures
   - **How to Run Tests**: the exact command (with PYTHONPATH)
5. After writing all files, read them back to verify correctness.
6. Output TASK_COMPLETE when done.

Focus on clean, working code. The module must be independently testable."""


def _build_worker_task_claude_cli(title: str, description: str, module_dir: str, environment: str = "") -> str:
    env_section = ""
    if environment:
        env_section = f"""## Project Environment
The following environment was set up by the head agent. ALL workers MUST use this same environment.
Use the exact python path and conda environment documented below when running commands.

{environment}

"""
    return f"""You are a software engineer. Implement the following module with complete, working code.

{env_section}## Module
**Name**: {title}
**Description**: {description}

## Output Directory
All file paths are RELATIVE to the working directory. The working directory name MUST NOT appear in any path or command you write.
- `Write`/`Read`: paths resolve relative to working directory automatically.
- `Bash`: commands ALREADY run inside the working directory.

## Import Path Setup (CRITICAL)
Modules live in `modules/<name>/` as Python packages with `__init__.py`. If your module imports from sibling modules (e.g., `from palindrome_core import is_palindrome`), you MUST ensure the import works at test time. Run tests with:

    PYTHONPATH=modules python -m pytest {module_dir}/ -v

If you have cross-module dependencies, test files should add `modules/` to sys.path OR use the PYTHONPATH approach above. Document the exact test command in your log.md.

## Instructions
1. Write the main implementation file(s) with complete, working Python code. Use type hints and docstrings.
2. Write unit tests that thoroughly cover the module's functionality. Use pytest or unittest.
3. Use **Bash** to run the tests with `PYTHONPATH=modules python -m pytest {module_dir}/ -v` and verify they pass. Fix any issues you find.
4. Write a `log.md` file in `{module_dir}/` documenting:
   - **Implementation Summary**: what was built and how it works
   - **Design Decisions**: key architectural choices and trade-offs
   - **Known Issues**: any limitations, edge cases not handled, or assumptions made
   - **Test Results**: which tests pass, coverage notes, any failures
   - **How to Run Tests**: the exact command to run tests (with PYTHONPATH)
5. After writing, use **Read** to verify all files were written correctly.
6. Output TASK_COMPLETE when done.

Focus on clean, working code. The module must be independently testable. Use your own reasoning for design decisions — do NOT spawn nested claude -p calls."""


def _build_worker_task_cpp_claude_cli(
    title: str, description: str, module_dir: str,
    files_to_create: list[str], environment: str = "",
) -> str:
    env_section = ""
    if environment:
        env_section = f"""## Project Environment
{environment}

"""
    file_list = "\n".join(f"  - {f}" for f in files_to_create)
    return f"""You are a C++ software engineer. Implement the following module with complete, working C++ code.

{env_section}## Module
**Name**: {title}
**Description**: {description}

## Files to Create
{file_list}

## Output Directory
All file paths are RELATIVE to the working directory. The working directory name MUST NOT appear in any path or command you write.
- `Write`: paths resolve relative to working directory automatically (e.g., `{module_dir}/include/foo.h`).
- `Bash`: commands ALREADY run inside the working directory. NEVER `cd <working_dir>` — you are already there.

## CRITICAL: Algorithm TODO Policy
This is an ARCHITECTURE-ONLY implementation. You MUST:
1. Write complete C++ interfaces, abstract base classes, data structures, and boilerplate
2. For ANY algorithmic logic (imaging, detection, feature extraction, refocus, etc.), write ONLY:
   - A clear function signature with proper parameter types
   - A `// TODO: Implement <algorithm name>` comment in the function body
   - A brief comment describing inputs, outputs, and expected behavior
   - Return a sensible default/empty value so the code compiles
3. Example:
   ```cpp
   // TODO: Implement SAR range-doppler focusing algorithm
   // Input: raw_echo - parsed echo data frames
   // Output: focused_sar_image - single-look complex image
   // Expected behavior: apply range compression, RCMC, azimuth compression
   SARImage performFocusing(const ParsedEchoData& raw_echo) {{
       SARImage result;
       // Placeholder: return empty image with correct dimensions
       return result;
   }}
   ```

## Instructions
1. Write ALL files listed above. Header files (.h/.hpp) go in include/ subdirs, source files (.cpp) go in src/ subdirs, test files go in tests/ subdirs.
2. Use modern C++17 with namespace sapp::<module>.
3. Write a `CMakeLists.txt` in `{module_dir}/` that builds a static library for this module.
4. Write Google Test unit tests that verify:
   - Data structures can be constructed correctly
   - Interfaces compile and virtual methods can be called
   - TODO placeholders return expected default values
   - Factory functions create valid objects
5. Use `Bash` to verify compilation: `cmake -B build -G Ninja && cmake --build build` (from the module directory). Fix any compilation errors.
6. Write a `log.md` file in `{module_dir}/` documenting implementation summary, design decisions, known issues, build instructions.
7. After writing all files, use `Read` to verify correctness.
8. Output TASK_COMPLETE when done.

Focus on clean, compilable C++ interfaces. No algorithm logic — only architecture and TODO stubs."""


def _build_worker_task_cpp_deepseek(
    title: str, description: str, module_dir: str,
    files_to_create: list[str], environment: str = "",
) -> str:
    env_section = ""
    if environment:
        env_section = f"""## Project Environment
{environment}

"""
    file_list = "\n".join(f"  - {f}" for f in files_to_create)
    return f"""You are a C++ software engineer. Implement the following module with complete, working C++ code.

{env_section}## Module
**Name**: {title}
**Description**: {description}

## Files to Create
{file_list}

## Output Directory
All file paths are RELATIVE to the working directory. The working directory name MUST NOT appear in any path or command you write.
- `write_file` / `write_lines`: paths resolve relative to working directory automatically (e.g., `{module_dir}/include/foo.h`).
- `run_command`: commands ALREADY run inside the working directory. NEVER `cd <working_dir>` — you are already there.

## CRITICAL: Algorithm TODO Policy
This is an ARCHITECTURE-ONLY implementation. You MUST:
1. Write complete C++ interfaces, abstract base classes, data structures, and boilerplate
2. For ANY algorithmic logic, write ONLY a function signature with a `// TODO: Implement <name>` comment
3. Return sensible default values so code compiles

## Instructions
1. Write ALL files listed above. Use `write_lines` for code files.
2. Use modern C++17 with namespace sapp::<module>.
3. Write a `CMakeLists.txt` in `{module_dir}/`.
4. Write Google Test unit tests for data structures and interfaces.
5. Use `run_command` to verify compilation. Fix any errors.
6. Write a `log.md` file in `{module_dir}/`.
7. Output TASK_COMPLETE when done.

Focus on clean, compilable C++ interfaces. No algorithm logic — only architecture and TODO stubs."""


def code_submodule(
    sub_task: dict[str, Any],
    working_dir: str,
    backend: str = "deepseek",
    environment: str = "",
    version: int = 1,
) -> dict[str, Any]:
    """Implement a single sub-module with code, tests, and a build log.

    Args:
        sub_task: dict with id, module_name, description, files_to_create, dependencies.
        working_dir: base working directory.
        backend: LLM backend.
        environment: contents of ENVIRONMENT.md for consistent worker setup.
        version: checkpoint version (auto-resumes from previous version if > 1).

    Returns:
        dict with sub_task_id, module_name, files, log_file, summary.
    """
    sub_id = sub_task["id"]
    agent_name = f"coderpp_worker_{sub_id:02d}"
    role = CoderPPWorkerRole()
    role.agent_name = agent_name

    return role.execute(
        working_dir=working_dir,
        backend=backend,
        version=version,
        sub_task=sub_task,
        environment=environment,
    )
