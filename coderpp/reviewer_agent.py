import os
from typing import Any

from agent import AgentResult, AgentRole
from tools import ToolRegistry
from utils import scan_review_verdict


class CoderPPReviewerRole(AgentRole):
    """Review a module's code, find and fix bugs, and write a review log."""

    agent_name = "coderpp_reviewer"
    max_steps = 15

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        return ToolRegistry.to_dicts(ToolRegistry.coderpp_reviewer_tools())

    def build_task(self, backend: str, worker_output: dict | None = None, **context: Any) -> str:
        assert worker_output is not None
        module_name = worker_output["module_name"]
        files = worker_output.get("files", [])
        log_file = worker_output.get("log_file", "")
        module_dir = f"modules/{module_name}"

        if backend == "claude_cli":
            return _build_review_task_claude_cli(module_name, files, log_file, module_dir)
        return _build_review_task_deepseek(module_name, files, log_file, module_dir)

    def parse_result(self, result: AgentResult, working_dir: str,
                     worker_output: dict | None = None, **context: Any) -> dict[str, Any]:
        assert worker_output is not None
        module_name = worker_output["module_name"]
        module_dir = f"modules/{module_name}"
        sub_id = worker_output["sub_task_id"]

        if not result.success:
            return {
                "sub_task_id": sub_id,
                "module_name": module_name,
                "files": [],
                "log_file": "",
                "passed": False,
            }

        # Check only AIMessages for the verdict — task prompts and tool results
        # may contain REVIEW_PASSED/REVIEW_FAILED as part of instructions/docs.
        review_passed = scan_review_verdict(result.messages) or False

        # Double-check: read the review.md file if it was written — it is the
        # authoritative source for the verdict.
        module_path = os.path.join(working_dir, module_dir)
        current_files: list[str] = []
        review_log = ""
        if os.path.isdir(module_path):
            for f in sorted(os.listdir(module_path)):
                full = os.path.join(module_dir, f)
                if os.path.isfile(os.path.join(working_dir, full)):
                    current_files.append(full)
                    if f.endswith("review.md"):
                        review_log = full

        # If review.md exists, use its verdict as the authoritative source
        if review_log:
            review_path = os.path.join(working_dir, review_log)
            try:
                with open(review_path) as rf:
                    review_content = rf.read()
                if "REVIEW_PASSED" in review_content and "REVIEW_FAILED" not in review_content:
                    review_passed = True
                elif "REVIEW_FAILED" in review_content:
                    review_passed = False
            except (OSError, IOError):
                pass

        return {
            "sub_task_id": sub_id,
            "module_name": module_name,
            "files": current_files,
            "log_file": review_log,
            "passed": review_passed,
        }


def _build_review_task_deepseek(
    module_name: str, files: list[str], log_file: str, module_dir: str,
) -> str:
    file_list = "\n".join(f"  - `{f}`" for f in files)
    return f"""You are a code reviewer. Review the following module for bugs and fix all issues found.

## Module
**Name**: {module_name}
**Location**: `{module_dir}/` (paths are relative to working directory — do NOT prepend the working directory name)

## Files
{file_list}

## Test Running
Modules are Python packages under `modules/`. Run tests with:

    PYTHONPATH=modules python -m pytest {module_dir}/ -v

## Instructions
1. Read ALL source code files and the worker's log file (`{log_file or module_dir + '/log.md'}`) to understand what was built.
2. Read the test files to understand expected behavior.
3. Run `PYTHONPATH=modules python -m pytest {module_dir}/ -v` with `run_command` to see which pass/fail.
4. Identify all bugs, errors, and issues:
   - Logic errors, off-by-one, incorrect assumptions
   - Missing edge case handling
   - Import errors, missing dependencies
   - Test failures and their root causes
5. Fix all identified issues by editing the source files in-place with `write_file`.
6. Re-run `PYTHONPATH=modules python -m pytest {module_dir}/ -v` to verify fixes. Iterate until tests pass.
7. Write a `review.md` file in `{module_dir}/` documenting:
   - **Issues Found**: each bug/issue with a brief description
   - **Fixes Applied**: what was changed and why
   - **Final Test Results**: all tests passing, or any remaining failures with explanation
   - **Verdict**: REVIEW_PASSED (all tests pass + code is solid) or REVIEW_FAILED (unresolvable issues remain)
8. Output TASK_COMPLETE when done.

Be thorough. Every bug you find and fix improves the final project quality. Focus on code correctness and design — do NOT waste time debugging import paths if pytest passes."""


def _build_review_task_claude_cli(
    module_name: str, files: list[str], log_file: str, module_dir: str,
) -> str:
    file_list = "\n".join(f"  - `{f}`" for f in files)
    return f"""You are a code reviewer. Review the following module for bugs and fix all issues found.

## Module
**Name**: {module_name}
**Location**: `{module_dir}/` (paths are relative to working directory — do NOT prepend the working directory name)

## Files
{file_list}

## Test Running (CRITICAL — read before running any commands)
Modules are Python packages under `modules/`. To run tests with correct imports, ALWAYS use:

    PYTHONPATH=modules python -m pytest {module_dir}/ -v

**Time limit**: Keep test execution UNDER 60 seconds. Run `--collect-only` first, then run only 2-3 representative tests. The worker already verified all tests pass — you only need a quick sanity check. Use `-k "not slow and not benchmark"` to skip slow tests if needed.

Do NOT use plain `python -c "from X import Y"` to test imports — that won't work because `modules/` is not on sys.path by default. pytest with PYTHONPATH=modules handles this correctly. If the quick tests pass, imports are fine — move on.

## Instructions
1. Use **Read** to read ALL source code files and the worker's log file (`{log_file or module_dir + '/log.md'}`) to understand what was built.
2. Use **Read** to read the test files to understand expected behavior.
3. Use **Bash** to run `PYTHONPATH=modules python -m pytest {module_dir}/ -v` and see which tests pass/fail.
4. Identify all bugs, errors, and issues:
   - Logic errors, off-by-one, incorrect assumptions
   - Missing edge case handling
   - Import errors, missing dependencies
   - Test failures and their root causes
5. Fix all identified issues by editing source files in-place with **Write**.
6. Re-run `PYTHONPATH=modules python -m pytest {module_dir}/ -v` to verify fixes. Iterate until tests pass.
7. Write a `review.md` file in `{module_dir}/` documenting:
   - **Issues Found**: each bug/issue with a brief description
   - **Fixes Applied**: what was changed and why
   - **Final Test Results**: all tests passing, or any remaining failures with explanation
   - **Verdict**: REVIEW_PASSED (all tests pass + code is solid) or REVIEW_FAILED (unresolvable issues remain)
8. Output TASK_COMPLETE when done.

Be thorough. Every bug you find and fix improves the final project quality. Focus on code correctness and design — do NOT waste time debugging import paths if pytest passes."""


def review_module(
    worker_output: dict[str, Any],
    working_dir: str,
    backend: str = "deepseek",
    version: int = 1,
) -> dict[str, Any]:
    """Review a module's code, find and fix bugs, and write a review log.

    Args:
        worker_output: dict from code_submodule with module_name, files, log_file, etc.
        working_dir: base working directory.
        backend: LLM backend.
        version: checkpoint version (auto-resumes from previous version if > 1).

    Returns:
        dict with sub_task_id, module_name, files, log_file, passed.
    """
    sub_id = worker_output["sub_task_id"]
    agent_name = f"coderpp_reviewer_{sub_id:02d}"

    role = CoderPPReviewerRole()
    role.agent_name = agent_name
    return role.execute(
        working_dir=working_dir,
        backend=backend,
        version=version,
        worker_output=worker_output,
    )
