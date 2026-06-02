"""Pipeline base classes and implementations.

BasePipeline provides output directory management, double-check confirmation,
and a standard run() lifecycle. Each pipeline subclass specializes decomposition,
graph building, and result display.
"""

import concurrent.futures
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from agent import AgentRole, BaseAgent, CheckpointManager, run_agent
from tools import TOOL_MAP, ToolRegistry
from topology.analyzer import TopologyAnalyzerRole
from topology.designer import TopologyDesignerRole
from topology.evaluator import TopologyEvaluatorRole
from topology.writer import TopologyWriterRole
from skill.scanner import SkillScannerRole
from skill.detectors import (ConfigDocsDetectorRole, InfraDetectorRole,
                              JSDetectorRole, PythonDetectorRole)
from skill.aggregator import SkillAggregatorRole
from skill.writer import SkillReportWriterRole


# ═══════════════════════════════════════════════════════════════════════════
# Base Pipeline
# ═══════════════════════════════════════════════════════════════════════════

class BasePipeline:
    """Abstract pipeline with output dir management and double-check confirmation."""

    # Override in subclass
    name: str = "base"
    default_output_dir: str = "output"

    def __init__(
        self,
        working_dir: str | None = None,
        backend: str = "deepseek",
        clean: bool = False,
        resume: bool = False,
        yes: bool = False,
    ):
        repo_root = Path(__file__).resolve().parent
        self.working_dir = str(working_dir) if working_dir else str(repo_root / self.default_output_dir)
        self.backend = backend
        self.clean = clean
        self.resume = resume
        self.yes = yes

    # --- Output directory management ---

    def manage_output_dir(self):
        """Prepare the working directory based on flags."""
        if self.clean:
            if os.path.exists(self.working_dir):
                shutil.rmtree(self.working_dir)
            os.makedirs(self.working_dir, exist_ok=True)
            return

        if os.path.exists(self.working_dir) and os.listdir(self.working_dir):
            if self.resume:
                print(f"[resume] Continuing from existing output in: {self.working_dir}")
            else:
                print(f"[warn] Output directory has prior content: {self.working_dir}")
                print(f"       Use --clean to start fresh or --resume to continue from checkpoints.")

        os.makedirs(self.working_dir, exist_ok=True)

    # --- Double-check mechanism ---

    def confirm_decomposition(self, input_spec: str) -> list[dict[str, Any]]:
        """Decompose the input, show results, and ask user to confirm.

        Returns the confirmed sub_tasks list.
        """
        sub_tasks = self._decompose(input_spec)

        if not sub_tasks:
            print("Warning: decomposition produced no sub-tasks.")
            return []

        if self.yes or not sys.stdin.isatty():
            self._display_decomposition(sub_tasks)
            return sub_tasks

        self._display_decomposition(sub_tasks)

        while True:
            try:
                choice = input("\nIs this what you want? [Y]es / [n]o / [e]dit: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return sub_tasks

            if choice in ("", "y", "yes"):
                return sub_tasks
            elif choice in ("n", "no"):
                print("Aborted. Please rephrase your requirement and try again.")
                sys.exit(0)
            elif choice in ("e", "edit"):
                sub_tasks = self._edit_decomposition(sub_tasks)
                if sub_tasks:
                    self._display_decomposition(sub_tasks)
            else:
                print("Please answer y, n, or e.")

    def _decompose(self, input_spec: str) -> list[dict[str, Any]]:
        """Override in subclass to decompose the input into sub-tasks."""
        raise NotImplementedError

    def _display_decomposition(self, sub_tasks: list[dict[str, Any]]):
        """Pretty-print the decomposition."""
        print(f"\nProposed sub-tasks ({len(sub_tasks)}):")
        print("-" * 50)
        for t in sub_tasks:
            name = t.get("module_name") or t.get("title", "?")
            desc = t.get("description", "")[:120]
            print(f"  [{t.get('id', '?')}] {name}")
            if desc:
                print(f"      {desc}")

    def _edit_decomposition(self, sub_tasks: list[dict]) -> list[dict] | None:
        """Let the user edit sub-tasks by removing unwanted ones."""
        print("\nEnter IDs to REMOVE (comma-separated), or press Enter to keep all:")
        try:
            line = input("Remove: ").strip()
        except (EOFError, KeyboardInterrupt):
            return sub_tasks
        if not line:
            return sub_tasks
        try:
            remove_ids = {int(x.strip()) for x in line.split(",")}
        except ValueError:
            print("Invalid input, keeping all.")
            return sub_tasks
        return [t for t in sub_tasks if t.get("id") not in remove_ids]

    # --- Lifecycle ---

    def run(self, input_spec: str):
        """Full pipeline run: manage dir → confirm → graph.invoke → print."""
        self.manage_output_dir()

        # Resume branch: try to reconstruct state from disk
        if self.resume:
            initial_state = self._try_load_resume_state(input_spec)
            if initial_state is not None:
                graph = self._build_graph()
                version = initial_state.get("version", 1)
                status = initial_state.get("status", "?")
                workers_done = sum(
                    1 for wo in initial_state.get("worker_outputs", [])
                    if wo.get("files")
                )
                print(f"\n[resume] Loaded v{version}, status={status}, "
                      f"{workers_done}/{initial_state.get('worker_stats', {}).get('total', '?')} workers have files")
                print(f"\nResuming {self.name} pipeline...")
                final_state = graph.invoke(initial_state)
                self._print_results(final_state)
                return

        sub_tasks = self.confirm_decomposition(input_spec)
        initial_state = self._build_initial_state(input_spec, sub_tasks)
        graph = self._build_graph()

        print(f"\nRunning {self.name} pipeline...")
        final_state = graph.invoke(initial_state)

        self._print_results(final_state)

    # --- Subclass interface ---

    def _build_graph(self) -> StateGraph:
        raise NotImplementedError

    def _build_initial_state(self, input_spec: str, sub_tasks: list[dict]) -> dict:
        raise NotImplementedError

    def _try_load_resume_state(self, input_spec: str) -> dict | None:
        """Reconstruct pipeline state from disk. Returns None if not possible.

        Subclasses override this to load pipeline-specific state
        (decomposition.json, checkpoints, module files, etc.).
        """
        return None

    def _print_results(self, final_state: dict):
        raise NotImplementedError

    # --- Shared helpers for subclasses ---

    @staticmethod
    def _topological_levels(sub_tasks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Group sub_tasks into dependency-ordered levels.

        Dependencies can reference tasks by ``id`` (int) or ``module_name`` (str).
        Tasks without dependencies all land in level 0, so the common case
        (no deps at all) produces a single level — equivalent to flat parallelism.
        """
        if not any(t.get("dependencies") for t in sub_tasks):
            return [list(sub_tasks)]

        id_to_task = {t["id"]: t for t in sub_tasks}

        def _task_key(t: dict) -> str:
            return t.get("module_name") or f"__id_{t['id']}"

        def _dep_keys(t: dict) -> set[str]:
            keys: set[str] = set()
            for d in t.get("dependencies", []):
                if isinstance(d, int):
                    match = id_to_task.get(d)
                    keys.add(_task_key(match) if match else f"__id_{d}")
                elif isinstance(d, str):
                    keys.add(d)
                elif isinstance(d, dict):
                    keys.add(d.get("module_name") or f"__id_{d.get('id', d)}")
            return keys

        remaining: set[str] = {_task_key(t) for t in sub_tasks}
        key_to_task: dict[str, dict] = {_task_key(t): t for t in sub_tasks}
        levels: list[list[dict[str, Any]]] = []

        while remaining:
            current = [key_to_task[k] for k in sorted(remaining)
                       if _dep_keys(key_to_task[k]).isdisjoint(remaining)]
            if not current:
                current = [key_to_task[k] for k in remaining]
            levels.append(current)
            remaining -= {_task_key(t) for t in current}

        return levels

    @staticmethod
    def _run_workers_with_deps(
        items: list[dict],
        agent_func,
        working_dir: str,
        backend: str,
        timeout: int = 300,
        retry_failures: bool = False,
        max_retries: int = 1,
    ) -> tuple[list[dict], int, int]:
        """Run agents respecting dependency ordering.

        When no task declares dependencies this degenerates to flat parallelism
        (identical to ``_run_parallel_agents``).  Otherwise tasks are grouped
        into topological levels; levels run sequentially, tasks within a level
        run in parallel.
        """
        levels = BasePipeline._topological_levels(items)
        if len(levels) == 1:
            return BasePipeline._run_parallel_agents(
                items, agent_func, working_dir, backend, timeout,
                max_workers=len(items),
                retry_failures=retry_failures, max_retries=max_retries,
            )

        all_outputs: list[dict] = []
        total_succeeded = 0
        total_failed = 0

        for level_idx, level_tasks in enumerate(levels):
            names = [t.get("module_name") or t.get("title", "?") for t in level_tasks]
            print(f"\n  [dependency level {level_idx}/{len(levels)}] Running: {names}")

            results, succeeded, failed = BasePipeline._run_parallel_agents(
                level_tasks, agent_func, working_dir, backend, timeout,
                max_workers=len(level_tasks),
                retry_failures=retry_failures, max_retries=max_retries,
            )
            all_outputs.extend(results)
            total_succeeded += succeeded
            total_failed += failed

            # Stop on dependency failure: dependent levels need the outputs of
            # this level — retry the failed dependency first (via version bump)
            # before running anything that depends on it.
            if failed > 0 and level_idx + 1 < len(levels):
                remaining = sum(len(l) for l in levels[level_idx + 1:])
                print(f"\n  [dependency] Stopping early: {failed} task(s) failed in level "
                      f"{level_idx} — {remaining} downstream task(s) deferred for retry.")
                break

        return all_outputs, total_succeeded, total_failed

    @staticmethod
    def _status_router(flow_map: dict[str, str], terminal_errors: set[str] | None = None):
        """Build a status-based router function for LangGraph."""
        terminal = terminal_errors or set()
        def router(state: dict) -> Literal["__end__"] | str:
            status = state.get("status", "")
            if status in terminal:
                return END
            if status in flow_map:
                return flow_map[status]
            return END
        return router

    @staticmethod
    def _run_parallel_agents(
        items: list[dict],
        agent_func,
        working_dir: str,
        backend: str,
        timeout: int = 300,
        max_workers: int | None = None,
        retry_failures: bool = False,
        max_retries: int = 1,
    ) -> tuple[list[dict], int, int]:
        """Run an agent function in parallel for each item in the list.

        Returns (outputs, succeeded, failed).
        """
        outputs: list[dict] = []
        succeeded = 0
        failed = 0

        max_w = len(items) if max_workers is None else min(len(items), max_workers)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as executor:
            future_to_item = {}
            for item in items:
                future = executor.submit(agent_func, item, working_dir, backend)
                future_to_item[future] = item

            for future in concurrent.futures.as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result(timeout=0)
                except concurrent.futures.TimeoutError:
                    outputs.append({
                        "sub_task_id": item.get("id"),
                        "module_name": item.get("module_name", item.get("title", "?")),
                        "files": [],
                        "log_file": "",
                        "summary": f"Agent timed out after {timeout}s.",
                    })
                    failed += 1
                except Exception as e:
                    outputs.append({
                        "sub_task_id": item.get("id"),
                        "module_name": item.get("module_name", item.get("title", "?")),
                        "files": [],
                        "log_file": "",
                        "summary": f"Agent exception: {e}",
                    })
                    failed += 1
                else:
                    files = result.get("files", [])
                    if files or result.get("output_file"):
                        succeeded += 1
                    else:
                        failed += 1
                    outputs.append(result)

        # Retry failures once if enabled
        if retry_failures and failed > 0 and max_retries > 0:
            id_to_output = {out.get("sub_task_id"): out for out in outputs}
            failed_items = [
                it for it in items
                if not id_to_output.get(it.get("id"), {}).get("files")
                and not id_to_output.get(it.get("id"), {}).get("output_file")
            ]
            if failed_items:
                retry_outputs, retry_ok, retry_fail = BasePipeline._run_parallel_agents(
                    failed_items, agent_func, working_dir, backend, timeout, max_workers,
                    retry_failures=False,
                )
                # Merge retry results
                for ro in retry_outputs:
                    rid = ro.get("sub_task_id")
                    for i, out in enumerate(outputs):
                        if out.get("sub_task_id") == rid:
                            outputs[i] = ro
                            break
                succeeded += retry_ok
                failed = retry_fail

        return outputs, succeeded, failed


# ═══════════════════════════════════════════════════════════════════════════
# Coder Pipeline (coder ↔ reviewer loop)
# ═══════════════════════════════════════════════════════════════════════════

class MultiAgentState(TypedDict):
    messages: list[dict[str, Any]]
    current_agent: str
    requirement: str
    working_dir: str
    review_passed: bool
    iteration: int
    backend: str


class CoderRole(AgentRole):
    """Generates code to fulfill a requirement."""

    agent_name = "coder"
    max_steps = 15

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        return ToolRegistry.to_dicts(ToolRegistry.coder_tools())

    def build_task(self, backend: str, requirement: str = "", **context: Any) -> str:
        return (
            f"Implement the following requirement by writing code files:\n\n"
            f"{requirement}\n\n"
            f"## Design Guidelines\n"
            f"- Separate core logic into testable functions with parameters — do NOT hardcode values inside functions.\n"
            f"- main() must accept CLI arguments (sys.argv or argparse) so the script is reusable, not a one-shot demo.\n"
            f"- A user running `python script.py` with no arguments should get sensible defaults AND a help/usage path.\n"
            f"- Write tests that cover both default behavior and custom inputs.\n\n"
            f"Write the code, run the tests, and mark TASK_COMPLETE when done."
        )


class ReviewerRole(AgentRole):
    """Reviews generated code for bugs and correctness."""

    agent_name = "reviewer"
    max_steps = 10

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        return ToolRegistry.to_dicts(ToolRegistry.reviewer_tools())

    def build_task(self, backend: str, requirement: str = "", **context: Any) -> str:
        return (
            f"Review the code written for this requirement:\n\n"
            f"{requirement}\n\n"
            f"Read the files, check for bugs, test if possible. "
            f"If correct, output REVIEW_PASSED then TASK_COMPLETE. "
            f"If issues found, describe them and output REVIEW_FAILED then TASK_COMPLETE."
        )


class CoderPipeline(BasePipeline):
    """Coder generates code, Reviewer reviews it. Max 5 cycles."""

    name = "coder"
    default_output_dir = "coder_output"

    def _decompose(self, input_spec: str) -> list[dict[str, Any]]:
        # Coder pipeline has no decomposition — return requirement as-is
        return [{"id": 1, "title": "Requirement", "description": input_spec}]

    def _display_decomposition(self, sub_tasks: list[dict]):
        print(f"\nRequirement: {sub_tasks[0]['description'][:200]}")

    def _build_initial_state(self, input_spec: str, sub_tasks: list[dict]) -> dict:
        return {
            "messages": [],
            "current_agent": "coder",
            "requirement": input_spec,
            "working_dir": self.working_dir,
            "review_passed": False,
            "iteration": 0,
            "backend": self.backend,
        }

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(MultiAgentState)
        backend = self.backend
        working_dir = self.working_dir

        coder_role = CoderRole()
        reviewer_role = ReviewerRole()

        def _coder_node(state: MultiAgentState) -> dict:
            result = coder_role.execute(
                working_dir=state["working_dir"],
                backend=state.get("backend", backend),
                requirement=state["requirement"],
            )
            # result is AgentResult — serialize messages
            serialized = []
            for m in result.messages:
                serialized.append({
                    "role": type(m).__name__,
                    "content": m.content if hasattr(m, "content") else str(m),
                })
            return {
                "messages": serialized,
                "current_agent": "reviewer",
                "review_passed": False,
                "iteration": state["iteration"] + 1,
            }

        def _reviewer_node(state: MultiAgentState) -> dict:
            result = reviewer_role.execute(
                working_dir=state["working_dir"],
                backend=state.get("backend", backend),
                requirement=state["requirement"],
            )
            review_passed = False
            for m in reversed(result.messages):
                if type(m).__name__ != "AIMessage":
                    continue
                content = m.content if hasattr(m, "content") else str(m)
                if "REVIEW_PASSED" in content and "REVIEW_FAILED" not in content:
                    review_passed = True
                    break
                elif "REVIEW_FAILED" in content:
                    review_passed = False
                    break
            serialized = []
            for m in result.messages:
                serialized.append({
                    "role": type(m).__name__,
                    "content": m.content if hasattr(m, "content") else str(m),
                })
            return {
                "messages": serialized,
                "review_passed": review_passed,
                "current_agent": "coder" if not review_passed else "reviewer",
                "iteration": state["iteration"] + 1,
            }

        def _router(state: MultiAgentState) -> Literal["coder", "reviewer", "__end__"]:
            if state["review_passed"]:
                return END
            if state["iteration"] >= 5:
                return END
            return state["current_agent"]

        workflow.add_node("coder", _coder_node)
        workflow.add_node("reviewer", _reviewer_node)
        workflow.set_entry_point("coder")
        workflow.add_conditional_edges("coder", _router, {"reviewer": "reviewer", "coder": "coder", END: END})
        workflow.add_conditional_edges("reviewer", _router, {"reviewer": "reviewer", "coder": "coder", END: END})
        return workflow.compile()

    def _print_results(self, final_state: dict):
        print("-" * 50)
        if final_state["review_passed"]:
            print("SUCCESS: Review passed!")
        else:
            print("FINISHED: Max iterations reached or review not passed.")
        print(f"Iterations: {final_state['iteration']}")
        print(f"Working directory: {self.working_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# Research Pipeline (head → workers → reviewer → writer)
# ═══════════════════════════════════════════════════════════════════════════

from research.head_agent import decompose_topic, _fallback_decompose
from research.worker_agent import research_subtask
from research.reviewer_agent import review_and_score
from research.writer import write_proposal

HEAD_TIMEOUT = 300
WORKER_TIMEOUT = 900
RESEARCH_MAX_VERSIONS = 6
RESEARCH_MAX_WORKER_RETRIES = 5


class ResearchState(TypedDict):
    topic: str
    working_dir: str
    backend: str
    sub_tasks: list[dict[str, Any]]
    worker_outputs: list[dict[str, Any]]
    scored_works: list[dict[str, Any]]
    latex_file: str
    status: str
    worker_stats: dict[str, int]
    version: int


class ResearchPipeline(BasePipeline):
    """Head decomposes topic → Workers research in parallel → Reviewer scores → Writer generates LaTeX."""

    name = "research"
    default_output_dir = "research_output"

    def _decompose(self, input_spec: str) -> list[dict[str, Any]]:
        return decompose_topic(input_spec, self.working_dir, self.backend)

    def _display_decomposition(self, sub_tasks: list[dict]):
        print(f"\nResearch sub-topics ({len(sub_tasks)}):")
        print("-" * 50)
        for t in sub_tasks:
            print(f"  [{t['id']}] {t['title']}")
            print(f"      {t['description'][:120]}")

    def _build_initial_state(self, input_spec: str, sub_tasks: list[dict]) -> dict:
        return {
            "topic": input_spec,
            "working_dir": self.working_dir,
            "backend": self.backend,
            "sub_tasks": sub_tasks,
            "worker_outputs": [],
            "scored_works": [],
            "latex_file": "",
            "status": "decomposed",  # skip head node
            "worker_stats": {"total": len(sub_tasks), "succeeded": 0, "failed": 0, "duplicates_skipped": 0},
            "version": 1,
        }

    def _try_load_resume_state(self, input_spec: str) -> dict | None:
        """Reconstruct Research state from decomposition.json and checkpoints."""
        wd = self.working_dir
        decomp_path = os.path.join(wd, "decomposition.json")
        if not os.path.exists(decomp_path):
            return None

        try:
            with open(decomp_path) as f:
                sub_tasks = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        if not sub_tasks or not isinstance(sub_tasks, list):
            return None

        log_dir = os.path.join(wd, "agent_log")

        # Scan worker checkpoints: determine max version AND actual success per worker
        max_version = 1
        worker_success: dict[str, bool] = {}  # worker_id → actually succeeded
        if os.path.isdir(log_dir):
            for fname in os.listdir(log_dir):
                m = re.match(r"worker_(\d+)_v(\d+)_checkpoint\.json", fname)
                if m:
                    wid = int(m.group(1))
                    ver = int(m.group(2))
                    if ver > max_version:
                        max_version = ver
                    # Check if this checkpoint declares success
                    try:
                        ck = json.load(open(os.path.join(log_dir, fname)))
                        if ck.get("success") or ck.get("has_written_output"):
                            worker_success[str(wid)] = True
                    except (json.JSONDecodeError, OSError):
                        pass

        # Build worker_outputs from .md files on disk, gated by checkpoint success
        worker_outputs: list[dict] = []
        for st in sub_tasks:
            sid = st.get("id", 0)
            title = st.get("title", "")
            sid_str = str(sid)
            has_checkpoint = any(
                fname.startswith(f"worker_{sid:02d}_v") and
                fname.endswith("_checkpoint.json")
                for fname in (os.listdir(log_dir) if os.path.isdir(log_dir) else [])
            )
            ok = worker_success.get(sid_str, False)
            output_file = ""
            # Worker succeeded if: checkpoint confirms it, OR no checkpoint but file exists
            if ok or not has_checkpoint:
                for fname in sorted(os.listdir(wd)) if os.path.isdir(wd) else []:
                    if fname.startswith(f"research_{sid:02d}_") and fname.endswith(".md"):
                        output_file = fname
                        break
                if not output_file:
                    prefix = f"research_{sid:02d}"
                    for fname in sorted(os.listdir(wd)) if os.path.isdir(wd) else []:
                        if fname.startswith(prefix) and fname.endswith(".md"):
                            output_file = fname
                            break

            worker_outputs.append({
                "sub_task_id": sid,
                "title": title,
                "output_file": output_file,
                "summary": "Resumed from disk" if output_file else "Pending",
            })

        # Scan for scoring report and latex
        scored_works: list[dict] = []
        scoring_path = os.path.join(wd, "scoring_report.json")
        if os.path.exists(scoring_path):
            try:
                with open(scoring_path) as f:
                    scored_works = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        latex_file = ""
        latex_path = os.path.join(wd, "research_proposal.tex")
        if os.path.exists(latex_path):
            latex_file = "research_proposal.tex"

        # Determine status
        succeeded = sum(1 for wo in worker_outputs if wo.get("output_file"))
        total = len(sub_tasks)

        if latex_file:
            status = "written"  # pipeline completed
        elif scored_works:
            status = "reviewed"
        elif succeeded == total:
            status = "researched"
        elif succeeded > 0:
            status = "worker_retry"
        else:
            status = "decomposed"

        return {
            "topic": input_spec,
            "working_dir": wd,
            "backend": self.backend,
            "sub_tasks": sub_tasks,
            "worker_outputs": worker_outputs,
            "scored_works": scored_works,
            "latex_file": latex_file,
            "status": status,
            "version": max(max_version, 1),
            "worker_stats": {
                "total": total,
                "succeeded": succeeded,
                "failed": total - succeeded,
                "duplicates_skipped": 0,
                "retries": max(max_version - 1, 0),
            },
        }

    def _build_graph(self) -> StateGraph:
        import hashlib

        workflow = StateGraph(ResearchState)
        backend = self.backend
        working_dir = self.working_dir

        def _head_node(state: ResearchState) -> dict:
            # Skip if already decomposed (from double-check or resume)
            if state.get("sub_tasks"):
                # Preserve incoming status (e.g. "worker_retry" on resume)
                return {"status": state.get("status") or "decomposed",
                        "worker_stats": state.get("worker_stats", {})}

            topic = state["topic"]
            sub_tasks: list[dict] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(decompose_topic, topic=topic, working_dir=working_dir, backend=backend)
                try:
                    sub_tasks = future.result(timeout=HEAD_TIMEOUT)
                except (concurrent.futures.TimeoutError, Exception):
                    sub_tasks = []
            if not sub_tasks or len(sub_tasks) < 2:
                sub_tasks = _fallback_decompose(topic)
            return {
                "sub_tasks": sub_tasks,
                "status": "decomposed",
                "worker_stats": {"total": len(sub_tasks), "succeeded": 0, "failed": 0, "duplicates_skipped": 0},
            }

        def _workers_node(state: ResearchState) -> dict:
            sub_tasks = state.get("sub_tasks", [])
            if not sub_tasks:
                return {"status": "error_no_subtasks"}

            version = state.get("version", 1)
            current_status = state.get("status", "")
            prev_outputs = state.get("worker_outputs", [])
            worker_retry_count = state.get("worker_stats", {}).get("retries", 0)

            # Map existing outputs by sub_task_id for retry preservation
            id_to_output: dict[int, dict] = {
                wo["sub_task_id"]: wo for wo in prev_outputs if wo.get("output_file")
            }

            # ── Determine which tasks to run ──
            if current_status == "worker_retry":
                # Retry only failed workers (no output file)
                failed = [
                    st for st in sub_tasks
                    if st["id"] not in id_to_output
                ]
                if not failed:
                    return {
                        "worker_outputs": list(id_to_output.values()),
                        "status": "researched",
                        "version": version,
                        "worker_stats": {"total": len(sub_tasks),
                                         "succeeded": len(id_to_output),
                                         "failed": 0, "duplicates_skipped": 0,
                                         "retries": version - 1},
                    }

                if worker_retry_count >= RESEARCH_MAX_WORKER_RETRIES:
                    print(f"\n[research worker retry] Max retries ({RESEARCH_MAX_WORKER_RETRIES}) reached. "
                          f"Proceeding with {len(id_to_output)}/{len(sub_tasks)} workers.")
                    for st in sub_tasks:
                        if st["id"] not in id_to_output:
                            id_to_output[st["id"]] = {
                                "sub_task_id": st["id"],
                                "title": st["title"],
                                "output_file": "",
                                "summary": "Worker failed after max retries.",
                            }
                    return {
                        "worker_outputs": list(id_to_output.values()),
                        "status": "researched_partial",
                        "version": version,
                        "worker_stats": {"total": len(sub_tasks),
                                         "succeeded": sum(1 for wo in id_to_output.values() if wo.get("output_file")),
                                         "failed": sum(1 for wo in id_to_output.values() if not wo.get("output_file")),
                                         "duplicates_skipped": 0,
                                         "retries": version - 1},
                    }

                print(f"\n[research version {version}] Retrying {len(failed)} failed worker(s): "
                      f"{[t['id'] for t in failed]}")
                new_version = version + 1
                tasks_to_run = failed
                # Preserve successful outputs
                kept_outputs = list(id_to_output.values())
                worker_retry_count += 1
            else:
                # Fresh start: run all workers
                new_version = version
                tasks_to_run = list(sub_tasks)
                kept_outputs = []
                worker_retry_count = 0

            # ── Execute tasks with dependency ordering ──
            outputs, succeeded, failed = BasePipeline._run_workers_with_deps(
                tasks_to_run,
                lambda item, wd, be: research_subtask(item, wd, be, version=new_version),
                working_dir, backend, WORKER_TIMEOUT,
            )

            # Verify output files exist
            for out in outputs:
                of = out.get("output_file", "")
                if of:
                    fp = os.path.join(working_dir, of)
                    if not os.path.exists(fp) or os.path.getsize(fp) == 0:
                        out["output_file"] = ""

            # Merge with preserved outputs
            all_outputs = kept_outputs + outputs

            # Dedup
            def _fp(text: str) -> str:
                return hashlib.md5(" ".join(text.lower().split())[:200].encode()).hexdigest()

            seen: set[str] = set()
            dupes = 0
            for out in all_outputs:
                s = out.get("summary", "")
                if not s:
                    continue
                f = _fp(s)
                if f in seen:
                    out["summary"] = "Skipped: duplicate output."
                    dupes += 1
                else:
                    seen.add(f)

            total_succeeded = sum(1 for wo in all_outputs if wo.get("output_file"))
            total_failed = len(sub_tasks) - total_succeeded

            # Retry if any workers failed and we haven't exceeded limits
            if total_failed > 0 and new_version < RESEARCH_MAX_VERSIONS and worker_retry_count < RESEARCH_MAX_WORKER_RETRIES:
                print(f"\n[research version {new_version}] {total_failed}/{len(sub_tasks)} workers failed — "
                      f"will retry failed workers.")
                return {
                    "worker_outputs": all_outputs,
                    "status": "worker_retry",
                    "version": new_version,
                    "worker_stats": {"total": len(sub_tasks), "succeeded": total_succeeded,
                                     "failed": total_failed, "duplicates_skipped": dupes,
                                     "retries": worker_retry_count},
                }

            return {
                "worker_outputs": all_outputs,
                "status": "researched" if total_succeeded > 0 else "researched_partial",
                "version": new_version,
                "worker_stats": {"total": len(sub_tasks), "succeeded": total_succeeded,
                                 "failed": total_failed, "duplicates_skipped": dupes,
                                 "retries": worker_retry_count},
            }

        def _reviewer_node(state: ResearchState) -> dict:
            outputs = state.get("worker_outputs", [])
            version = state.get("version", 1)
            reviewable = [wo for wo in outputs if wo.get("output_file") or (
                wo.get("summary") and "timed out" not in wo["summary"].lower()
                and "skipped" not in wo["summary"].lower()
            )]
            if not reviewable:
                return {"status": "error_no_reviewable"}

            try:
                scored = review_and_score(reviewable, state["topic"], working_dir, backend, version=version)
            except Exception:
                scored = []

            if not scored:
                # Fallback: rank by file existence and summary length
                scored = sorted(reviewable, key=lambda w: (
                    1 if w.get("output_file") and os.path.exists(os.path.join(working_dir, w["output_file"])) else 0,
                    len(w.get("summary", "")),
                ), reverse=True)
                for i, item in enumerate(scored):
                    item.setdefault("scores", {
                        "depth": 5, "accuracy": 5, "relevance": 5, "clarity": 5, "originality": 5,
                        "justification": "Auto-ranked (reviewer was unable to score).",
                    })
                    item["total_score"] = 25
                    item["rank"] = i + 1

            return {"scored_works": scored, "status": "reviewed"}

        def _writer_node(state: ResearchState) -> dict:
            scored = state.get("scored_works", [])
            version = state.get("version", 1)
            if not scored:
                return {"status": "error_no_scored_works"}
            try:
                latex_file = write_proposal(scored, state["topic"], working_dir, backend, version=version)
            except Exception:
                latex_file = ""

            # Merge all agent checkpoints on completion
            sub_tasks = state.get("sub_tasks", [])
            for st in sub_tasks:
                ckm = CheckpointManager(working_dir, f"worker_{st['id']:02d}")
                ckm.merge()
            # Also merge head and reviewer
            for name in ("head_decompose", "reviewer"):
                ckm = CheckpointManager(working_dir, name)
                ckm.merge()

            return {"latex_file": latex_file, "status": "written" if latex_file else "error_latex_failed"}

        flow = {"decomposed": "workers", "worker_retry": "workers", "researched": "reviewer", "researched_partial": "reviewer", "reviewed": "writer", "written": END}
        terminal = {"error_no_subtasks", "error_no_reviewable", "error_no_scored_works"}
        router = BasePipeline._status_router(flow, terminal)

        workflow.add_node("head", _head_node)
        workflow.add_node("workers", _workers_node)
        workflow.add_node("reviewer", _reviewer_node)
        workflow.add_node("writer", _writer_node)
        workflow.set_entry_point("head")
        for node in ("head", "workers", "reviewer", "writer"):
            workflow.add_conditional_edges(node, router, {"workers": "workers", "reviewer": "reviewer", "writer": "writer", END: END})

        return workflow.compile()

    def _print_results(self, final_state: dict):
        print("-" * 50)
        print(f"Status: {final_state['status']}")
        print(f"Sub-tasks: {len(final_state.get('sub_tasks', []))}")
        print(f"Worker outputs: {len(final_state.get('worker_outputs', []))}")
        scored = final_state.get("scored_works", [])
        print(f"Scored works: {len(scored)}")
        if scored:
            print("\n--- SCORED RESEARCH PROPOSALS ---")
            for i, item in enumerate(scored):
                print(f"  {i+1}. [{item.get('total_score', '?')}/50] {item.get('title', 'Untitled')}")
        latex = final_state.get("latex_file", "")
        if latex and os.path.exists(latex):
            print(f"\nLaTeX: {latex}")
        else:
            print("\nWarning: LaTeX file not generated.")
        print(f"\nOutputs in: {self.working_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# CoderPP Pipeline (head → workers → reviewer → organizer)
# ═══════════════════════════════════════════════════════════════════════════

from coderpp.head_agent import decompose_to_modules, _fallback_decompose as _coderpp_fallback, observe_workers
from coderpp.worker_agent import code_submodule
from coderpp.reviewer_agent import review_module
from coderpp.organizer import assemble_project

CPP_HEAD_TIMEOUT = 120
CPP_WORKER_TIMEOUT = 600
CPP_REVIEWER_TIMEOUT = 600
CPP_MAX_VERSIONS = 3
CPP_MAX_WORKER_RETRIES = 3


class CoderPPState(TypedDict):
    input_spec: str
    working_dir: str
    backend: str
    sub_tasks: list[dict[str, Any]]
    worker_outputs: list[dict[str, Any]]
    reviewed_modules: list[dict[str, Any]]
    project_dir: str
    status: str
    worker_stats: dict[str, int]
    version: int
    environment: str  # contents of ENVIRONMENT.md for workers


class CoderPPPipeline(BasePipeline):
    """Head decomposes → Workers code → Reviewer fixes → Organizer assembles project."""

    name = "coderpp"
    default_output_dir = "coderpp_output"

    def _decompose(self, input_spec: str) -> list[dict[str, Any]]:
        # Read spec files (.tex, .md) if provided as a file path
        spec = input_spec
        if os.path.exists(input_spec) and input_spec.endswith((".tex", ".md")):
            with open(input_spec) as f:
                content = f.read()
            if input_spec.endswith(".tex"):
                spec = f"Implement the ideas, future work, and optimizations described in this research proposal:\n\n{content[:8000]}"
            else:
                spec = f"Implement the pipeline, agent roles, and tests described in this specification:\n\n{content[:8000]}"
        return decompose_to_modules(spec, self.working_dir, self.backend)

    def _display_decomposition(self, sub_tasks: list[dict]):
        print(f"\nCode modules ({len(sub_tasks)}):")
        print("-" * 50)
        for t in sub_tasks:
            deps = t.get("dependencies", [])
            if deps:
                dep_names = []
                for d in deps:
                    if isinstance(d, str):
                        dep_names.append(d)
                    elif isinstance(d, int):
                        # Look up module name by ID
                        match = next((t2 for t2 in sub_tasks if t2.get("id") == d), None)
                        dep_names.append(match["module_name"] if match else str(d))
                    elif isinstance(d, dict):
                        dep_names.append(d.get("module_name", str(d)))
                    else:
                        dep_names.append(str(d))
                dep_str = f" (depends on: {', '.join(dep_names)})"
            else:
                dep_str = ""
            print(f"  [{t.get('id', '?')}] {t.get('module_name', '?')}{dep_str}")
            print(f"      {t.get('description', '')[:120]}")
            files = t.get("files_to_create", [])
            if files:
                print(f"      Files: {', '.join(files)}")

    def _build_initial_state(self, input_spec: str, sub_tasks: list[dict]) -> dict:
        return {
            "input_spec": input_spec,
            "working_dir": self.working_dir,
            "backend": self.backend,
            "sub_tasks": sub_tasks,
            "worker_outputs": [],
            "reviewed_modules": [],
            "project_dir": "",
            "status": "decomposed",
            "worker_stats": {"total": len(sub_tasks), "succeeded": 0, "failed": 0, "retries": 0},
            "version": 1,
            "environment": "",
        }

    def _try_load_resume_state(self, input_spec: str) -> dict | None:
        """Reconstruct CoderPP state from decomposition.json and checkpoints."""
        wd = self.working_dir
        decomp_path = os.path.join(wd, "decomposition.json")
        if not os.path.exists(decomp_path):
            return None

        try:
            with open(decomp_path) as f:
                sub_tasks = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        if not sub_tasks or not isinstance(sub_tasks, list):
            return None

        log_dir = os.path.join(wd, "agent_log")
        modules_dir = os.path.join(wd, "modules")

        # Scan worker checkpoints: determine max version AND actual success per worker
        max_version = 1
        worker_success: dict[str, bool] = {}  # module_name → actually succeeded
        if os.path.isdir(log_dir):
            for fname in os.listdir(log_dir):
                m = re.match(r"coderpp_worker_(\d+)_v(\d+)_checkpoint\.json", fname)
                if m:
                    wid = int(m.group(1))
                    ver = int(m.group(2))
                    if ver > max_version:
                        max_version = ver
                    # Check if this checkpoint declares success
                    try:
                        ck = json.load(open(os.path.join(log_dir, fname)))
                        if ck.get("success") or ck.get("has_written_output"):
                            worker_success[str(wid)] = True
                    except (json.JSONDecodeError, OSError):
                        pass

        # Build worker_outputs from files on disk, gated by checkpoint success.
        # Three cases:
        #   1. Checkpoint exists with success → count files on disk
        #   2. Checkpoint exists without success → worker failed, ignore files
        #   3. No checkpoint exists → trust files on disk as evidence of success
        worker_outputs: list[dict] = []
        for st in sub_tasks:
            name = st.get("module_name", "")
            mid = st.get("id", 0)
            mid_str = str(mid)
            has_checkpoint = any(
                fname.startswith(f"coderpp_worker_{mid:02d}_v") and
                fname.endswith("_checkpoint.json")
                for fname in (os.listdir(log_dir) if os.path.isdir(log_dir) else [])
            )
            # Worker succeeded if: checkpoint confirms it, OR no checkpoint but
            # real implementation files exist (not just __init__.py skeleton).
            ok = worker_success.get(mid_str, False)
            files: list[str] = []
            mod_dir = os.path.join(modules_dir, name)
            if ok or (not has_checkpoint and os.path.isdir(mod_dir)):
                if os.path.isdir(mod_dir):
                    for f in os.listdir(mod_dir):
                        if f.endswith(".py") and "__pycache__" not in f:
                            files.append(f"modules/{name}/{f}")
            # If no checkpoint and only skeleton files (__init__.py only, no test
            # files, no implementation), treat as NOT succeeded — worker needs retry.
            if not ok and not has_checkpoint and files:
                impl_files = [f for f in files if os.path.basename(f) != "__init__.py"]
                if not impl_files:
                    files = []
            worker_outputs.append({
                "sub_task_id": mid,
                "module_name": name,
                "files": files,
                "log_file": "",
                "summary": "Resumed from disk" if files else "Pending",
            })

        # Scan reviewer checkpoints for previously reviewed modules
        reviewed_modules: list[dict] = []
        if os.path.isdir(log_dir):
            reviewer_versions: dict[str, int] = {}
            for fname in sorted(os.listdir(log_dir)):
                m = re.match(r"coderpp_reviewer_(\d+)_v(\d+)_checkpoint\.json", fname)
                if m:
                    mod_id = int(m.group(1))
                    ver = int(m.group(2))
                    reviewer_versions[str(mod_id)] = max(
                        reviewer_versions.get(str(mod_id), 0), ver,
                    )
                    # Try to load the reviewer result
                    rpath = os.path.join(log_dir, fname)
                    try:
                        rdata = json.load(open(rpath))
                        extra = rdata.get("extra", {})
                        reviewed_modules.append({
                            "sub_task_id": mod_id,
                            "module_name": extra.get("module_name", ""),
                            "passed": extra.get("passed", False),
                            "files": extra.get("files", []),
                            "feedback": extra.get("feedback", ""),
                        })
                    except (json.JSONDecodeError, OSError):
                        pass

        # Read environment
        environment = ""
        env_path = os.path.join(wd, "ENVIRONMENT.md")
        if os.path.exists(env_path):
            try:
                with open(env_path) as f:
                    environment = f.read()
            except OSError:
                pass

        # Detect project directory from organizer output
        project_dir = ""
        project_path = os.path.join(wd, "project")
        if os.path.isdir(project_path):
            project_dir = "project"

        # Determine status
        succeeded = sum(1 for wo in worker_outputs if wo["files"])
        total = len(sub_tasks)
        all_have_files = succeeded == total

        if reviewed_modules and all_have_files:
            status = "worker_all_success"  # reviewer needs to re-check or organizer
        elif all_have_files:
            status = "worker_all_success"
        elif succeeded > 0:
            status = "worker_retry"
        else:
            status = "decomposed"

        return {
            "input_spec": input_spec,
            "working_dir": wd,
            "backend": self.backend,
            "sub_tasks": sub_tasks,
            "worker_outputs": worker_outputs,
            "reviewed_modules": reviewed_modules,
            "project_dir": project_dir,
            "status": status,
            "version": max(max_version, 1),
            "environment": environment,
            "worker_stats": {
                "total": total,
                "succeeded": succeeded,
                "failed": total - succeeded,
                "retries": max(max_version - 1, 0),
                "worker_retries": max(max_version - 1, 0),
            },
        }

    def _build_graph(self) -> StateGraph:
        import hashlib

        workflow = StateGraph(CoderPPState)
        backend = self.backend
        working_dir = self.working_dir
        resume = self.resume

        def _read_environment(wd: str) -> str:
            env_path = os.path.join(wd, "ENVIRONMENT.md")
            if os.path.exists(env_path):
                try:
                    with open(env_path) as f:
                        return f.read()
                except (OSError, IOError):
                    pass
            return ""

        def _head_node(state: CoderPPState) -> dict:
            # Skip if already decomposed — but still read environment from disk
            if state.get("sub_tasks"):
                env = state.get("environment", "") or _read_environment(working_dir)
                # Preserve incoming status (e.g. "worker_retry" on resume)
                return {"status": state.get("status") or "decomposed",
                        "worker_stats": state.get("worker_stats", {}),
                        "environment": env}

            spec = state["input_spec"]
            if spec.endswith(".tex") and os.path.exists(spec):
                with open(spec) as f:
                    tex = f.read()
                spec = f"Implement the ideas, future work, and optimizations described in this research proposal:\n\n{tex[:8000]}"

            sub_tasks: list[dict] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(decompose_to_modules, input_spec=spec, working_dir=working_dir, backend=backend)
                try:
                    sub_tasks = future.result(timeout=CPP_HEAD_TIMEOUT)
                except (concurrent.futures.TimeoutError, Exception):
                    sub_tasks = []
            if not sub_tasks or len(sub_tasks) < 2:
                sub_tasks = _coderpp_fallback(spec)

            # Read ENVIRONMENT.md written by the head agent
            environment = _read_environment(working_dir)
            if environment:
                print(f"\n[head] Environment documented ({len(environment)} bytes)")
                for line in environment.split('\n'):
                    stripped = line.strip()
                    if stripped.startswith('- Path:') or stripped.startswith('- Version:') or stripped.startswith('- Environment:'):
                        print(f"  {stripped}")
            else:
                print("\n[head] Warning: ENVIRONMENT.md not found — workers may use inconsistent environments.")

            return {
                "sub_tasks": sub_tasks,
                "status": "decomposed",
                "worker_stats": {"total": len(sub_tasks), "succeeded": 0, "failed": 0, "retries": 0},
                "environment": environment,
            }

        def _workers_node(state: CoderPPState) -> dict:
            sub_tasks = state.get("sub_tasks", [])
            if not sub_tasks:
                return {"status": "error_no_subtasks"}

            env = state.get("environment", "")

            # Wrapper that bakes environment and version into every worker call
            def _worker_func(item, wd, be):
                return code_submodule(item, wd, be, environment=env, version=version)

            version = state.get("version", 1)
            current_status = state.get("status", "")
            previously_reviewed = state.get("reviewed_modules", [])
            prev_outputs = state.get("worker_outputs", [])
            worker_retry_count = state.get("worker_stats", {}).get("worker_retries", 0)

            # Map existing outputs by module name
            all_outputs_map: dict[str, dict] = {
                wo["module_name"]: wo for wo in prev_outputs if wo.get("files")
            }

            # ── Determine which tasks to run ──
            tasks_to_run: list[dict] = []
            reused_outputs: list[dict] = []

            if version > 1 and previously_reviewed:
                # ── Post-reviewer retry ──
                # Circuit breaker: if we've already hit max versions, don't retry.
                if version > CPP_MAX_VERSIONS:
                    print(f"\n[worker gate] Version {version} exceeds max ({CPP_MAX_VERSIONS}) — "
                          f"proceeding with current results.")
                    return {
                        "worker_outputs": list(all_outputs_map.values()),
                        "status": "worker_skip_observer",
                        "version": version,
                        "worker_stats": {"total": len(sub_tasks),
                                         "succeeded": len(all_outputs_map),
                                         "failed": len(sub_tasks) - len(all_outputs_map),
                                         "retries": version - 1,
                                         "worker_retries": worker_retry_count},
                    }

                failed_rm = {rm["module_name"]: rm for rm in previously_reviewed if not rm.get("passed")}
                worker_retry_names: set[str] = set()
                reviewer_retry_names: set[str] = set()
                for name, rm in failed_rm.items():
                    files = rm.get("files", [])
                    has_py_files = any(
                        f.endswith(".py") and "/test_" not in f
                        and os.path.basename(f) != "__init__.py"
                        for f in files
                    )
                    if has_py_files:
                        reviewer_retry_names.add(name)
                    else:
                        worker_retry_names.add(name)

                if reviewer_retry_names:
                    print(f"\n[version {version}] Reviewer failed for: {', '.join(sorted(reviewer_retry_names))} "
                          f"(worker code is fine — reviewer will re-check these modules)")
                    for wo in prev_outputs:
                        if wo.get("module_name") in reviewer_retry_names:
                            reused_outputs.append(wo)

                if worker_retry_names:
                    print(f"\n[version {version}] Worker needs retry for: {', '.join(sorted(worker_retry_names))}"
                          f" (files missing or incomplete)")
                    tasks_to_run = [st for st in sub_tasks if st["module_name"] in worker_retry_names]
                elif not reviewer_retry_names:
                    return {"status": "error_no_subtasks"}
                # If only reviewer retries: tasks_to_run stays empty, go straight to worker_all_success

            elif current_status == "worker_retry":
                # ── Pre-reviewer worker retry: resume failed workers from checkpoint ──
                failed = [
                    st for st in sub_tasks
                    if st["module_name"] not in all_outputs_map
                ]
                if not failed:
                    # All workers have files now
                    return {
                        "worker_outputs": list(all_outputs_map.values()),
                        "status": "worker_all_success",
                        "worker_stats": {"total": len(sub_tasks),
                                         "succeeded": len(all_outputs_map),
                                         "failed": 0, "retries": version - 1,
                                         "worker_retries": worker_retry_count},
                    }

                if worker_retry_count >= CPP_MAX_WORKER_RETRIES:
                    print(f"\n[worker gate] Max worker retries ({CPP_MAX_WORKER_RETRIES}) reached. "
                          f"Proceeding with {len(all_outputs_map)}/{len(sub_tasks)} workers.")
                    final_outputs = list(all_outputs_map.values())
                    # Mark remaining as failed (no files)
                    for st in sub_tasks:
                        if st["module_name"] not in all_outputs_map:
                            final_outputs.append({
                                "sub_task_id": st["id"],
                                "module_name": st["module_name"],
                                "files": [],
                                "log_file": "",
                                "summary": "Worker failed after max retries.",
                            })
                    return {
                        "worker_outputs": final_outputs,
                        "status": "worker_all_success",  # proceed with what we have
                        "worker_stats": {"total": len(sub_tasks),
                                         "succeeded": len(all_outputs_map),
                                         "failed": len(sub_tasks) - len(all_outputs_map),
                                         "retries": version - 1,
                                         "worker_retries": worker_retry_count},
                    }

                print(f"\n[worker gate] Retrying {len(failed)} failed worker(s): "
                      f"{[t['module_name'] for t in failed]}")
                tasks_to_run = failed
                worker_retry_count += 1
                version += 1  # bump version so agents auto-resume from previous checkpoint
                # Preserve outputs from successful workers
                reused_outputs = list(all_outputs_map.values())

            else:
                # ── Fresh start: run all workers ──
                tasks_to_run = list(sub_tasks)
                reused_outputs = []

            # ── Execute tasks with dependency ordering ──
            if tasks_to_run:
                levels = BasePipeline._topological_levels(tasks_to_run)

                for level_idx, level_tasks in enumerate(levels):
                    names = [t["module_name"] for t in level_tasks]
                    if len(levels) > 1:
                        print(f"\n  [dependency level {level_idx}] Running: {names}")

                    results, succeeded, failed = BasePipeline._run_parallel_agents(
                        level_tasks, _worker_func, working_dir, backend,
                        CPP_WORKER_TIMEOUT, retry_failures=True, max_retries=1,
                    )

                    for r in results:
                        name = r.get("module_name")
                        if name:
                            all_outputs_map[name] = r

                    # Validate file content: workers may report success but
                    # produce empty/trivial files (e.g. when JSON parsing fails).
                    for r in results:
                        name = r.get("module_name", "")
                        files = r.get("files", [])
                        if name and files:
                            empty_files = []
                            for f in files:
                                fpath = os.path.join(working_dir, f)
                                if os.path.isfile(fpath) and os.path.getsize(fpath) < 100:
                                    empty_files.append(f)
                            if empty_files:
                                print(f"  [validate] {name}: {len(empty_files)} empty/skeletal file(s): "
                                      f"{[os.path.basename(ef) for ef in empty_files]}")
                                r["files"] = [f for f in files if f not in empty_files]
                                all_outputs_map[name] = r

                    # Check if this level completed successfully
                    level_failed = [r for r in results if not r.get("files")]
                    if level_failed and current_status != "reviewed_retry":
                        # Pre-reviewer: don't proceed to next level until current succeeds
                        failed_names = [r["module_name"] for r in level_failed]
                        print(f"\n  [dependency level {level_idx}] Failed: {failed_names}")
                        print(f"  Will retry from checkpoints on next iteration.")
                        final_outputs = list(all_outputs_map.values())
                        return {
                            "worker_outputs": final_outputs,
                            "status": "worker_retry",
                            "version": version,
                            "worker_stats": {"total": len(sub_tasks),
                                             "succeeded": sum(1 for wo in final_outputs if wo.get("files")),
                                             "failed": len(sub_tasks) - sum(1 for wo in final_outputs if wo.get("files")),
                                             "retries": version - 1,
                                             "worker_retries": worker_retry_count},
                        }

            # ── Build final output ──
            final_outputs = list(all_outputs_map.values())
            all_have_files = all(wo.get("files") for wo in final_outputs)

            # For post-reviewer retry: skip observer, go straight to reviewer.
            if version > 1 and previously_reviewed:
                succeeded = sum(1 for wo in final_outputs if wo.get("files"))
                failed = len(sub_tasks) - succeeded
                if failed > 0:
                    print(f"\n[version {version}] {succeeded}/{len(sub_tasks)} workers succeeded, "
                          f"{failed} failed — proceeding to reviewer.")
                return {
                    "worker_outputs": final_outputs,
                    "status": "worker_skip_observer",
                    "version": version,
                    "worker_stats": {"total": len(sub_tasks),
                                     "succeeded": succeeded,
                                     "failed": failed,
                                     "retries": version - 1,
                                     "worker_retries": worker_retry_count},
                }

            return {
                "worker_outputs": final_outputs,
                "status": "worker_all_success" if all_have_files else "worker_retry",
                "version": version,
                "worker_stats": {"total": len(sub_tasks),
                                 "succeeded": sum(1 for wo in final_outputs if wo.get("files")),
                                 "failed": len(sub_tasks) - sum(1 for wo in final_outputs if wo.get("files")),
                                 "retries": version - 1,
                                 "worker_retries": worker_retry_count},
            }

        def _reviewer_node(state: CoderPPState) -> dict:
            outputs = state.get("worker_outputs", [])
            version = state.get("version", 1)
            prev_reviewed = state.get("reviewed_modules", [])

            # On retry: only review modules that failed last time
            if version > 1 and prev_reviewed:
                passed_names = {rm["module_name"] for rm in prev_reviewed if rm.get("passed")}
                retry_names = {rm["module_name"] for rm in prev_reviewed if not rm.get("passed")}
                # Only review modules that failed, have files, and need re-review
                reviewable = [
                    wo for wo in outputs
                    if wo.get("files") and wo.get("module_name") in retry_names
                ]
                if not reviewable:
                    # All previously-failed modules now missing files → need worker retry
                    return {"reviewed_modules": prev_reviewed, "status": "reviewed_retry"}
                print(f"\n[version {version}] Re-reviewing {len(reviewable)} failed module(s): "
                      f"{[r['module_name'] for r in reviewable]}")
            else:
                reviewable = [wo for wo in outputs if wo.get("files")]
                if not reviewable:
                    return {"status": "error_no_reviewable"}

            def _review_func(item, wd, be):
                return review_module(item, wd, be, version=version)

            reviewed, _, _ = BasePipeline._run_parallel_agents(
                reviewable, _review_func, working_dir, backend,
                CPP_REVIEWER_TIMEOUT, retry_failures=True, max_retries=1,
            )

            # Merge newly-reviewed results with previously-passed modules
            if version > 1 and prev_reviewed:
                passed_prev = {rm["module_name"]: rm for rm in prev_reviewed if rm.get("passed")}
                newly_reviewed_names = {rm["module_name"] for rm in reviewed}
                for name, prev_rm in passed_prev.items():
                    if name not in newly_reviewed_names:
                        reviewed.append(prev_rm)
                passed_count = sum(1 for r in reviewed if r.get("passed"))
            else:
                passed_count = sum(1 for r in reviewed if r.get("passed"))

            total = len(reviewed)

            print(f"\n[version {version}] Review: {passed_count}/{total} modules passed")
            for r in reviewed:
                verdict = "PASSED" if r.get("passed") else "FAILED"
                print(f"  [{r['module_name']}] {verdict}")

            # Determine status for routing
            if passed_count == total and total > 0:
                return {"reviewed_modules": reviewed, "status": "reviewed_all_passed"}
            elif version >= CPP_MAX_VERSIONS:
                print(f"\n[version {version}] Max versions ({CPP_MAX_VERSIONS}) reached — proceeding with partial success.")
                return {"reviewed_modules": reviewed, "status": "reviewed_max_versions"}
            else:
                return {"reviewed_modules": reviewed, "status": "reviewed_retry"}

        def _organizer_node(state: CoderPPState) -> dict:
            modules = state.get("reviewed_modules", []) or state.get("worker_outputs", [])
            if not modules:
                return {"status": "error_no_modules"}
            # Only use passed modules for assembly
            passed_modules = [m for m in modules if m.get("passed")]
            if not passed_modules:
                # Fall back to all modules if none passed (last resort)
                passed_modules = modules
                print("\nWarning: No modules passed review. Assembling with all available modules.")
            else:
                print(f"\nAssembling project from {len(passed_modules)}/{len(modules)} passed modules.")
            try:
                project_dir = assemble_project(passed_modules, state["input_spec"], working_dir, backend)
            except Exception:
                project_dir = ""
            return {"project_dir": project_dir, "status": "assembled" if project_dir else "error_assembly_failed"}

        # Flow: workers run with dependency ordering; must all succeed before observer.
        # Observer runs after workers, before reviewer — head agent spies on progress.
        # Worker retry loops until all produce files (or max retries reached).
        # Reviewer retry loops back to workers for failed modules (up to max versions).
        flow = {
            "decomposed": "workers",
            "worker_all_success": "observer",
            "worker_retry": "workers",
            "worker_skip_observer": "reviewer",  # post-reviewer retry: skip re-observing
            "observed": "reviewer",
            "reviewed_all_passed": "organizer",
            "reviewed_max_versions": "organizer",
            "reviewed_retry": "workers",
            "assembled": END,
        }
        terminal = {"error_no_subtasks", "error_no_reviewable", "error_no_modules", "error_assembly_failed"}
        router = BasePipeline._status_router(flow, terminal)

        def _observer_node(state: CoderPPState) -> dict:
            outputs = state.get("worker_outputs", [])
            sub_tasks = state.get("sub_tasks", [])
            reviewable = [wo for wo in outputs if wo.get("files")]
            if not reviewable or not sub_tasks:
                return {"status": "observed"}
            print(f"\n[head] Observing {len(reviewable)} worker(s)...")
            try:
                obs_path = observe_workers(outputs, sub_tasks, working_dir, backend)
                if obs_path:
                    print(f"  [head] Observations written to {obs_path}")
                else:
                    print("  [head] Observer did not produce output.")
            except Exception as e:
                print(f"  [head] Observer failed: {e}")
            return {"status": "observed"}

        # Post-review hook: increment version on retry, merge checkpoints on completion
        def _reviewer_with_version(state: CoderPPState) -> dict:
            result = _reviewer_node(state)
            newly_reviewed = result.get("reviewed_modules", [])
            version = state.get("version", 1)

            # Circuit breaker: cap reviewer retries so pipeline doesn't loop forever.
            if result.get("status") == "reviewed_retry":
                if version >= CPP_MAX_VERSIONS:
                    print(f"\n[version {version}] Max reviewer versions ({CPP_MAX_VERSIONS}) reached — "
                          f"proceeding to organizer with current results.")
                    result["status"] = "reviewed_max_versions"
                else:
                    result["version"] = version + 1

            is_final = result.get("status") not in ("reviewed_retry",)

            # On final version or module passed: merge all checkpoints
            if is_final:
                sub_tasks = state.get("sub_tasks", [])
                if sub_tasks:
                    print(f"\n[version {version}] Final — merging worker and reviewer checkpoints:")
                    for st in sub_tasks:
                        sub_id = st["id"]
                        for prefix in ("coderpp_worker_", "coderpp_reviewer_"):
                            ckm = CheckpointManager(working_dir, f"{prefix}{sub_id:02d}")
                            merge_path = ckm.merge()
                            if merge_path:
                                print(f"  [merge] {ckm.agent_name} -> {os.path.basename(merge_path)}")

            return result

        workflow.add_node("head", _head_node)
        workflow.add_node("workers", _workers_node)
        workflow.add_node("observer", _observer_node)
        workflow.add_node("reviewer", _reviewer_with_version)
        workflow.add_node("organizer", _organizer_node)
        workflow.set_entry_point("head")
        for node in ("head", "workers", "observer", "reviewer", "organizer"):
            workflow.add_conditional_edges(node, router, {"workers": "workers", "observer": "observer", "reviewer": "reviewer", "organizer": "organizer", END: END})

        return workflow.compile()

    def _print_results(self, final_state: dict):
        print("-" * 50)
        print(f"Status: {final_state['status']}")
        print(f"Version: {final_state.get('version', 1)}")
        stats = final_state.get("worker_stats", {})
        print(f"Workers: {stats.get('succeeded', 0)}/{stats.get('total', 0)} succeeded (retries: {stats.get('retries', 0)})")
        reviewed = final_state.get("reviewed_modules", [])
        print(f"Reviewed modules: {len(reviewed)}")
        passed = 0
        for rm in reviewed:
            verdict = "PASSED" if rm.get("passed") else "FAILED"
            if rm.get("passed"):
                passed += 1
            files = ", ".join(rm.get("files", []))
            print(f"  [{rm['module_name']}] {verdict}: {files}")
        print(f"Review summary: {passed}/{len(reviewed)} passed")
        project_dir = final_state.get("project_dir", "")
        if project_dir:
            print(f"\nProject at: {os.path.join(self.working_dir, project_dir)}")
        else:
            print("\nWarning: Project assembly incomplete.")
        print(f"\nOutputs in: {self.working_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# Topology Optimizer Pipeline
# ═══════════════════════════════════════════════════════════════════════════


class TopologyState(TypedDict):
    input_spec: str
    working_dir: str
    backend: str
    complexity_factors: dict[str, Any]
    candidate_topologies: list[dict[str, Any]]
    evaluated_topologies: list[dict[str, Any]]
    topology_spec: dict[str, Any]
    status: str


class TopologyPipeline(BasePipeline):
    """Analyze task → Design topologies → Evaluate → Write final spec.

    A linear 4-stage pipeline with no retry loops. Each stage is an
    AgentRole subclass: TopologyAnalyzerRole, TopologyDesignerRole,
    TopologyEvaluatorRole, TopologyWriterRole.
    """

    name = "topology"
    default_output_dir = "topology_output"

    def _decompose(self, input_spec: str) -> list[dict[str, Any]]:
        """No traditional decomposition — the pipeline graph handles everything."""
        return []

    def _display_decomposition(self, sub_tasks: list[dict]):
        print("Topology Optimizer: analyzing task and designing optimal agent topology...")
        print(f"Backend: {self.backend}")

    def _build_initial_state(self, input_spec: str, sub_tasks: list[dict]) -> dict:
        return {
            "input_spec": input_spec,
            "working_dir": self.working_dir,
            "backend": self.backend,
            "complexity_factors": {},
            "candidate_topologies": [],
            "evaluated_topologies": [],
            "topology_spec": {},
            "status": "initialized",
        }

    def _build_graph(self):
        working_dir = self.working_dir
        backend = self.backend

        # ── Analyzer node ──────────────────────────────────────────────
        def _analyzer_node(state: TopologyState) -> dict:
            print("\n[analyzer] Assessing task complexity...")
            try:
                role = TopologyAnalyzerRole()
                factors = role.execute(
                    working_dir=working_dir, backend=backend,
                    input_spec=state["input_spec"],
                )
                print(f"  Overall complexity: {factors.get('overall_complexity', 'unknown')}")
                return {"complexity_factors": factors, "status": "analyzed"}
            except Exception as e:
                print(f"  [analyzer] Failed: {e}")
                return {"status": "error_analysis_failed"}

        # ── Designer node ─────────────────────────────────────────────
        def _designer_node(state: TopologyState) -> dict:
            print("\n[designer] Proposing candidate topologies...")
            try:
                role = TopologyDesignerRole()
                topologies = role.execute(
                    working_dir=working_dir, backend=backend,
                    complexity_factors=state.get("complexity_factors", {}),
                    input_spec=state["input_spec"],
                )
                print(f"  Proposed {len(topologies)} candidate(s)")
                for t in topologies:
                    print(f"    - {t.get('name', '?')} ({t.get('pattern', '?')}): {len(t.get('agents', []))} agents")
                return {"candidate_topologies": topologies, "status": "designed"}
            except Exception as e:
                print(f"  [designer] Failed: {e}")
                return {"status": "error_design_failed"}

        # ── Evaluator node ────────────────────────────────────────────
        def _evaluator_node(state: TopologyState) -> dict:
            print("\n[evaluator] Scoring candidate topologies...")
            try:
                role = TopologyEvaluatorRole()
                evaluated = role.execute(
                    working_dir=working_dir, backend=backend,
                    candidate_topologies=state.get("candidate_topologies", []),
                )
                for e in evaluated:
                    print(f"  {e.get('name', '?')}: {e.get('total_score', 0)}/50")
                return {"evaluated_topologies": evaluated, "status": "evaluated"}
            except Exception as e:
                print(f"  [evaluator] Failed: {e}")
                return {"status": "error_evaluation_failed"}

        # ── Writer node ───────────────────────────────────────────────
        def _writer_node(state: TopologyState) -> dict:
            print("\n[writer] Producing final topology spec...")
            try:
                role = TopologyWriterRole()
                result = role.execute(
                    working_dir=working_dir, backend=backend,
                    evaluated_topologies=state.get("evaluated_topologies", []),
                    candidate_topologies=state.get("candidate_topologies", []),
                    input_spec=state["input_spec"],
                )
                spec = result.get("spec", {})
                print(f"  Recommended: {spec.get('recommended_topology', '?')} ({spec.get('total_score', 0)}/50)")
                print(f"  Spec: {result.get('spec_path', '?')}")
                print(f"  Report: {result.get('report_path', '?')}")
                return {"topology_spec": result, "status": "written"}
            except Exception as e:
                print(f"  [writer] Failed: {e}")
                return {"status": "error_writer_failed"}

        workflow = StateGraph(TopologyState)

        workflow.add_node("analyzer", _analyzer_node)
        workflow.add_node("designer", _designer_node)
        workflow.add_node("evaluator", _evaluator_node)
        workflow.add_node("writer", _writer_node)

        workflow.set_entry_point("analyzer")

        flow = {
            "initialized": "analyzer",
            "analyzed": "designer",
            "designed": "evaluator",
            "evaluated": "writer",
            "written": END,
        }
        terminal = {"error_analysis_failed", "error_design_failed", "error_evaluation_failed", "error_writer_failed"}
        router = BasePipeline._status_router(flow, terminal)

        for node in ("analyzer", "designer", "evaluator", "writer"):
            workflow.add_conditional_edges(node, router, {
                "analyzer": "analyzer", "designer": "designer",
                "evaluator": "evaluator", "writer": "writer", END: END,
            })

        return workflow.compile()

    def _print_results(self, final_state: dict):
        print("-" * 50)
        print(f"Status: {final_state['status']}")
        spec = final_state.get("topology_spec", {})
        if spec.get("spec"):
            s = spec["spec"]
            print(f"Recommended Topology: {s.get('recommended_topology', '?')}")
            print(f"Design Pattern: {s.get('design_pattern', '?')}")
            print(f"Total Score: {s.get('total_score', 0)}/50")
            agents = s.get("agents", [])
            print(f"Agents: {len(agents)}")
            for a in agents:
                name = a.get("agent_name") or a.get("name") or "?"
                role = a.get("role_type") or a.get("description") or "?"
                if len(role) > 80:
                    role = role[:80] + "..."
                print(f"  - {name}: {role}")
            guide = s.get("pipeline_implementation_guide", {})
            if guide:
                print(f"\nImplementation Guide:")
                print(f"  {guide.get('overview', '')[:200]}")
        print(f"\nOutputs in: {self.working_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# Skill Summarizer Pipeline
# ═══════════════════════════════════════════════════════════════════════════

_DETECTOR_CLASSES: dict[str, type] = {
    "Python": PythonDetectorRole,
    "JavaScript": JSDetectorRole,
    "Infrastructure": InfraDetectorRole,
    "Configuration & Documentation": ConfigDocsDetectorRole,
}

_DETECTOR_OUTPUT_FILES: dict[str, str] = {
    "Python": "python_report.json",
    "JavaScript": "javascript_report.json",
    "Infrastructure": "infrastructure_report.json",
    "Configuration & Documentation": "configdocs_report.json",
}


def _run_detector(item: dict[str, Any], working_dir: str, backend: str) -> dict[str, Any]:
    """Execute a single domain detector with LLM + fallback to deterministic scan."""
    domain = item.get("domain", "")
    detector_cls = _DETECTOR_CLASSES.get(domain)
    output_file = _DETECTOR_OUTPUT_FILES.get(domain, "")

    if detector_cls is None:
        return {"output_file": "", "domain": domain, "data": {},
                "summary": f"Unknown domain: {domain}", "files": []}

    try:
        role = detector_cls()
        try:
            report = role.execute(working_dir=working_dir, backend=backend, project_dir=".")
        except Exception:
            report = {}

        if not report or not report.get("domain"):
            report = role._fallback_detect(project_dir=".", working_dir=working_dir)

        report_path = os.path.join(working_dir, output_file)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        skill_count = len(report.get("skills", []))
        return {
            "output_file": output_file, "domain": domain, "data": report,
            "summary": f"Detected {skill_count} skills in the {domain} domain" if skill_count
            else f"No skills detected in the {domain} domain",
            "files": [output_file],
        }
    except Exception as exc:
        return {"output_file": "", "domain": domain, "data": {},
                "summary": f"Detector exception: {exc}", "files": []}


class SkillState(TypedDict):
    input_spec: str
    working_dir: str
    backend: str
    project_scan: dict[str, Any]
    detector_outputs: list[dict[str, Any]]
    skill_inventory: dict[str, Any]
    status: str


class SkillPipeline(BasePipeline):
    """Skill Summarizer Pipeline with 7-agent fan-out/fan-in topology.

    scanner → 4 parallel detectors → aggregator → writer → END
    """

    name = "skill"
    default_output_dir = "skill_output"

    def _decompose(self, input_spec: str) -> list[dict[str, Any]]:
        return []

    def _display_decomposition(self, sub_tasks: list[dict]):
        print("Skill Summarizer Pipeline — analyzing project to detect skills.")
        print(f"Backend: {self.backend}")

    def _build_initial_state(self, input_spec: str, sub_tasks: list[dict]) -> dict:
        return {
            "input_spec": input_spec, "working_dir": self.working_dir,
            "backend": self.backend, "project_scan": {},
            "detector_outputs": [], "skill_inventory": {},
            "status": "initialized",
        }

    def _build_graph(self):
        working_dir = self.working_dir
        backend = self.backend

        def _scanner_node(state: SkillState) -> dict:
            print("\n[scanner] Analyzing project directory structure...")
            project_dir = state.get("input_spec", "")
            existing = state.get("project_scan", {})
            if existing and existing.get("file_categories"):
                n = existing.get("total_files", "?")
                print(f"  [scanner] Using existing scan ({n} files)")
                return {"status": "scanned"}

            scan: dict[str, Any] = {}
            try:
                role = SkillScannerRole()
                scan = role.execute(working_dir=working_dir, backend=state.get("backend", backend),
                                    project_dir=project_dir)
            except Exception as exc:
                print(f"  [scanner] Agent error: {exc}")

            if not scan or not scan.get("file_categories"):
                print("  [scanner] Falling back to deterministic scanner...")
                scan = SkillScannerRole._fallback_scanner(project_dir=project_dir, working_dir=working_dir)
                scan_path = os.path.join(working_dir, "project_scan.json")
                try:
                    with open(scan_path, "w") as f:
                        json.dump(scan, f, indent=2, default=str)
                except OSError:
                    pass

            total = scan.get("total_files", 0)
            print(f"  [scanner] Found {total} files")
            return {"project_scan": scan, "status": "scanned"}

        def _detectors_node(state: SkillState) -> dict:
            print("\n[detectors] Running 4 domain detectors in parallel...")
            items: list[dict[str, Any]] = [
                {"domain": "Python"}, {"domain": "JavaScript"},
                {"domain": "Infrastructure"}, {"domain": "Configuration & Documentation"},
            ]
            outputs, succeeded, failed = BasePipeline._run_parallel_agents(
                items, _run_detector, working_dir, state.get("backend", backend),
                max_workers=4,
            )
            for out in outputs:
                of = out.get("output_file", "")
                if of:
                    fp = os.path.join(working_dir, of)
                    if not os.path.exists(fp) or os.path.getsize(fp) == 0:
                        out["output_file"] = ""
                        if of in out.get("files", []):
                            out["files"].remove(of)

            ok = sum(1 for o in outputs if o.get("output_file"))
            print(f"  [detectors] {ok}/{len(outputs)} succeeded")
            for out in outputs:
                mark = "✓" if out.get("output_file") else "✗"
                print(f"    {mark} {out['domain']}: {out.get('summary', 'no output')[:120]}")
            return {"detector_outputs": outputs, "status": "detected" if ok > 0 else "error_no_detector_outputs"}

        def _aggregator_node(state: SkillState) -> dict:
            print("\n[aggregator] Aggregating domain reports...")
            detector_outputs = state.get("detector_outputs", [])
            project_dir = state.get("input_spec", "")
            inventory: dict[str, Any] = {}
            if any(d.get("output_file") for d in detector_outputs):
                try:
                    role = SkillAggregatorRole()
                    inventory = role.execute(working_dir=working_dir, backend=state.get("backend", backend))
                except Exception as exc:
                    print(f"  [aggregator] Agent error: {exc}")

            if not inventory or not inventory.get("skills"):
                print("  [aggregator] Falling back to rule-based aggregation...")
                inventory = SkillAggregatorRole._fallback_aggregator(project_dir=project_dir, working_dir=working_dir)
                inv_path = os.path.join(working_dir, "skill_inventory.json")
                try:
                    with open(inv_path, "w") as f:
                        json.dump(inventory, f, indent=2, default=str)
                except OSError:
                    pass

            total = inventory.get("summary", {}).get("total_skills", 0)
            print(f"  [aggregator] Aggregated {total} skills")
            return {"skill_inventory": inventory, "status": "aggregated"}

        def _writer_node(state: SkillState) -> dict:
            print("\n[writer] Generating final reports...")
            inventory = state.get("skill_inventory", {})
            project_dir = state.get("input_spec", "")
            proj_name = os.path.basename(project_dir.rstrip("/")) or "Project"

            if not inventory.get("skills"):
                print("  [writer] No skill inventory — nothing to write")
                return {"status": "error_no_inventory"}

            agent_ok = False
            try:
                role = SkillReportWriterRole()
                role.execute(working_dir=working_dir, backend=state.get("backend", backend),
                             project_name=proj_name)
                if os.path.exists(os.path.join(working_dir, "skills.json")):
                    agent_ok = True
            except Exception as exc:
                print(f"  [writer] Agent error: {exc}")

            if not agent_ok:
                print("  [writer] Falling back to template generation...")
                skills_data = SkillReportWriterRole._fallback_skills_json(proj_name, inventory)
                SkillReportWriterRole._write_skills_json(working_dir, skills_data)
                SkillReportWriterRole._fallback_report_md(proj_name, inventory, working_dir)

            for fname in ("skills.json", "skills_report.md"):
                fpath = os.path.join(working_dir, fname)
                if os.path.exists(fpath):
                    print(f"  [writer] ✓ {fname} ({os.path.getsize(fpath)} bytes)")
                else:
                    print(f"  [writer] ✗ {fname} not found")
            return {"status": "written"}

        workflow = StateGraph(SkillState)
        workflow.add_node("scanner", _scanner_node)
        workflow.add_node("detectors", _detectors_node)
        workflow.add_node("aggregator", _aggregator_node)
        workflow.add_node("writer", _writer_node)
        workflow.set_entry_point("scanner")

        flow = {
            "initialized": "scanner", "scanned": "detectors",
            "detected": "aggregator", "detected_partial": "aggregator",
            "aggregated": "writer", "written": END,
        }
        terminal = {"error_no_detector_outputs", "error_no_inventory"}
        router = BasePipeline._status_router(flow, terminal)
        for node in ("scanner", "detectors", "aggregator", "writer"):
            workflow.add_conditional_edges(node, router, {
                "scanner": "scanner", "detectors": "detectors",
                "aggregator": "aggregator", "writer": "writer", END: END,
            })
        return workflow.compile()

    def _print_results(self, final_state: dict):
        print("-" * 60)
        status = final_state.get("status", "unknown")
        print(f"Skill Summarizer Pipeline — {status.upper()}")
        print("-" * 60)
        scan = final_state.get("project_scan", {})
        if scan:
            print(f"\nProject Scan:")
            print(f"   Total files: {scan.get('total_files', '?')}")
            print(f"   Directories: {scan.get('total_dirs', '?')}")
            cats = scan.get("file_categories", {})
            if cats:
                for cat_name in ("source", "test", "config", "docs"):
                    count = len(cats.get(cat_name, []))
                    if count:
                        print(f"   {cat_name}: {count} files")
        detectors = final_state.get("detector_outputs", [])
        if detectors:
            print(f"\nDomain Detectors:")
            for d in detectors:
                mark = "✓" if d.get("output_file") else "✗"
                data = d.get("data", {})
                sc = len(data.get("skills", []))
                print(f"   {mark} {d['domain']}: {sc} skills")
        inventory = final_state.get("skill_inventory", {})
        if inventory:
            s = inventory.get("summary", {})
            print(f"\nAggregated Inventory:")
            print(f"   Total skills: {s.get('total_skills', 0)}")
            domains = s.get("domains_covered", [])
            if domains:
                print(f"   Domains: {', '.join(domains)}")
        print(f"\nOutputs in: {self.working_dir}")
