"""BasePipeline — abstract pipeline with output dir management and shared helpers."""

from __future__ import annotations

import concurrent.futures
import os
import shutil
import sys
from typing import Any, Literal

from langgraph.graph import END, StateGraph


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
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent
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

        # Validate dependency graph before proceeding
        dep_issues = self._validate_dependencies(sub_tasks)
        if dep_issues:
            print(f"\n[WARNING] Dependency issues found ({len(dep_issues)}):")
            for issue in dep_issues:
                print(f"  - {issue}")

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

        When a dependency cycle is detected, the weakest edge is removed to break
        the cycle rather than silently running everything in one flat level.
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
                # ── Cycle detected ──────────────────────────────────────
                cycle_tasks = [key_to_task[k] for k in remaining]
                cycle_names = [t.get("module_name", f"id={t.get('id', '?')}") for t in cycle_tasks]
                print(f"\n  [WARNING] Dependency cycle detected among {len(cycle_names)} tasks: "
                      f"{cycle_names[:5]}{'...' if len(cycle_names) > 5 else ''}")

                # Iteratively remove edges until at least one task becomes
                # eligible.  A single edge removal may not suffice when there
                # are multiple overlapping cycles (e.g. A→B→C→A plus
                # integration-task→A).
                broken = False
                max_attempts = len(cycle_tasks) * 2
                for _attempt in range(max_attempts):
                    # Recompute: has at least one task become eligible?
                    current_candidate = [key_to_task[k] for k in sorted(remaining)
                                         if _dep_keys(key_to_task[k]).isdisjoint(remaining)]
                    if current_candidate:
                        broken = True
                        break

                    # Still no progress — pick the task with the most
                    # intra-cycle deps and remove its last dependency
                    # (most likely to be incorrectly specified).
                    still_stuck = [key_to_task[k] for k in sorted(remaining)
                                   if not _dep_keys(key_to_task[k]).isdisjoint(remaining)]
                    best = max(still_stuck,
                               key=lambda t: len(_dep_keys(t) & remaining))
                    intra_deps = sorted(_dep_keys(best) & remaining)

                    # Remove the LAST intra-cycle dep (the one least likely
                    # to be load-bearing — earlier deps are usually more
                    # foundational).
                    dep_to_remove = intra_deps[-1]
                    new_deps = []
                    removed = []
                    for d in best.get("dependencies", []):
                        d_key = None
                        if isinstance(d, int):
                            match = id_to_task.get(d)
                            d_key = _task_key(match) if match else f"__id_{d}"
                        elif isinstance(d, str):
                            d_key = d
                        elif isinstance(d, dict):
                            d_key = d.get("module_name") or f"__id_{d.get('id', d)}"
                        if d_key == dep_to_remove:
                            removed.append(d)
                        else:
                            new_deps.append(d)
                    best["dependencies"] = new_deps
                    best_name = best.get("module_name", f"id={best.get('id', '?')}")
                    print(f"  [cycle break] Removed dep '{dep_to_remove}' from "
                          f"'{best_name}' (was: {removed})")

                if broken:
                    # Recompute current level with broken dependencies
                    current = [key_to_task[k] for k in sorted(remaining)
                               if _dep_keys(key_to_task[k]).isdisjoint(remaining)]
                    if not current:
                        print(f"  [WARNING] Cycle persists after {max_attempts} "
                              f"edge removals — running all {len(cycle_tasks)} "
                              f"tasks in parallel.")
                        current = cycle_tasks
                else:
                    print(f"  [WARNING] Could not break cycle — running all "
                          f"{len(cycle_tasks)} tasks in parallel.")
                    current = cycle_tasks

            levels.append(current)
            remaining -= {_task_key(t) for t in current}

        return levels

    @staticmethod
    def _validate_dependencies(sub_tasks: list[dict[str, Any]]) -> list[str]:
        """Validate the dependency graph of a decomposition.

        Returns a list of human-readable issue strings. An empty list means
        the dependency graph is valid (acyclic, no unresolved references).

        Call this before execution to catch problems early.
        """
        issues: list[str] = []

        if not sub_tasks:
            return issues

        # Build lookup tables
        id_to_task = {t["id"]: t for t in sub_tasks}
        name_to_task: dict[str, dict] = {}
        for t in sub_tasks:
            name = t.get("module_name", "")
            if name:
                if name in name_to_task:
                    issues.append(
                        f"Duplicate module_name '{name}' "
                        f"(ids: {name_to_task[name]['id']}, {t['id']})"
                    )
                name_to_task[name] = t

        def _resolve_dep(dep) -> str | None:
            """Resolve a dependency to its module_name, or None if unresolvable."""
            if isinstance(dep, int):
                match = id_to_task.get(dep)
                return match.get("module_name") if match else None
            elif isinstance(dep, str):
                if dep in name_to_task:
                    return dep
                # Also try matching by id (string that looks like a number)
                return None
            elif isinstance(dep, dict):
                name = dep.get("module_name", "")
                if name in name_to_task:
                    return name
                return None
            return None

        # Check 1: unresolved dependency references
        for t in sub_tasks:
            t_name = t.get("module_name", f"id={t['id']}")
            for dep in t.get("dependencies", []):
                resolved = _resolve_dep(dep)
                if resolved is None:
                    dep_repr = dep.get("module_name", str(dep)) if isinstance(dep, dict) else str(dep)
                    issues.append(
                        f"Unresolved dependency: '{t_name}' depends on "
                        f"'{dep_repr}' which does not match any task"
                    )

        # Check 2: dependency cycles via DFS
        def _has_cycle() -> list[str] | None:
            """Return the first cycle found as a list of module_names, or None."""
            WHITE, GRAY, BLACK = 0, 1, 2
            color: dict[str, int] = {t.get("module_name", f"__id_{t['id']}"): WHITE
                                      for t in sub_tasks}
            parent: dict[str, str | None] = {}

            def _dfs(node: str) -> list[str] | None:
                color[node] = GRAY
                task = name_to_task.get(node) or id_to_task.get(
                    int(node.replace("__id_", "")) if node.startswith("__id_") else 0
                )
                deps = task.get("dependencies", []) if task else []
                for dep in deps:
                    neighbor = _resolve_dep(dep)
                    if neighbor is None:
                        continue
                    if color.get(neighbor) == GRAY:
                        # Found cycle — reconstruct path
                        cycle = [neighbor, node]
                        cur = node
                        while parent.get(cur) and parent[cur] != neighbor:
                            cur = parent[cur]
                            cycle.append(cur)
                        cycle.append(neighbor)
                        cycle.reverse()
                        return cycle
                    if color.get(neighbor) == WHITE:
                        parent[neighbor] = node
                        result = _dfs(neighbor)
                        if result:
                            return result
                color[node] = BLACK
                return None

            for node in color:
                if color[node] == WHITE:
                    result = _dfs(node)
                    if result:
                        return result
            return None

        cycle = _has_cycle()
        if cycle:
            issues.append(
                f"Dependency cycle detected: {' → '.join(cycle)}"
            )

        return issues

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
        # Map sub_task_id → output info so dependent tasks can read upstream results
        completed: dict[int | str, dict[str, Any]] = {}

        for level_idx, level_tasks in enumerate(levels):
            names = [t.get("module_name") or t.get("title", "?") for t in level_tasks]
            print(f"\n  [dependency level {level_idx}/{len(levels)}] Running: {names}")

            # Inject dependency outputs into tasks that declare dependencies
            for t in level_tasks:
                deps = t.get("dependencies", [])
                if deps:
                    dep_files: list[dict[str, Any]] = []
                    for d in deps:
                        # Resolve dependency by id (int) or name (str)
                        dep_id: int | str | None = None
                        if isinstance(d, int):
                            dep_id = d
                        elif isinstance(d, str):
                            dep_id = d
                        elif isinstance(d, dict):
                            dep_id = d.get("id") or d.get("module_name")
                        if dep_id is not None and dep_id in completed:
                            cinfo = completed[dep_id]
                            dep_files.append({
                                "dep_id": dep_id,
                                "title": cinfo.get("title") or cinfo.get("module_name", ""),
                                "output_file": cinfo.get("output_file", ""),
                                "files": cinfo.get("files", []),
                            })
                    if dep_files:
                        t["_dependency_outputs"] = dep_files

            results, succeeded, failed = BasePipeline._run_parallel_agents(
                level_tasks, agent_func, working_dir, backend, timeout,
                max_workers=len(level_tasks),
                retry_failures=retry_failures, max_retries=max_retries,
            )
            all_outputs.extend(results)
            total_succeeded += succeeded
            total_failed += failed

            # Record completed outputs so the next level can use them.
            # Register both sub_task_id and module_name as keys so dependencies
            # can be resolved by either (int id from Research, str name from CoderPP).
            for out in results:
                sid = out.get("sub_task_id")
                if sid is not None:
                    completed[sid] = out
                mname = out.get("module_name")
                if mname is not None:
                    completed[mname] = out

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
