"""CoderPP Pipeline — head → workers → reviewer → organizer for multi-file code generation."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agent import CheckpointManager
from .base import BasePipeline
from coderpp.head_agent import decompose_to_modules, _fallback_decompose as _coderpp_fallback, observe_workers
from coderpp.worker_agent import code_submodule
from coderpp.reviewer_agent import review_module
from coderpp.organizer import assemble_project


CPP_HEAD_TIMEOUT = 500
CPP_WORKER_TIMEOUT = 1200
CPP_REVIEWER_TIMEOUT = 1200
CPP_MAX_VERSIONS = 5
CPP_MAX_WORKER_RETRIES = 5


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

        # Validate dependency graph on resume — warn about cycles early
        dep_issues = BasePipeline._validate_dependencies(sub_tasks)
        if dep_issues:
            print(f"\n[resume] WARNING: Dependency issues in decomposition.json ({len(dep_issues)}):")
            for issue in dep_issues:
                print(f"  - {issue}")
            print("  The pipeline will attempt to break cycles at runtime, but results may be suboptimal.")
            print("  Consider fixing decomposition.json and re-running.")

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
            ok = worker_success.get(mid_str, False)
            files: list[str] = []
            mod_dir = os.path.join(modules_dir, name)
            if ok or (not has_checkpoint and os.path.isdir(mod_dir)):
                if os.path.isdir(mod_dir):
                    for f in os.listdir(mod_dir):
                        if f.endswith(".py") and "__pycache__" not in f:
                            files.append(f"modules/{name}/{f}")
            # If no checkpoint and only skeleton files, treat as NOT succeeded
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
