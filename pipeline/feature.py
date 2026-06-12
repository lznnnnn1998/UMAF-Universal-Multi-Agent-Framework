"""Feature Pipeline — 5-node graph for adding/editing code in existing projects.

Nodes: scanner → planner → coders ↔ reviewer → writer → END

v2: Multi-coder parallelism with topological-level execution, dependency injection,
and cross-coder integration review.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from feature.scanner import FeatureScannerRole
from feature.planner import FeaturePlannerRole
from feature.coder import FeatureCoderRole, _feature_coder_worker
from feature.reviewer import FeatureReviewerRole
from feature.writer import FeatureReportWriterRole
from .base import BasePipeline


class FeatureState(TypedDict, total=False):
    """State for the Feature Pipeline v2 — 17 fields (3 new for multi-coder)."""
    input_spec: str
    working_dir: str
    backend: str
    project_dir: str
    status: str
    iteration: int
    version: int
    project_context: dict[str, Any]
    implementation_plan: dict[str, Any]
    changed_files: list[str]
    review_passed: bool
    review_issues: list[str]
    feature_report: str
    # ── New for multi-coder parallelism (v2) ──
    sub_tasks: list[dict[str, Any]]
    coder_outputs: list[dict[str, Any]]
    dependency_graph: dict[str, Any]


class FeaturePipeline(BasePipeline):
    """5-node pipeline: scanner → planner → coders ↔ reviewer → writer.

    Coders run in topological levels with dependency injection between levels.
    Within each level, coders execute in parallel.
    """

    name = "feature"
    default_output_dir = "feature_output"
    _MAX_CODER_RETRIES = 3

    def _decompose(self, input_spec: str) -> list[dict[str, Any]]:
        return [{"id": 1, "title": "Feature", "description": input_spec}]

    def _display_decomposition(self, sub_tasks: list[dict]):
        print(f"\nFeature: {sub_tasks[0]['description'][:200]}")

    def _build_initial_state(self, input_spec: str,
                             sub_tasks: list[dict]) -> FeatureState:
        target = getattr(self, "target_dir", None)
        project_dir = target or "."
        return {
            "input_spec": input_spec,
            "working_dir": self.working_dir,
            "backend": self.backend,
            "project_dir": project_dir,
            "status": "initialized",
            "iteration": 0,
            "version": 1,
            "project_context": {},
            "implementation_plan": {},
            "changed_files": [],
            "review_passed": False,
            "review_issues": [],
            "feature_report": "",
            "sub_tasks": [],
            "coder_outputs": [],
            "dependency_graph": {},
        }

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(FeatureState)
        backend = self.backend
        working_dir = self.working_dir

        scanner_role = FeatureScannerRole()
        planner_role = FeaturePlannerRole()
        reviewer_role = FeatureReviewerRole()
        writer_role = FeatureReportWriterRole()

        # ── Node: scanner ──────────────────────────────────────────────────
        def _scanner_node(state: FeatureState) -> dict:
            project_dir = state.get("project_dir", ".")
            print(f"\n[scanner] Analyzing project ({project_dir})...")
            wd = state.get("working_dir", working_dir)
            result = scanner_role.execute(
                working_dir=wd,
                backend=state.get("backend", backend),
                project_dir=project_dir,
            )
            ctx = result if isinstance(result, dict) else {}
            if not ctx:
                ctx = scanner_role._fallback_scanner(project_dir, wd)
            # Write project_context.json to disk so downstream agents can read it.
            # parse_result only writes in the fallback path — ensure the file
            # exists even when the plan was extracted from agent messages.
            ctx_path = os.path.join(wd, "project_context.json")
            if not os.path.exists(ctx_path):
                try:
                    with open(ctx_path, "w") as f:
                        json.dump(ctx, f, indent=2, default=str)
                except OSError:
                    pass
            print(f"  [scanner] Found {ctx.get('total_files', '?')} files, "
                  f"language={ctx.get('language', '?')}")
            return {"project_context": ctx, "status": "scanned"}

        # ── Node: planner ──────────────────────────────────────────────────
        def _planner_node(state: FeatureState) -> dict:
            wd = state.get("working_dir", working_dir)
            print("\n[planner] Planning implementation...")
            result = planner_role.execute(
                working_dir=wd,
                backend=state.get("backend", backend),
                feature_description=state.get("input_spec", ""),
            )
            plan = result if isinstance(result, dict) else {}
            creates = len(plan.get("files_to_create", []))
            modifies = len(plan.get("files_to_modify", []))
            sub_tasks = plan.get("sub_tasks", [])
            dep_graph = _build_dependency_graph(sub_tasks)

            # Write plan files to disk so the coder can read them.
            # parse_result only writes in the fallback path — ensure the
            # files exist even when the plan was extracted from agent messages.
            plan_path = os.path.join(wd, "implementation_plan.json")
            if not os.path.exists(plan_path):
                try:
                    with open(plan_path, "w") as f:
                        json.dump(plan, f, indent=2, default=str)
                except OSError:
                    pass
            decomp_path = os.path.join(wd, "decomposition.json")
            if not os.path.exists(decomp_path) and sub_tasks:
                try:
                    with open(decomp_path, "w") as f:
                        json.dump(sub_tasks, f, indent=2, default=str)
                except OSError:
                    pass

            print(f"  [planner] Plan: {creates} to create, {modifies} to modify, "
                  f"{len(sub_tasks)} sub-task(s)")
            for st in sub_tasks:
                deps = st.get("dependencies", [])
                mod_name = st.get("module_name") or f"id={st.get('id', '?')}"
                dep_suffix = f" (depends on: {deps})" if deps else ""
                print(f"    - {mod_name}{dep_suffix}")
            return {
                "implementation_plan": plan,
                "sub_tasks": sub_tasks,
                "dependency_graph": dep_graph,
                "status": "planned_with_deps",
            }

        # ── Node: coders (multi-coder with topological levels) ──────────────
        def _coders_node(state: FeatureState) -> dict:
            sub_tasks: list[dict[str, Any]] = state.get("sub_tasks", [])
            version = state.get("version", 1)
            project_dir = state.get("project_dir", ".")
            wd = state.get("working_dir", working_dir)
            review_issues = state.get("review_issues", [])

            if not sub_tasks:
                # Fallback: single coder mode (original behavior)
                print("\n[coders] No sub_tasks found — running single coder (fallback)...")
                coder_role = FeatureCoderRole()
                result = coder_role.execute(
                    working_dir=wd,
                    backend=state.get("backend", backend),
                    version=version,
                    project_dir=project_dir,
                    review_issues=review_issues if review_issues else None,
                )
                parsed = result if isinstance(result, dict) else {}
                changed = parsed.get("changed_files", [])
                coder_outputs = [{
                    "sub_task_id": 1,
                    "module_name": "feature_implementation",
                    "files": changed,
                    "summary": parsed.get("summary", f"Changed {len(changed)} files"),
                    "dependency_verification": parsed.get("dependency_verification"),
                }]
                all_changed = changed
            else:
                # ── Topological level execution ────────────────────────────
                levels = BasePipeline._topological_levels(sub_tasks)
                print(f"\n[coders] Running {len(sub_tasks)} sub-task(s) in "
                      f"{len(levels)} topological level(s) (v{version})")

                if review_issues:
                    print(f"  [coders] Fixing {len(review_issues)} review issues...")

                all_outputs: list[dict[str, Any]] = []
                completed: dict[int | str, dict[str, Any]] = {}
                failed_count = 0

                for level_idx, level_tasks in enumerate(levels):
                    names = [t.get("module_name", f"id={t.get('id', '?')}")
                             for t in level_tasks]
                    print(f"\n  [level {level_idx + 1}/{len(levels)}] Running: {names}")

                    # Inject dependency outputs from prior levels
                    for t in level_tasks:
                        deps = t.get("dependencies", [])
                        if deps:
                            dep_files: list[dict[str, Any]] = []
                            for d in deps:
                                dep_key: int | str | None = None
                                if isinstance(d, int):
                                    dep_key = d
                                elif isinstance(d, str):
                                    dep_key = d
                                elif isinstance(d, dict):
                                    dep_key = d.get("id") or d.get("module_name")
                                if dep_key is not None and dep_key in completed:
                                    cinfo = completed[dep_key]
                                    dep_files.append({
                                        "dep_id": dep_key,
                                        "module_name": cinfo.get("module_name", ""),
                                        "title": cinfo.get("module_name", ""),
                                        "files": cinfo.get("files", []),
                                        "summary": cinfo.get("summary", ""),
                                        "output_file": cinfo.get("output_file", ""),
                                    })
                            if dep_files:
                                t["_dependency_outputs"] = dep_files

                    # Run coders in parallel within this level
                    results, succeeded, failed = BasePipeline._run_parallel_agents(
                        level_tasks,
                        _feature_coder_worker,
                        wd,
                        state.get("backend", backend),
                        timeout=600,
                        max_workers=len(level_tasks),
                    )

                    all_outputs.extend(results)
                    failed_count += failed

                    # Register completed outputs for next level
                    for out in results:
                        sid = out.get("sub_task_id")
                        if sid is not None:
                            completed[sid] = out
                        mname = out.get("module_name")
                        if mname is not None:
                            completed[mname] = out

                    # Print per-level results
                    for r in results:
                        fcount = len(r.get("files", []))
                        dep_verif = r.get("dependency_verification")
                        status_str = "✓" if fcount > 0 else "✗"
                        dep_str = ""
                        if dep_verif is True:
                            dep_str = " [deps verified]"
                        elif dep_verif is False:
                            dep_str = " [deps ISSUE]"
                        print(f"    {status_str} {r.get('module_name', '?')}: "
                              f"{fcount} file(s){dep_str}")

                    # Stop on dependency failure: dependent levels can't run
                    if failed > 0 and level_idx + 1 < len(levels):
                        remaining = sum(len(l) for l in levels[level_idx + 1:])
                        print(f"\n  [coders] Stopping early: {failed} task(s) failed in "
                              f"level {level_idx} — {remaining} downstream task(s) "
                              f"deferred for retry.")
                        break

                coder_outputs = all_outputs
                all_changed: list[str] = []
                for out in all_outputs:
                    all_changed.extend(out.get("files", []))

            # Print summary
            total_files = len(all_changed)
            succeeded = sum(1 for o in coder_outputs if o.get("files"))
            print(f"\n  [coders] Done: {succeeded}/{len(coder_outputs)} coders succeeded, "
                  f"{total_files} total file(s) changed")

            return {
                "changed_files": all_changed,
                "coder_outputs": coder_outputs,
                "status": "coders_done",
                "iteration": state.get("iteration", 0) + 1,
            }

        # ── Node: reviewer ─────────────────────────────────────────────────
        def _reviewer_node(state: FeatureState) -> dict:
            wd = state.get("working_dir", working_dir)
            project_dir = state.get("project_dir", ".")
            version = state.get("version", 1)
            changed = state.get("changed_files", [])
            coder_outputs = state.get("coder_outputs", [])

            if coder_outputs:
                print(f"\n[reviewer] Reviewing {len(changed)} changed files from "
                      f"{len(coder_outputs)} coder(s) (v{version})...")
            else:
                print(f"\n[reviewer] Reviewing {len(changed)} changed files (v{version})...")

            # Pass all_coder_outputs so the reviewer can perform cross-coder
            # integration verification when multiple coders were used.
            result = reviewer_role.execute(
                working_dir=wd,
                backend=state.get("backend", backend),
                version=version,
                project_dir=project_dir,
                changed_files=changed,
                feature_description=state.get("input_spec", ""),
                all_coder_outputs=coder_outputs if coder_outputs else None,
            )
            parsed = result if isinstance(result, dict) else {}
            passed = parsed.get("review_passed", False)
            issues = parsed.get("review_issues", [])

            if passed:
                print("  [reviewer] REVIEW_PASSED")
            else:
                print(f"  [reviewer] REVIEW_FAILED — {len(issues)} issues")
                for i in issues[:5]:
                    print(f"    - {i}")

            # Bump version on failure so the next coder iteration loads prior
            # checkpoint context via BaseAgent.load_previous().
            result_dict: dict[str, Any] = {"review_passed": passed, "review_issues": issues}
            if not passed:
                result_dict["version"] = version + 1
            return result_dict

        # ── Node: writer ───────────────────────────────────────────────────
        def _writer_node(state: FeatureState) -> dict:
            wd = state.get("working_dir", working_dir)
            coder_outputs = state.get("coder_outputs", [])
            print(f"\n[writer] Writing feature report...")
            result = writer_role.execute(
                working_dir=wd,
                backend=state.get("backend", backend),
                changed_files=state.get("changed_files", []),
                review_passed=state.get("review_passed", False),
                review_issues=state.get("review_issues", []),
                feature_description=state.get("input_spec", ""),
            )
            parsed = result if isinstance(result, dict) else {}
            report = parsed.get("feature_report", "")
            print(f"  [writer] Report: {report}")
            return {"feature_report": report, "status": "written"}

        # ── Routers ────────────────────────────────────────────────────────
        linear_flow = {
            "initialized": "scanner",
            "scanned": "planner",
            "planned": "coders",
            "planned_with_deps": "coders",
            "written": END,
        }
        terminal = {"error_scanner_failed", "error_planner_failed"}
        status_router = BasePipeline._status_router(linear_flow, terminal)

        def _coders_router(state: FeatureState) -> Literal["reviewer", "__end__"]:
            if state.get("status") == "coders_done":
                return "reviewer"
            return END

        def _reviewer_router(state: FeatureState) -> Literal["coders", "writer", "__end__"]:
            if state.get("review_passed"):
                return "writer"
            version = state.get("version", 1)
            if version > FeaturePipeline._MAX_CODER_RETRIES:
                print(f"  [pipeline] Max coder retries "
                      f"({FeaturePipeline._MAX_CODER_RETRIES}) reached, "
                      f"proceeding to writer")
                return "writer"
            return "coders"

        # ── Wire graph ─────────────────────────────────────────────────────
        workflow.add_node("scanner", _scanner_node)
        workflow.add_node("planner", _planner_node)
        workflow.add_node("coders", _coders_node)
        workflow.add_node("reviewer", _reviewer_node)
        workflow.add_node("writer", _writer_node)

        workflow.set_entry_point("scanner")
        workflow.add_conditional_edges("scanner", status_router, {
            "planner": "planner", END: END,
        })
        workflow.add_conditional_edges("planner", status_router, {
            "coders": "coders", END: END,
        })
        workflow.add_conditional_edges("coders", _coders_router, {
            "reviewer": "reviewer", END: END,
        })
        workflow.add_conditional_edges("reviewer", _reviewer_router, {
            "coders": "coders", "writer": "writer", END: END,
        })
        workflow.add_conditional_edges("writer", status_router, {
            END: END,
        })

        return workflow.compile()

    def _print_results(self, final_state: dict):
        print("\n" + "=" * 60)
        if final_state.get("review_passed"):
            print("Feature Pipeline: PASSED")
        else:
            print("Feature Pipeline: COMPLETED (review not passed or max retries)")
        print(f"Iterations: {final_state.get('iteration', 0)}")
        print(f"Version: {final_state.get('version', 1)}")

        # Show coder output summary if multi-coder was used
        coder_outputs = final_state.get("coder_outputs", [])
        if coder_outputs:
            print(f"\nCoder Outputs ({len(coder_outputs)} coder(s)):")
            # Group by estimated levels (order of appearance)
            for out in coder_outputs:
                mod = out.get("module_name", "?")
                fcount = len(out.get("files", []))
                dep_verif = out.get("dependency_verification")
                verif_str = ""
                if dep_verif is True:
                    verif_str = " (deps verified)"
                elif dep_verif is False:
                    verif_str = " (deps ISSUE)"
                elif dep_verif is None:
                    verif_str = " (no deps)"
                print(f"  [{mod}] {fcount} file(s){verif_str}")
                for f in out.get("files", [])[:10]:
                    print(f"    - {f}")
                if len(out.get("files", [])) > 10:
                    print(f"    ... and {len(out['files']) - 10} more")

        changed = final_state.get("changed_files", [])
        print(f"\nTotal files changed: {len(changed)}")
        for f in changed[:20]:
            print(f"  - {f}")
        if len(changed) > 20:
            print(f"  ... and {len(changed) - 20} more")

        report = final_state.get("feature_report", "")
        if report:
            print(f"Report: {report}")
        print(f"Working directory: {self.working_dir}")
        print("=" * 60)


def _build_dependency_graph(sub_tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a dependency graph from sub_tasks for display and analysis.

    Returns a dict with nodes (module_names) and edges (dependency relationships),
    plus metadata about levels and parallelism opportunities.
    """
    if not sub_tasks:
        return {"nodes": [], "edges": [], "levels": 0, "max_parallelism": 0}

    nodes = []
    for t in sub_tasks:
        nodes.append({
            "id": t.get("id"),
            "module_name": t.get("module_name", f"id={t.get('id', '?')}"),
            "file_count": (
                len(t.get("files_to_create", []))
                + len(t.get("files_to_modify", []))
            ),
        })

    edges = []
    for t in sub_tasks:
        src = t.get("module_name", f"id={t.get('id', '?')}")
        for dep in t.get("dependencies", []):
            dep_name = dep if isinstance(dep, str) else str(dep)
            # Resolve int deps to module names
            if isinstance(dep, int):
                for st in sub_tasks:
                    if st.get("id") == dep:
                        dep_name = st.get("module_name", f"id={dep}")
                        break
            edges.append({"from": dep_name, "to": src})

    levels = BasePipeline._topological_levels(sub_tasks)
    max_parallelism = max((len(l) for l in levels), default=0)

    return {
        "nodes": nodes,
        "edges": edges,
        "levels": len(levels),
        "max_parallelism": max_parallelism,
    }
