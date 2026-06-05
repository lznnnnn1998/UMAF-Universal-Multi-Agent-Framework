"""Feature Pipeline — 5-node graph for adding/editing code in existing projects.

Nodes: scanner → planner → coder ↔ reviewer → writer → END
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from feature.scanner import FeatureScannerRole
from feature.planner import FeaturePlannerRole
from feature.coder import FeatureCoderRole
from feature.reviewer import FeatureReviewerRole
from feature.writer import FeatureReportWriterRole
from .base import BasePipeline


class FeatureState(TypedDict, total=False):
    """State for the Feature Pipeline v2 — 12 fields."""
    input_spec: str
    working_dir: str
    backend: str
    status: str
    iteration: int
    project_context: dict[str, Any]
    implementation_plan: dict[str, Any]
    changed_files: list[str]
    review_passed: bool
    review_issues: list[str]
    feature_report: str


class FeaturePipeline(BasePipeline):
    """5-node pipeline: scanner → planner → coder ↔ reviewer → writer."""

    name = "feature"
    default_output_dir = "feature_output"

    def _decompose(self, input_spec: str) -> list[dict[str, Any]]:
        return [{"id": 1, "title": "Feature", "description": input_spec}]

    def _display_decomposition(self, sub_tasks: list[dict]):
        print(f"\nFeature: {sub_tasks[0]['description'][:200]}")

    def _build_initial_state(self, input_spec: str,
                             sub_tasks: list[dict]) -> FeatureState:
        return {
            "input_spec": input_spec,
            "working_dir": self.working_dir,
            "backend": self.backend,
            "status": "initialized",
            "iteration": 0,
            "project_context": {},
            "implementation_plan": {},
            "changed_files": [],
            "review_passed": False,
            "review_issues": [],
            "feature_report": "",
        }

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(FeatureState)
        backend = self.backend
        working_dir = self.working_dir

        scanner_role = FeatureScannerRole()
        planner_role = FeaturePlannerRole()
        coder_role = FeatureCoderRole()
        reviewer_role = FeatureReviewerRole()
        writer_role = FeatureReportWriterRole()

        # ── Node: scanner ──────────────────────────────────────────────────
        def _scanner_node(state: FeatureState) -> dict:
            print("\n[scanner] Analyzing project...")
            result = scanner_role.execute(
                working_dir=state.get("working_dir", working_dir),
                backend=state.get("backend", backend),
                project_dir=".",
            )
            ctx = result if isinstance(result, dict) else {}
            if not ctx:
                ctx = scanner_role._fallback_scanner(".", state.get("working_dir", working_dir))
            print(f"  [scanner] Found {ctx.get('total_files', '?')} files, "
                  f"language={ctx.get('language', '?')}")
            return {"project_context": ctx, "status": "scanned"}

        # ── Node: planner ──────────────────────────────────────────────────
        def _planner_node(state: FeatureState) -> dict:
            print("\n[planner] Planning implementation...")
            result = planner_role.execute(
                working_dir=state.get("working_dir", working_dir),
                backend=state.get("backend", backend),
                feature_description=state.get("input_spec", ""),
            )
            plan = result if isinstance(result, dict) else {}
            creates = len(plan.get("files_to_create", []))
            modifies = len(plan.get("files_to_modify", []))
            print(f"  [planner] Plan: {creates} to create, {modifies} to modify")
            return {"implementation_plan": plan, "status": "planned"}

        # ── Node: coder ────────────────────────────────────────────────────
        def _coder_node(state: FeatureState) -> dict:
            iteration = state.get("iteration", 0) + 1
            review_issues = state.get("review_issues", [])
            wd = state.get("working_dir", working_dir)

            if review_issues:
                print(f"\n[coder] Iteration {iteration} — fixing "
                      f"{len(review_issues)} review issues...")
            else:
                print(f"\n[coder] Iteration {iteration} — implementing...")

            result = coder_role.execute(
                working_dir=wd,
                backend=state.get("backend", backend),
                review_issues=review_issues if review_issues else None,
            )
            parsed = result if isinstance(result, dict) else {}
            changed = parsed.get("changed_files", [])
            print(f"  [coder] Changed {len(changed)} files")
            for f in changed:
                print(f"    - {f}")
            return {
                "changed_files": changed,
                "status": "coded",
                "iteration": iteration,
            }

        # ── Node: reviewer ─────────────────────────────────────────────────
        def _reviewer_node(state: FeatureState) -> dict:
            wd = state.get("working_dir", working_dir)
            changed = state.get("changed_files", [])
            print(f"\n[reviewer] Reviewing {len(changed)} changed files...")

            result = reviewer_role.execute(
                working_dir=wd,
                backend=state.get("backend", backend),
                changed_files=changed,
                feature_description=state.get("input_spec", ""),
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

            return {"review_passed": passed, "review_issues": issues}

        # ── Node: writer ───────────────────────────────────────────────────
        def _writer_node(state: FeatureState) -> dict:
            wd = state.get("working_dir", working_dir)
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
            "planned": "coder",
            "written": END,
        }
        terminal = {"error_scanner_failed", "error_planner_failed"}
        status_router = BasePipeline._status_router(linear_flow, terminal)

        def _coder_router(state: FeatureState) -> Literal["reviewer", "__end__"]:
            if state.get("status") == "coded":
                return "reviewer"
            return END

        def _reviewer_router(state: FeatureState) -> Literal["coder", "writer", "__end__"]:
            if state.get("review_passed"):
                return "writer"
            if state.get("iteration", 0) >= 5:
                print("  [pipeline] Max iterations reached, proceeding to writer")
                return "writer"
            return "coder"

        # ── Wire graph ─────────────────────────────────────────────────────
        workflow.add_node("scanner", _scanner_node)
        workflow.add_node("planner", _planner_node)
        workflow.add_node("coder", _coder_node)
        workflow.add_node("reviewer", _reviewer_node)
        workflow.add_node("writer", _writer_node)

        workflow.set_entry_point("scanner")
        workflow.add_conditional_edges("scanner", status_router, {
            "planner": "planner", END: END,
        })
        workflow.add_conditional_edges("planner", status_router, {
            "coder": "coder", END: END,
        })
        workflow.add_conditional_edges("coder", _coder_router, {
            "reviewer": "reviewer", END: END,
        })
        workflow.add_conditional_edges("reviewer", _reviewer_router, {
            "coder": "coder", "writer": "writer", END: END,
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
            print("Feature Pipeline: COMPLETED (review not passed or max cycles)")
        print(f"Iterations: {final_state.get('iteration', 0)}")
        changed = final_state.get("changed_files", [])
        print(f"Files changed: {len(changed)}")
        for f in changed:
            print(f"  - {f}")
        report = final_state.get("feature_report", "")
        if report:
            print(f"Report: {report}")
        print(f"Working directory: {self.working_dir}")
        print("=" * 60)
