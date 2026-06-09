"""Self-Evolution Pipeline — UMAF analyzes and improves itself.

Nodes: analyzer → planner → coder ↔ reviewer → writer → END

The self-evolution pipeline allows UMAF to:
1. Analyze its own codebase and agent logs for improvement opportunities
2. Create an implementation plan based on analysis findings
3. Implement improvements to its own source code
4. Verify changes by running the test suite
5. Document the evolution in a report

Safety: operates in the current git branch. All changes can be reverted with
``git checkout -- .``. Run with ``--clean`` to start from a fresh working directory.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from self_evolution.analyzer import SelfEvolutionAnalyzerRole
from self_evolution.planner import SelfEvolutionPlannerRole
from self_evolution.coder import SelfEvolutionCoderRole
from self_evolution.reviewer import SelfEvolutionReviewerRole
from self_evolution.writer import SelfEvolutionWriterRole
from .base import BasePipeline


class SelfEvolutionState(TypedDict, total=False):
    input_spec: str
    working_dir: str
    backend: str
    project_dir: str
    status: str
    iteration: int
    analysis_report: dict[str, Any]
    implementation_plan: dict[str, Any]
    changed_files: list[str]
    review_passed: bool
    review_issues: list[str]
    test_results: str
    evolution_report: str


class SelfEvolutionPipeline(BasePipeline):
    """Self-evolution pipeline: analyze UMAF → plan improvements → implement → verify."""

    name = "self_evolution"
    default_output_dir = "self_evolution_output"

    MAX_ITERATIONS = 3

    def _decompose(self, input_spec: str) -> list[dict[str, Any]]:
        return [{"id": 1, "title": "Self-Evolution", "description": input_spec}]

    def _display_decomposition(self, sub_tasks: list[dict]):
        print(f"\nSelf-Evolution Goal: {sub_tasks[0]['description'][:200]}")

    def _build_initial_state(self, input_spec: str,
                             sub_tasks: list[dict]) -> SelfEvolutionState:
        target = getattr(self, "target_dir", None)
        project_dir = target or os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        return {
            "input_spec": input_spec,
            "working_dir": self.working_dir,
            "backend": self.backend,
            "project_dir": project_dir,
            "status": "initialized",
            "iteration": 0,
            "analysis_report": {},
            "implementation_plan": {},
            "changed_files": [],
            "review_passed": False,
            "review_issues": [],
            "test_results": "",
            "evolution_report": "",
        }

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(SelfEvolutionState)
        backend = self.backend
        working_dir = self.working_dir

        analyzer = SelfEvolutionAnalyzerRole()
        planner = SelfEvolutionPlannerRole()
        coder = SelfEvolutionCoderRole()
        reviewer = SelfEvolutionReviewerRole()
        writer = SelfEvolutionWriterRole()

        def _analyzer_node(state: SelfEvolutionState) -> dict:
            project_dir = state.get("project_dir", ".")
            print(f"\n[analyzer] Scanning UMAF codebase at {project_dir}...")
            result = analyzer.execute(
                working_dir=working_dir,
                backend=state.get("backend", backend),
                project_dir=project_dir,
            )
            analysis = result if isinstance(result, dict) else {}
            opps = analysis.get("improvement_opportunities", [])
            print(f"[analyzer] Found {len(opps)} improvement opportunities.")
            return {"analysis_report": analysis, "status": "analyzed"}

        def _planner_node(state: SelfEvolutionState) -> dict:
            analysis = state.get("analysis_report", {})
            project_dir = state.get("project_dir", ".")
            print(f"\n[planner] Creating improvement plan...")
            result = planner.execute(
                working_dir=working_dir,
                backend=state.get("backend", backend),
                analysis_report=analysis,
                project_dir=project_dir,
            )
            plan = result if isinstance(result, dict) else {}
            improvements = plan.get("improvements", [])
            print(f"[planner] Plan created with {len(improvements)} improvements.")
            return {"implementation_plan": plan, "status": "planned"}

        def _coder_node(state: SelfEvolutionState) -> dict:
            plan = state.get("implementation_plan", {})
            project_dir = state.get("project_dir", ".")
            review_issues = state.get("review_issues", [])
            print(f"\n[coder] Implementing improvements...")
            result = coder.execute(
                working_dir=working_dir,
                backend=state.get("backend", backend),
                project_dir=project_dir,
                implementation_plan=plan,
                review_issues=review_issues if review_issues else None,
            )
            coder_result = result if isinstance(result, dict) else {}
            changed = coder_result.get("changed_files", [])
            print(f"[coder] Changed {len(changed)} file(s).")
            return {
                "changed_files": changed,
                "review_passed": False,
                "status": "implemented",
                "iteration": state["iteration"] + 1,
            }

        def _reviewer_node(state: SelfEvolutionState) -> dict:
            changed = state.get("changed_files", [])
            project_dir = state.get("project_dir", ".")
            print(f"\n[reviewer] Verifying {len(changed)} changed file(s)...")
            result = reviewer.execute(
                working_dir=working_dir,
                backend=state.get("backend", backend),
                project_dir=project_dir,
                changed_files=changed,
            )
            review_result = result if isinstance(result, dict) else {}
            passed = review_result.get("review_passed", False)
            issues = review_result.get("review_issues", [])
            test_results = review_result.get("test_results", "")
            if passed:
                print("[reviewer] REVIEW PASSED")
            elif state["iteration"] >= SelfEvolutionPipeline.MAX_ITERATIONS:
                print(f"[reviewer] REVIEW FAILED but max iterations ({SelfEvolutionPipeline.MAX_ITERATIONS}) reached — proceeding to writer.")
                return {
                    "review_passed": False,
                    "review_issues": issues,
                    "test_results": test_results,
                    "status": "verified",
                    "iteration": state["iteration"] + 1,
                }
            else:
                print(f"[reviewer] REVIEW FAILED — {len(issues)} issue(s)")
            return {
                "review_passed": passed,
                "review_issues": issues,
                "test_results": test_results,
                "status": "verified" if passed else "plan_revision",
                "iteration": state["iteration"] + 1,
            }

        def _writer_node(state: SelfEvolutionState) -> dict:
            changed = state.get("changed_files", [])
            passed = state.get("review_passed", False)
            test_results = state.get("test_results", "")
            print(f"\n[writer] Generating evolution report...")
            result = writer.execute(
                working_dir=working_dir,
                backend=state.get("backend", backend),
                changed_files=changed,
                review_passed=passed,
                test_results=test_results,
            )
            report_result = result if isinstance(result, dict) else {}
            report_path = report_result.get("evolution_report", "")
            print(f"[writer] Report: {report_path}")
            return {"evolution_report": report_path, "status": "completed"}

        flow = {
            "analyzed": "planner",
            "planned": "coder",
            "implemented": "reviewer",
            "verified": "writer",
            "plan_revision": "coder",
            "completed": END,
        }
        router = BasePipeline._status_router(flow, set())

        workflow.add_node("analyzer", _analyzer_node)
        workflow.add_node("planner", _planner_node)
        workflow.add_node("coder", _coder_node)
        workflow.add_node("reviewer", _reviewer_node)
        workflow.add_node("writer", _writer_node)
        workflow.set_entry_point("analyzer")
        for node in ("analyzer", "planner", "coder", "reviewer", "writer"):
            workflow.add_conditional_edges(node, router, {
                "planner": "planner", "coder": "coder",
                "reviewer": "reviewer", "writer": "writer", END: END,
            })

        return workflow.compile()

    def _print_results(self, final_state: dict):
        print("-" * 50)
        print(f"Status: {final_state.get('status', 'unknown')}")
        print(f"Iterations: {final_state.get('iteration', 0)}")
        changed = final_state.get("changed_files", [])
        print(f"Changed files: {len(changed)}")
        for f in changed:
            print(f"  - {f}")
        passed = final_state.get("review_passed", False)
        print(f"Review: {'PASSED' if passed else 'NOT PASSED'}")
        report = final_state.get("evolution_report", "")
        if report and os.path.exists(report):
            print(f"\nReport: {report}")
        print(f"\nOutputs in: {self.working_dir}")
