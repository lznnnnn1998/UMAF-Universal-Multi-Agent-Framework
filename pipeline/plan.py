"""Plan Pipeline — scanner → decomposer → 4 parallel analyzers → writer."""

from __future__ import annotations

import json
import os
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from plan.scanner import PlanScannerRole
from plan.decomposer import PlanDecomposerRole
from plan.dependency import PlanDependencyAnalyzerRole
from plan.risk import PlanRiskAssessorRole
from plan.resource import PlanResourceEstimatorRole
from plan.cross_cutting import PlanCrossCuttingAnalyzerRole
from plan.writer import PlanWriterRole
from .base import BasePipeline


_ANALYZER_CLASSES: dict[str, type] = {
    "Dependency Analysis": PlanDependencyAnalyzerRole,
    "Risk Assessment": PlanRiskAssessorRole,
    "Resource Estimation": PlanResourceEstimatorRole,
    "Cross-Cutting Concerns": PlanCrossCuttingAnalyzerRole,
}

_ANALYZER_OUTPUT_FILES: dict[str, str] = {
    "Dependency Analysis": "dependency_graph.json",
    "Risk Assessment": "risk_matrix.json",
    "Resource Estimation": "resource_plan.json",
    "Cross-Cutting Concerns": "cross_cutting_concerns.json",
}


def _run_analyzer(item: dict[str, Any], working_dir: str,
                  backend: str) -> dict[str, Any]:
    """Execute a single plan dimension analyzer with LLM + fallback."""
    domain = item.get("domain", "")
    analyzer_cls = _ANALYZER_CLASSES.get(domain)
    output_file = _ANALYZER_OUTPUT_FILES.get(domain, "")
    task_tree = item.get("task_tree", {})

    if analyzer_cls is None:
        return {"output_file": "", "domain": domain, "data": {},
                "summary": f"Unknown domain: {domain}", "files": []}

    try:
        role = analyzer_cls()
        try:
            analysis = role.execute(working_dir=working_dir, backend=backend,
                                    task_tree=task_tree)
        except Exception:
            analysis = {}

        if not analysis or (not analysis.get("task_risks")
                            and not analysis.get("nodes")
                            and not analysis.get("task_estimates")
                            and not analysis.get("concerns")):
            # Call the appropriate fallback
            if hasattr(analyzer_cls, "_fallback_dependency"):
                analysis = analyzer_cls._fallback_dependency(task_tree, working_dir)
            elif hasattr(analyzer_cls, "_fallback_risk"):
                analysis = analyzer_cls._fallback_risk(task_tree, working_dir)
            elif hasattr(analyzer_cls, "_fallback_resource"):
                analysis = analyzer_cls._fallback_resource(task_tree, working_dir)
            elif hasattr(analyzer_cls, "_fallback_cross_cutting"):
                analysis = analyzer_cls._fallback_cross_cutting(task_tree, working_dir)

        report_path = os.path.join(working_dir, output_file)
        if analysis:
            try:
                with open(report_path, "w") as f:
                    json.dump(analysis, f, indent=2, default=str)
            except OSError:
                pass

        node_count = (len(analysis.get("nodes", []))
                      or len(analysis.get("task_risks", []))
                      or len(analysis.get("task_estimates", []))
                      or len(analysis.get("concerns", [])))
        summary = f"Analyzed {node_count} items" if node_count else "No items analyzed"
        return {
            "output_file": output_file, "domain": domain, "data": analysis,
            "summary": summary,
            "files": [output_file] if os.path.exists(report_path) else [],
        }
    except Exception as exc:
        return {"output_file": "", "domain": domain, "data": {},
                "summary": f"Analyzer exception: {exc}", "files": []}


class PlanState(TypedDict, total=False):
    """State for the Plan Pipeline.

    Fields are populated progressively as the pipeline executes:
    scanner → decomposer → 4 parallel analyzers → writer.
    """
    input_spec: str
    working_dir: str
    backend: str
    target: str
    project_context: dict[str, Any]
    task_tree: dict[str, Any]
    dependency_graph: dict[str, Any]
    risk_matrix: dict[str, Any]
    resource_plan: dict[str, Any]
    cross_cutting_concerns: dict[str, Any]
    plan_spec: dict[str, Any]
    analysis_outputs: list[dict[str, Any]]
    status: str
    version: int


class PlanPipeline(BasePipeline):
    """Plan Pipeline with 7-agent fan-out/fan-in topology.

    scanner → decomposer → [dependency ‖ risk ‖ resource ‖ cross-cutting] → writer → END

    Transforms a natural language task description into a comprehensive,
    structured implementation plan with hierarchical task decomposition,
    dependency analysis, risk assessment, resource estimation, cross-cutting
    concern mapping, and both machine-readable (JSON) and human-readable
    (Markdown) deliverables.
    """

    name = "plan"
    default_output_dir = "plan_output"

    def _decompose(self, input_spec: str) -> list[dict[str, Any]]:
        return []

    def _display_decomposition(self, sub_tasks: list[dict]):
        print("Plan Pipeline — generating structured implementation plan.")
        print(f"Backend: {self.backend}")

    def _build_initial_state(self, input_spec: str,
                             sub_tasks: list[dict]) -> dict:
        target = getattr(self, "target_dir", None)
        return {
            "input_spec": input_spec,
            "working_dir": self.working_dir,
            "backend": self.backend,
            "target": target or "",
            "project_context": {},
            "task_tree": {},
            "dependency_graph": {},
            "risk_matrix": {},
            "resource_plan": {},
            "cross_cutting_concerns": {},
            "plan_spec": {},
            "analysis_outputs": [],
            "status": "initialized",
            "version": 1,
        }

    def _build_graph(self):
        working_dir = self.working_dir
        backend = self.backend

        def _scanner_node(state: PlanState) -> dict:
            print("\n[scanner] Gathering project context...")
            target = state.get("target") or "."
            print(f"  Target directory: {target}")

            # Skip scanning if we already have context
            existing = state.get("project_context", {})
            if existing and existing.get("file_manifest"):
                print("  [scanner] Using existing project context")
                return {"status": "scanned"}

            scan: dict[str, Any] = {}
            try:
                role = PlanScannerRole()
                scan = role.execute(working_dir=working_dir,
                                    backend=state.get("backend", backend),
                                    project_dir=target)
            except Exception as exc:
                print(f"  [scanner] Agent error: {exc}")

            if not scan or not scan.get("file_manifest"):
                print("  [scanner] Falling back to deterministic scan...")
                scan = PlanScannerRole._fallback_scanner(
                    project_dir=target, working_dir=working_dir)

            lang = scan.get("language", "?")
            files = scan.get("total_files", 0)
            print(f"  [scanner] Language: {lang}, Files: {files}")
            return {
                "project_context": scan,
                "status": "scanned",
            }

        def _decomposer_node(state: PlanState) -> dict:
            print("\n[decomposer] Building hierarchical task tree...")
            input_spec = state.get("input_spec", "")
            project_context = state.get("project_context", {})

            # Skip if task tree already provided (resume / pre-populated state)
            existing_tree = state.get("task_tree", {})
            if existing_tree and existing_tree.get("tree") is not None:
                print("  [decomposer] Using existing task tree")
                return {"status": "decomposed"}

            task_tree: dict[str, Any] = {}
            try:
                role = PlanDecomposerRole()
                task_tree = role.execute(working_dir=working_dir,
                                         backend=state.get("backend", backend),
                                         input_spec=input_spec,
                                         project_context=project_context)
            except Exception as exc:
                print(f"  [decomposer] Agent error: {exc}")

            if not task_tree or not task_tree.get("tree"):
                print("  [decomposer] Falling back to template decomposition...")
                task_tree = PlanDecomposerRole._fallback_decompose(
                    input_spec, working_dir)
                out_path = os.path.join(working_dir, "task_tree.json")
                try:
                    with open(out_path, "w") as f:
                        json.dump(task_tree, f, indent=2)
                except OSError:
                    pass

            nodes = task_tree.get("total_nodes", 0)
            complexity = task_tree.get("complexity_level", "?")
            print(f"  [decomposer] Complexity: {complexity}, Nodes: {nodes}")
            return {
                "task_tree": task_tree,
                "status": "decomposed",
            }

        def _analyzers_node(state: PlanState) -> dict:
            print("\n[analyzers] Running 4 parallel analyzers...")
            task_tree = state.get("task_tree", {})

            items: list[dict[str, Any]] = [
                {"domain": "Dependency Analysis", "task_tree": task_tree},
                {"domain": "Risk Assessment", "task_tree": task_tree},
                {"domain": "Resource Estimation", "task_tree": task_tree},
                {"domain": "Cross-Cutting Concerns", "task_tree": task_tree},
            ]

            # Limit parallelism for claude_cli backend
            max_w = 1 if state.get("backend", backend) == "claude_cli" else 4
            outputs, succeeded, failed = BasePipeline._run_parallel_agents(
                items, _run_analyzer, working_dir,
                state.get("backend", backend),
                max_workers=max_w,
            )

            # Map outputs to state fields
            dep_graph: dict[str, Any] = {}
            risk_matrix: dict[str, Any] = {}
            resource_plan: dict[str, Any] = {}
            cross_cutting: dict[str, Any] = {}

            ok = 0
            for out in outputs:
                domain = out.get("domain", "")
                data = out.get("data", {})
                if out.get("output_file"):
                    ok += 1
                if domain == "Dependency Analysis":
                    dep_graph = data
                elif domain == "Risk Assessment":
                    risk_matrix = data
                elif domain == "Resource Estimation":
                    resource_plan = data
                elif domain == "Cross-Cutting Concerns":
                    cross_cutting = data

            print(f"  [analyzers] {ok}/{len(outputs)} succeeded")
            for out in outputs:
                mark = "✓" if out.get("output_file") else "✗"
                print(f"    {mark} {out['domain']}: {out.get('summary', 'no output')[:120]}")

            status = "analyzed" if ok > 0 else "error_no_analysis"
            return {
                "dependency_graph": dep_graph,
                "risk_matrix": risk_matrix,
                "resource_plan": resource_plan,
                "cross_cutting_concerns": cross_cutting,
                "analysis_outputs": outputs,
                "status": status,
            }

        def _writer_node(state: PlanState) -> dict:
            print("\n[writer] Synthesizing final plan deliverables...")
            project_context = state.get("project_context", {})
            task_tree = state.get("task_tree", {})
            dep_graph = state.get("dependency_graph", {})
            risk_matrix = state.get("risk_matrix", {})
            resource_plan = state.get("resource_plan", {})
            cross_cutting = state.get("cross_cutting_concerns", {})

            if not task_tree.get("tree"):
                print("  [writer] No task tree — nothing to write")
                return {"status": "error_no_task_tree"}

            agent_ok = False
            try:
                role = PlanWriterRole()
                result = role.execute(
                    working_dir=working_dir,
                    backend=state.get("backend", backend),
                    project_context=project_context,
                    task_tree=task_tree,
                    dependency_graph=dep_graph,
                    risk_matrix=risk_matrix,
                    resource_plan=resource_plan,
                    cross_cutting_concerns=cross_cutting,
                )
                if result.get("success"):
                    agent_ok = True
            except Exception as exc:
                print(f"  [writer] Agent error: {exc}")

            if not agent_ok:
                print("  [writer] Falling back to template generation...")
                spec = PlanWriterRole._fallback_writer(
                    working_dir, task_tree=task_tree,
                    dependency_graph=dep_graph,
                    risk_matrix=risk_matrix,
                    resource_plan=resource_plan,
                    cross_cutting_concerns=cross_cutting,
                )
                PlanWriterRole._write_plan_spec(working_dir, spec)
                PlanWriterRole._fallback_report_md(
                    working_dir, task_tree=task_tree,
                    dependency_graph=dep_graph,
                    risk_matrix=risk_matrix,
                    resource_plan=resource_plan,
                    cross_cutting_concerns=cross_cutting,
                    spec=spec,
                )

            for fname in ("plan_spec.json", "plan_report.md"):
                fpath = os.path.join(working_dir, fname)
                if os.path.exists(fpath):
                    size = os.path.getsize(fpath)
                    print(f"  [writer] ✓ {fname} ({size} bytes)")
                else:
                    print(f"  [writer] ✗ {fname} not found")

            return {"status": "written"}

        workflow = StateGraph(PlanState)
        workflow.add_node("scanner", _scanner_node)
        workflow.add_node("decomposer", _decomposer_node)
        workflow.add_node("analyzers", _analyzers_node)
        workflow.add_node("writer", _writer_node)
        workflow.set_entry_point("scanner")

        flow = {
            "initialized": "scanner",
            "scanned": "decomposer",
            "decomposed": "analyzers",
            "analyzed": "writer",
            "written": END,
        }
        terminal = {"error_no_context", "error_no_task_tree", "error_no_analysis"}
        router = BasePipeline._status_router(flow, terminal)
        for node in ("scanner", "decomposer", "analyzers", "writer"):
            workflow.add_conditional_edges(node, router, {
                "scanner": "scanner", "decomposer": "decomposer",
                "analyzers": "analyzers", "writer": "writer", END: END,
            })
        return workflow.compile()

    def _print_results(self, final_state: dict):
        print("-" * 60)
        status = final_state.get("status", "unknown")
        print(f"Plan Pipeline — {status.upper()}")
        print("-" * 60)

        pc = final_state.get("project_context", {})
        if pc:
            print(f"\nProject Context:")
            print(f"   Language: {pc.get('language', '?')}")
            print(f"   Files: {pc.get('total_files', '?')}")

        tt = final_state.get("task_tree", {})
        if tt:
            print(f"\nTask Tree:")
            print(f"   Complexity: {tt.get('complexity_level', '?')}")
            print(f"   Total Nodes: {tt.get('total_nodes', '?')}")
            val = tt.get("validation", {})
            if val:
                print(f"   Validation: complete={val.get('complete')}, "
                      f"coherent={val.get('coherent')}, "
                      f"acyclic={val.get('acyclic')}")

        ao = final_state.get("analysis_outputs", [])
        if ao:
            print(f"\nAnalysis Dimensions:")
            for a in ao:
                mark = "✓" if a.get("output_file") else "✗"
                print(f"   {mark} {a['domain']}: {a.get('summary', 'no output')}")

        print(f"\nOutputs in: {self.working_dir}")
