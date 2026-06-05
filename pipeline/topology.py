"""Topology Optimizer Pipeline — analyze task → design → evaluate → write spec."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from topology.analyzer import TopologyAnalyzerRole
from topology.designer import TopologyDesignerRole
from topology.evaluator import TopologyEvaluatorRole
from topology.writer import TopologyWriterRole
from .base import BasePipeline


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
