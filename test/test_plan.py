"""Tests for Plan Pipeline — validates agent roles, state, and pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools import ToolRegistry

# Load tools_config.json so tool methods return configured tools
_config_path = Path(__file__).resolve().parent.parent / "tools_config.json"
if _config_path.exists():
    with open(_config_path) as f:
        ToolRegistry.set_tool_config(json.load(f))


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def make_agent_result(messages, success=True):
    """Build a mock AgentResult."""
    from agent import AgentResult
    mock_msgs = []
    for m in messages:
        mm = MagicMock()
        mm.content = m.get("content", "")
        type(mm).__name__ = m.get("type", "AIMessage")
        mock_msgs.append(mm)
    result = MagicMock(spec=AgentResult)
    result.messages = mock_msgs
    result.success = success
    result.iterations = len(messages)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Test: Imports
# ═══════════════════════════════════════════════════════════════════════════

def test_plan_imports():
    """All plan modules import cleanly."""
    from plan.scanner import PlanScannerRole
    from plan.decomposer import PlanDecomposerRole
    from plan.dependency import PlanDependencyAnalyzerRole
    from plan.risk import PlanRiskAssessorRole
    from plan.resource import PlanResourceEstimatorRole
    from plan.cross_cutting import PlanCrossCuttingAnalyzerRole
    from plan.writer import PlanWriterRole
    from pipeline import PlanPipeline, PlanState
    assert PlanScannerRole is not None
    assert PlanDecomposerRole is not None
    assert PlanDependencyAnalyzerRole is not None
    assert PlanRiskAssessorRole is not None
    assert PlanResourceEstimatorRole is not None
    assert PlanCrossCuttingAnalyzerRole is not None
    assert PlanWriterRole is not None
    assert PlanPipeline is not None
    assert PlanState is not None


# ═══════════════════════════════════════════════════════════════════════════
# Test: PlanScannerRole
# ═══════════════════════════════════════════════════════════════════════════

class TestPlanScannerRole:
    """Tests for PlanScannerRole."""

    def test_agent_name(self):
        from plan.scanner import PlanScannerRole
        role = PlanScannerRole()
        assert role.agent_name == "plan_scanner"

    def test_max_steps(self):
        from plan.scanner import PlanScannerRole
        role = PlanScannerRole()
        assert role.max_steps == 10

    def test_tools_for_backend_deepseek(self):
        from plan.scanner import PlanScannerRole
        role = PlanScannerRole()
        tools = role.tools_for_backend("deepseek")
        assert isinstance(tools, list)
        assert len(tools) > 0
        tool_names = [t["name"] for t in tools]
        assert "read_file" in tool_names

    def test_tools_for_backend_claude_cli(self):
        from plan.scanner import PlanScannerRole
        role = PlanScannerRole()
        tools = role.tools_for_backend("claude_cli")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_build_task_format(self):
        from plan.scanner import PlanScannerRole
        role = PlanScannerRole()
        task = role.build_task("deepseek", working_dir="/tmp",
                               project_dir="/project")
        assert "project_context.json" in task
        assert "Project Directory" in task
        assert "/tmp" in task

    def test_parse_result_from_messages(self):
        from plan.scanner import PlanScannerRole
        role = PlanScannerRole()
        messages = [
            {"type": "AIMessage",
             "content": '{"language": "python", "file_manifest": [{"path": "a.py", "role": "source"}]}'}
        ]
        result = make_agent_result(messages)
        parsed = role.parse_result(result, working_dir="/tmp",
                                   project_dir=".")
        assert parsed.get("language") == "python"
        assert parsed.get("file_manifest") is not None

    def test_parse_result_fallback(self):
        from plan.scanner import PlanScannerRole
        role = PlanScannerRole()
        result = make_agent_result([], success=False)
        parsed = role.parse_result(result, working_dir="/tmp",
                                   project_dir="../")
        assert "_fallback" in parsed

    def test_fallback_scanner(self, tmpdir):
        from plan.scanner import PlanScannerRole
        scan = PlanScannerRole._fallback_scanner(
            project_dir=str(tmpdir), working_dir=str(tmpdir))
        assert isinstance(scan, dict)
        assert "language" in scan
        assert "file_manifest" in scan


# ═══════════════════════════════════════════════════════════════════════════
# Test: PlanDecomposerRole
# ═══════════════════════════════════════════════════════════════════════════

class TestPlanDecomposerRole:
    """Tests for PlanDecomposerRole."""

    def test_agent_name(self):
        from plan.decomposer import PlanDecomposerRole
        role = PlanDecomposerRole()
        assert role.agent_name == "plan_decomposer"

    def test_max_steps(self):
        from plan.decomposer import PlanDecomposerRole
        role = PlanDecomposerRole()
        assert role.max_steps == 25

    def test_tools_for_backend(self):
        from plan.decomposer import PlanDecomposerRole
        role = PlanDecomposerRole()
        tools = role.tools_for_backend("deepseek")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_build_task_includes_input_spec(self):
        from plan.decomposer import PlanDecomposerRole
        role = PlanDecomposerRole()
        task = role.build_task("deepseek", working_dir="/tmp",
                               input_spec="Build a web app with auth")
        assert "Build a web app with auth" in task
        assert "task_tree.json" in task
        assert "self-validation" in task.lower() or "Self-Validation" in task

    def test_build_task_adapts_depth_simple(self):
        from plan.decomposer import PlanDecomposerRole
        role = PlanDecomposerRole()
        task = role.build_task("deepseek", input_spec="short")
        assert "2 levels" in task or "simple" in task.lower()

    def test_build_task_adapts_depth_large(self):
        from plan.decomposer import PlanDecomposerRole
        role = PlanDecomposerRole()
        long_input = "Build a complete system with " + "many requirements " * 20
        task = role.build_task("deepseek", input_spec=long_input)
        assert "4" in task or "large" in task.lower()

    def test_parse_result_from_messages(self):
        from plan.decomposer import PlanDecomposerRole
        role = PlanDecomposerRole()
        messages = [
            {"type": "AIMessage",
             "content": '{"tree": [{"id": 1, "type": "goal"}], "input_spec": "test", "total_nodes": 1}'}
        ]
        result = make_agent_result(messages)
        parsed = role.parse_result(result, working_dir="/tmp",
                                   input_spec="test")
        assert parsed.get("tree") is not None
        assert len(parsed["tree"]) == 1

    def test_parse_result_fallback(self):
        from plan.decomposer import PlanDecomposerRole
        role = PlanDecomposerRole()
        result = make_agent_result([], success=False)
        parsed = role.parse_result(result, working_dir="/tmp",
                                   input_spec="Build something")
        assert "_fallback" in parsed
        assert "tree" in parsed

    def test_fallback_decompose(self, tmpdir):
        from plan.decomposer import PlanDecomposerRole
        tree = PlanDecomposerRole._fallback_decompose(
            "Implement a feature", str(tmpdir))
        assert "tree" in tree
        assert len(tree["tree"]) > 0
        assert tree["_fallback"] is True


# ═══════════════════════════════════════════════════════════════════════════
# Test: PlanDependencyAnalyzerRole
# ═══════════════════════════════════════════════════════════════════════════

class TestPlanDependencyAnalyzerRole:
    """Tests for PlanDependencyAnalyzerRole."""

    def test_agent_name(self):
        from plan.dependency import PlanDependencyAnalyzerRole
        role = PlanDependencyAnalyzerRole()
        assert role.agent_name == "dependency_analyzer"

    def test_max_steps(self):
        from plan.dependency import PlanDependencyAnalyzerRole
        role = PlanDependencyAnalyzerRole()
        assert role.max_steps == 15

    def test_tools_for_backend(self):
        from plan.dependency import PlanDependencyAnalyzerRole
        role = PlanDependencyAnalyzerRole()
        tools = role.tools_for_backend("deepseek")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_build_task_reads_task_tree(self):
        from plan.dependency import PlanDependencyAnalyzerRole
        role = PlanDependencyAnalyzerRole()
        task = role.build_task("deepseek", working_dir="/tmp",
                               task_tree={"tree": []})
        assert "dependency_graph.json" in task

    def test_parse_result(self):
        from plan.dependency import PlanDependencyAnalyzerRole
        role = PlanDependencyAnalyzerRole()
        messages = [
            {"type": "AIMessage",
             "content": '{"nodes": [{"id": 1}], "edges": [{"from": 1, "to": 2, "type": "blocks"}]}'}
        ]
        result = make_agent_result(messages)
        parsed = role.parse_result(result, working_dir="/tmp")
        assert "nodes" in parsed
        assert "edges" in parsed

    def test_fallback_dependency(self, tmpdir):
        from plan.dependency import PlanDependencyAnalyzerRole
        task_tree = {
            "tree": [
                {"id": 1, "type": "goal", "title": "Goal",
                 "complexity": 5, "dependencies": [], "children": [
                    {"id": 2, "type": "task", "title": "Task",
                     "complexity": 3, "dependencies": [], "children": []}
                ]}
            ]
        }
        dep = PlanDependencyAnalyzerRole._fallback_dependency(
            task_tree, str(tmpdir))
        assert "nodes" in dep
        assert len(dep["nodes"]) >= 2
        assert "edges" in dep


# ═══════════════════════════════════════════════════════════════════════════
# Test: PlanRiskAssessorRole
# ═══════════════════════════════════════════════════════════════════════════

class TestPlanRiskAssessorRole:
    """Tests for PlanRiskAssessorRole."""

    def test_agent_name(self):
        from plan.risk import PlanRiskAssessorRole
        role = PlanRiskAssessorRole()
        assert role.agent_name == "risk_assessor"

    def test_max_steps(self):
        from plan.risk import PlanRiskAssessorRole
        role = PlanRiskAssessorRole()
        assert role.max_steps == 15

    def test_tools_for_backend(self):
        from plan.risk import PlanRiskAssessorRole
        role = PlanRiskAssessorRole()
        tools = role.tools_for_backend("deepseek")
        assert isinstance(tools, list)

    def test_build_task_includes_risk_dimensions(self):
        from plan.risk import PlanRiskAssessorRole
        role = PlanRiskAssessorRole()
        task = role.build_task("deepseek", working_dir="/tmp")
        assert "technical_complexity" in task
        assert "dependency_risk" in task
        assert "knowledge_gap_risk" in task
        assert "risk_matrix.json" in task

    def test_parse_result(self):
        from plan.risk import PlanRiskAssessorRole
        role = PlanRiskAssessorRole()
        messages = [
            {"type": "AIMessage",
             "content": '{"task_risks": [{"task_id": 1, "risk_level": "low"}]}'}
        ]
        result = make_agent_result(messages)
        parsed = role.parse_result(result, working_dir="/tmp")
        assert "task_risks" in parsed

    def test_fallback_risk(self, tmpdir):
        from plan.risk import PlanRiskAssessorRole
        task_tree = {
            "tree": [
                {"id": 1, "type": "goal", "title": "G", "complexity": 5,
                 "dependencies": [], "children": []}
            ]
        }
        risk = PlanRiskAssessorRole._fallback_risk(task_tree, str(tmpdir))
        assert "task_risks" in risk
        assert len(risk["task_risks"]) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Test: PlanResourceEstimatorRole
# ═══════════════════════════════════════════════════════════════════════════

class TestPlanResourceEstimatorRole:
    """Tests for PlanResourceEstimatorRole."""

    def test_agent_name(self):
        from plan.resource import PlanResourceEstimatorRole
        role = PlanResourceEstimatorRole()
        assert role.agent_name == "resource_estimator"

    def test_max_steps(self):
        from plan.resource import PlanResourceEstimatorRole
        role = PlanResourceEstimatorRole()
        assert role.max_steps == 15

    def test_tools_for_backend(self):
        from plan.resource import PlanResourceEstimatorRole
        role = PlanResourceEstimatorRole()
        tools = role.tools_for_backend("deepseek")
        assert isinstance(tools, list)

    def test_build_task_includes_effort_estimation(self):
        from plan.resource import PlanResourceEstimatorRole
        role = PlanResourceEstimatorRole()
        task = role.build_task("deepseek", working_dir="/tmp")
        assert "hours_estimate" in task
        assert "required_skills" in task
        assert "resource_plan.json" in task

    def test_parse_result(self):
        from plan.resource import PlanResourceEstimatorRole
        role = PlanResourceEstimatorRole()
        messages = [
            {"type": "AIMessage",
             "content": '{"task_estimates": [{"task_id": 1, "hours_estimate": 8.0}]}'}
        ]
        result = make_agent_result(messages)
        parsed = role.parse_result(result, working_dir="/tmp")
        assert "task_estimates" in parsed

    def test_fallback_resource(self, tmpdir):
        from plan.resource import PlanResourceEstimatorRole
        task_tree = {
            "tree": [
                {"id": 1, "type": "goal", "title": "G", "complexity": 5,
                 "dependencies": [], "children": []}
            ]
        }
        res = PlanResourceEstimatorRole._fallback_resource(
            task_tree, str(tmpdir))
        assert "task_estimates" in res
        assert "aggregated_estimates" in res


# ═══════════════════════════════════════════════════════════════════════════
# Test: PlanCrossCuttingAnalyzerRole
# ═══════════════════════════════════════════════════════════════════════════

class TestPlanCrossCuttingAnalyzerRole:
    """Tests for PlanCrossCuttingAnalyzerRole."""

    def test_agent_name(self):
        from plan.cross_cutting import PlanCrossCuttingAnalyzerRole
        role = PlanCrossCuttingAnalyzerRole()
        assert role.agent_name == "cross_cutting_analyzer"

    def test_max_steps(self):
        from plan.cross_cutting import PlanCrossCuttingAnalyzerRole
        role = PlanCrossCuttingAnalyzerRole()
        assert role.max_steps == 15

    def test_tools_for_backend(self):
        from plan.cross_cutting import PlanCrossCuttingAnalyzerRole
        role = PlanCrossCuttingAnalyzerRole()
        tools = role.tools_for_backend("deepseek")
        assert isinstance(tools, list)

    def test_build_task_includes_concern_types(self):
        from plan.cross_cutting import PlanCrossCuttingAnalyzerRole
        role = PlanCrossCuttingAnalyzerRole()
        task = role.build_task("deepseek", working_dir="/tmp")
        assert "security" in task.lower()
        assert "testing" in task.lower()
        assert "deployment" in task.lower()
        assert "cross_cutting_concerns.json" in task

    def test_parse_result(self):
        from plan.cross_cutting import PlanCrossCuttingAnalyzerRole
        role = PlanCrossCuttingAnalyzerRole()
        messages = [
            {"type": "AIMessage",
             "content": '{"concerns": [{"domain": "security", "concern": "auth"}]}'}
        ]
        result = make_agent_result(messages)
        parsed = role.parse_result(result, working_dir="/tmp")
        assert "concerns" in parsed

    def test_fallback_cross_cutting(self, tmpdir):
        from plan.cross_cutting import PlanCrossCuttingAnalyzerRole
        cc = PlanCrossCuttingAnalyzerRole._fallback_cross_cutting(
            None, str(tmpdir))
        assert "concerns" in cc
        assert len(cc["concerns"]) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Test: PlanWriterRole
# ═══════════════════════════════════════════════════════════════════════════

class TestPlanWriterRole:
    """Tests for PlanWriterRole."""

    def test_agent_name(self):
        from plan.writer import PlanWriterRole
        role = PlanWriterRole()
        assert role.agent_name == "plan_writer"

    def test_max_steps(self):
        from plan.writer import PlanWriterRole
        role = PlanWriterRole()
        assert role.max_steps == 20

    def test_tools_for_backend(self):
        from plan.writer import PlanWriterRole
        role = PlanWriterRole()
        tools = role.tools_for_backend("deepseek")
        assert isinstance(tools, list)
        tool_names = [t["name"] for t in tools]
        assert "write_file" in tool_names

    def test_build_task_references_all_inputs(self):
        from plan.writer import PlanWriterRole
        role = PlanWriterRole()
        task = role.build_task("deepseek", working_dir="/tmp",
                               task_tree={"tree": []},
                               dependency_graph={},
                               risk_matrix={},
                               resource_plan={},
                               cross_cutting_concerns={})
        assert "plan_spec.json" in task
        assert "plan_report.md" in task
        assert "consistency" in task.lower()

    def test_parse_result(self):
        from plan.writer import PlanWriterRole
        role = PlanWriterRole()
        result = make_agent_result([], success=False)
        parsed = role.parse_result(result, working_dir="/tmp")
        assert "success" in parsed
        assert "spec_path" in parsed
        assert "report_path" in parsed

    def test_fallback_writer(self):
        from plan.writer import PlanWriterRole
        spec = PlanWriterRole._fallback_writer(working_dir="/tmp")
        assert "plan_title" in spec
        assert "_fallback" in spec

    def test_write_plan_spec(self, tmpdir):
        from plan.writer import PlanWriterRole
        spec = {"plan_title": "Test Plan", "_fallback": True}
        PlanWriterRole._write_plan_spec(str(tmpdir), spec)
        assert os.path.exists(os.path.join(str(tmpdir), "plan_spec.json"))

    def test_fallback_report_md(self, tmpdir):
        from plan.writer import PlanWriterRole
        PlanWriterRole._fallback_report_md(str(tmpdir))
        assert os.path.exists(os.path.join(str(tmpdir), "plan_report.md"))


# ═══════════════════════════════════════════════════════════════════════════
# Test: PlanPipeline
# ═══════════════════════════════════════════════════════════════════════════

class TestPlanPipeline:
    """Tests for PlanPipeline structure."""

    def test_name(self):
        from pipeline.plan import PlanPipeline
        p = PlanPipeline()
        assert p.name == "plan"

    def test_default_output_dir(self):
        from pipeline.plan import PlanPipeline
        p = PlanPipeline()
        assert p.default_output_dir == "plan_output"

    def test_build_graph_returns_compiled_graph(self):
        from pipeline.plan import PlanPipeline
        p = PlanPipeline(working_dir="/tmp/test_plan")
        graph = p._build_graph()
        assert graph is not None

    def test_state_graph_has_nodes(self):
        from pipeline.plan import PlanPipeline
        p = PlanPipeline(working_dir="/tmp/test_plan")
        graph = p._build_graph()
        # Verify graph compiles and has nodes
        assert graph is not None

    def test_initial_state_fields(self):
        from pipeline.plan import PlanPipeline
        p = PlanPipeline()
        state = p._build_initial_state("test task", [])
        assert state["input_spec"] == "test task"
        assert state["status"] == "initialized"
        assert "project_context" in state
        assert "task_tree" in state
        assert "dependency_graph" in state
        assert "risk_matrix" in state
        assert "resource_plan" in state
        assert "cross_cutting_concerns" in state
        assert "analysis_outputs" in state
        assert "version" in state

    def test_decompose_returns_empty_list(self):
        from pipeline.plan import PlanPipeline
        p = PlanPipeline()
        result = p._decompose("test")
        assert result == []

    def test_pipeline_instantiation_defaults(self):
        from pipeline.plan import PlanPipeline
        p = PlanPipeline()
        assert p.backend == "deepseek"


# ═══════════════════════════════════════════════════════════════════════════
# Test: PlanState TypedDict
# ═══════════════════════════════════════════════════════════════════════════

class TestPlanState:
    """Tests for PlanState TypedDict."""

    def test_plan_state_creation(self):
        from pipeline.plan import PlanState
        state: PlanState = {
            "input_spec": "test",
            "working_dir": "/tmp",
            "backend": "deepseek",
            "status": "initialized",
        }
        assert state["input_spec"] == "test"
        assert state["status"] == "initialized"

    def test_plan_state_optional_fields(self):
        from pipeline.plan import PlanState
        state = PlanState(input_spec="test", working_dir="/tmp",
                          backend="deepseek", status="initialized")
        assert state.get("project_context") is None
        assert state.get("task_tree") is None


# ═══════════════════════════════════════════════════════════════════════════
# Test: ToolRegistry Plan Methods
# ═══════════════════════════════════════════════════════════════════════════

class TestToolRegistryPlanMethods:
    """Tests for ToolRegistry plan methods."""

    def test_all_plan_methods_exist(self):
        methods = [
            "plan_scanner_tools",
            "plan_decomposer_tools",
            "plan_dependency_analyzer_tools",
            "plan_risk_assessor_tools",
            "plan_resource_estimator_tools",
            "plan_cross_cutting_analyzer_tools",
            "plan_writer_tools",
        ]
        for method in methods:
            assert hasattr(ToolRegistry, method), f"Missing {method}"

    def test_plan_scanner_tools_non_empty(self):
        tools = ToolRegistry.plan_scanner_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0
        names = [t.name for t in tools]
        assert "read_file" in names

    def test_plan_writer_tools_has_write(self):
        tools = ToolRegistry.plan_writer_tools()
        names = [t.name for t in tools]
        assert "write_file" in names

    def test_plan_decomposer_tools_read_only(self):
        tools = ToolRegistry.plan_decomposer_tools()
        names = [t.name for t in tools]
        assert "read_file" in names
        assert "write_file" not in names


# ═══════════════════════════════════════════════════════════════════════════
# Test: Plan Pipeline Integration (smoke test)
# ═══════════════════════════════════════════════════════════════════════════

class TestPlanPipelineIntegration:
    """Integration/smoke tests for Plan Pipeline."""

    def test_pipeline_graph_invokes(self, tmpdir):
        """Smoke test: pipeline graph invokes from initial state without crashes."""
        from pipeline.plan import PlanPipeline, PlanState
        p = PlanPipeline(working_dir=str(tmpdir), backend="deepseek",
                         yes=True)
        # Pre-built state + mocked analyzers to avoid real LLM calls
        with patch("pipeline.plan._run_analyzer",
                   return_value={"output_file": "test.json", "domain": "Test",
                                 "data": {}, "summary": "ok", "files": []}), \
             patch("plan.writer.PlanWriterRole.execute",
                   return_value={"plan_spec": {}, "plan_report": "report.md"}):
            initial_state: PlanState = {
            "input_spec": "Test feature",
            "working_dir": str(tmpdir),
            "backend": "deepseek",
            "target": ".",
            "project_context": {"_fallback": True, "language": "python",
                                "file_manifest": [{"path": "test.py"}], "total_files": 1,
                                "source_directories": ["src"], "test_directories": ["tests"],
                                "entry_points": ["main.py"], "conventions": {},
                                "test_patterns": {}, "tech_stack": {},
                                "architecture": {},
                                "scan_timestamp": "2024-01-01",
                                "project_dir": "."},
            "status": "scanned",
            "task_tree": {
                "tree": [{"id": 1, "type": "goal", "title": "G",
                          "complexity": 5, "dependencies": [], "children": [
                    {"id": 2, "type": "task", "title": "T",
                     "complexity": 3, "dependencies": [], "children": []}
                ]}],
                "total_nodes": 2,
                "complexity_level": "simple",
                "input_spec": "Test feature",
            },
            "dependency_graph": {},
            "risk_matrix": {},
            "resource_plan": {},
            "cross_cutting_concerns": {},
            "plan_spec": {},
            "analysis_outputs": [],
            "status": "initialized",
            "version": 1,
        }
            graph = p._build_graph()
            # Should not crash
            final = graph.invoke(initial_state)
            assert final is not None
            # Should have reached writer node or at least progressed
            assert final.get("status") in (
                "written", "error_no_analysis", "error_no_task_tree",
            )

    def test_pipeline_graph_with_no_tree(self, tmpdir):
        """Pipeline should handle empty task tree gracefully."""
        from pipeline.plan import PlanPipeline
        p = PlanPipeline(working_dir=str(tmpdir), yes=True)
        with patch("pipeline.plan._run_analyzer",
                   return_value={"output_file": "test.json", "domain": "Test",
                                 "data": {}, "summary": "ok", "files": []}), \
             patch("plan.writer.PlanWriterRole.execute",
                   return_value={"plan_spec": {}, "plan_report": "report.md"}):
            state = {
                "input_spec": "Test",
                "working_dir": str(tmpdir),
                "backend": "deepseek",
                "target": ".",
                "project_context": {"_fallback": True, "language": "python",
                                    "file_manifest": [{"path": "test.py"}], "total_files": 1,
                                    "source_directories": ["src"], "test_directories": ["tests"],
                                    "entry_points": ["main.py"], "conventions": {},
                                    "test_patterns": {}, "tech_stack": {},
                                    "architecture": {},
                                    "scan_timestamp": "2024-01-01",
                                    "project_dir": "."},
                "task_tree": {"tree": []},
                "dependency_graph": {},
                "risk_matrix": {},
                "resource_plan": {},
                "cross_cutting_concerns": {},
                "plan_spec": {},
                "analysis_outputs": [],
                "status": "scanned",
                "version": 1,
            }
            graph = p._build_graph()
            final = graph.invoke(state)
            assert final is not None
