"""Tests for SelfEvolutionPipeline -- analyzer -> planner -> coder <-> reviewer -> writer."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest
import test.conftest  # noqa: F401 -- loads tools_config.json

from pipeline.self_evolution import SelfEvolutionPipeline, SelfEvolutionState
from self_evolution.analyzer import SelfEvolutionAnalyzerRole
from self_evolution.planner import SelfEvolutionPlannerRole
from self_evolution.coder import SelfEvolutionCoderRole
from self_evolution.reviewer import SelfEvolutionReviewerRole
from self_evolution.writer import SelfEvolutionWriterRole


# ============================================================
# SelfEvolutionState
# ============================================================

class TestSelfEvolutionState:
    def test_all_required_fields(self):
        fields = SelfEvolutionState.__annotations__
        required = [
            "input_spec", "working_dir", "backend", "project_dir",
            "status", "iteration", "analysis_report", "implementation_plan",
            "changed_files", "review_passed", "review_issues", "test_results",
            "evolution_report",
        ]
        for f in required:
            assert f in fields, f"Missing field: {f}"


# ============================================================
# Analyzer role
# ============================================================

class TestAnalyzerRole:
    def test_agent_name(self):
        role = SelfEvolutionAnalyzerRole()
        assert role.agent_name == "self_evolution_analyzer"

    def test_max_steps(self):
        role = SelfEvolutionAnalyzerRole()
        assert role.max_steps == 20

    def test_tools_for_backend(self):
        role = SelfEvolutionAnalyzerRole()
        for backend in ("deepseek", "claude_cli"):
            tools = role.tools_for_backend(backend)
            assert isinstance(tools, list)
            assert len(tools) > 0

    def test_fallback_analyze_produces_valid_report(self, tmpdir):
        analysis = SelfEvolutionAnalyzerRole._fallback_analyze(".", tmpdir)
        assert "project_overview" in analysis
        assert "log_analysis" in analysis
        assert "improvement_opportunities" in analysis
        assert "summary" in analysis
        opps = analysis["improvement_opportunities"]
        assert len(opps) >= 1
        for opp in opps:
            assert "id" in opp
            assert "category" in opp
            assert "title" in opp
            assert "severity" in opp

    def test_build_task_includes_project_dir(self):
        role = SelfEvolutionAnalyzerRole()
        task = role.build_task("deepseek", working_dir="/tmp", project_dir="myproject")
        assert "myproject" in task
        assert "analysis_report.json" in task
        assert "codebase" in task.lower() or "log" in task.lower()

    def test_build_task_includes_analysis_steps(self):
        role = SelfEvolutionAnalyzerRole()
        task = role.build_task("deepseek", working_dir="/tmp", project_dir=".")
        assert "Scan Codebase" in task or "codebase" in task.lower()
        assert "Agent Logs" in task or "agent log" in task.lower()
        assert "Improvement" in task

    def test_parse_result_reads_from_disk(self, tmpdir):
        report = {
            "project_overview": {"total_python_files": 10},
            "log_analysis": {"logs_found": False},
            "improvement_opportunities": [{"id": "SEO-001", "title": "Fix bug"}],
            "summary": "Test",
        }
        with open(os.path.join(tmpdir, "analysis_report.json"), "w") as f:
            json.dump(report, f)
        role = SelfEvolutionAnalyzerRole()
        mock = MagicMock()
        mock.messages = []
        result = role.parse_result(mock, working_dir=tmpdir, project_dir=".")
        assert result["summary"] == "Test"

    def test_parse_result_falls_back(self, tmpdir):
        role = SelfEvolutionAnalyzerRole()
        mock = MagicMock()
        mock.messages = []
        result = role.parse_result(mock, working_dir=tmpdir, project_dir=".")
        assert "_fallback" in result or "project_overview" in result


# ============================================================
# Planner role
# ============================================================

class TestPlannerRole:
    def test_agent_name(self):
        role = SelfEvolutionPlannerRole()
        assert role.agent_name == "self_evolution_planner"

    def test_max_steps(self):
        role = SelfEvolutionPlannerRole()
        assert role.max_steps == 15

    def test_tools_for_backend(self):
        role = SelfEvolutionPlannerRole()
        tools = role.tools_for_backend("deepseek")
        assert isinstance(tools, list)

    def test_fallback_plan_is_deterministic(self):
        plan = SelfEvolutionPlannerRole._fallback_plan(".", "/tmp")
        assert plan["_fallback"] is True
        assert "improvements" in plan
        assert "estimated_impact" in plan
        assert "risk_assessment" in plan

    def test_build_task_with_analysis_report(self):
        role = SelfEvolutionPlannerRole()
        report = {
            "summary": "Needs better test coverage",
            "improvement_opportunities": [
                {"id": "SEO-001", "severity": "high", "category": "test_gaps",
                 "title": "Add more tests"},
            ],
        }
        task = role.build_task("deepseek", working_dir="/tmp", analysis_report=report)
        assert "Needs better test coverage" in task
        assert "SEO-001" in task
        assert "implementation_plan.json" in task

    def test_build_task_without_report(self):
        role = SelfEvolutionPlannerRole()
        task = role.build_task("deepseek", working_dir="/tmp")
        assert "implementation_plan.json" in task

    def test_parse_result_reads_from_disk(self, tmpdir):
        plan = {"improvements": [{"id": "SEO-001"}], "estimated_impact": "Better"}
        with open(os.path.join(tmpdir, "implementation_plan.json"), "w") as f:
            json.dump(plan, f)
        role = SelfEvolutionPlannerRole()
        mock = MagicMock()
        mock.messages = []
        result = role.parse_result(mock, working_dir=tmpdir)
        assert result["improvements"][0]["id"] == "SEO-001"

    def test_parse_result_falls_back(self, tmpdir):
        role = SelfEvolutionPlannerRole()
        mock = MagicMock()
        mock.messages = []
        result = role.parse_result(mock, working_dir=tmpdir)
        assert result["_fallback"] is True


# ============================================================
# Coder role
# ============================================================

class TestCoderRole:
    def test_agent_name(self):
        role = SelfEvolutionCoderRole()
        assert role.agent_name == "self_evolution_coder"

    def test_max_steps(self):
        role = SelfEvolutionCoderRole()
        assert role.max_steps == 30

    def test_tools_include_write_and_run(self):
        role = SelfEvolutionCoderRole()
        tools = role.tools_for_backend("deepseek")
        names = {t["name"] for t in tools}
        assert "write_file" in names
        assert "run_command" in names

    def test_build_task_includes_improvement_plan(self):
        role = SelfEvolutionCoderRole()
        plan = {
            "improvements": [
                {"id": "SEO-001", "title": "Fix timeout", "action": "modify",
                 "files_to_modify": [{"path": "pipeline/research.py"}],
                 "files_to_create": [],
                 "verification": "pytest test/"},
            ],
        }
        task = role.build_task("deepseek", working_dir="/tmp",
                               project_dir=".", implementation_plan=plan)
        assert "SEO-001" in task
        assert "Fix timeout" in task
        assert "modify" in task

    def test_build_task_with_review_issues(self):
        role = SelfEvolutionCoderRole()
        task = role.build_task("deepseek", working_dir="/tmp",
                               project_dir=".",
                               review_issues=["Import missing in pipeline.py"])
        assert "Import missing" in task
        assert "FIX" in task

    def test_parse_result_empty_when_no_changes(self, tmpdir):
        role = SelfEvolutionCoderRole()
        mock = MagicMock()
        mock.messages = []
        mock.success = True
        result = role.parse_result(mock, working_dir=tmpdir, project_dir=".")
        assert "changed_files" in result


# ============================================================
# Reviewer role
# ============================================================

class TestReviewerRole:
    def test_agent_name(self):
        role = SelfEvolutionReviewerRole()
        assert role.agent_name == "self_evolution_reviewer"

    def test_max_steps(self):
        role = SelfEvolutionReviewerRole()
        assert role.max_steps == 12

    def test_tools_include_read_and_run(self):
        role = SelfEvolutionReviewerRole()
        tools = role.tools_for_backend("deepseek")
        names = {t["name"] for t in tools}
        assert "read_file" in names
        assert "run_command" in names

    def test_build_task_includes_changed_files(self):
        role = SelfEvolutionReviewerRole()
        task = role.build_task("deepseek", working_dir="/tmp",
                               project_dir=".",
                               changed_files=["pipeline/research.py", "agent.py"])
        assert "pipeline/research.py" in task
        assert "agent.py" in task
        assert "REVIEW_PASSED" in task
        assert "REVIEW_FAILED" in task

    def test_build_task_without_files(self):
        role = SelfEvolutionReviewerRole()
        task = role.build_task("deepseek", working_dir="/tmp", project_dir=".")
        assert "pytest" in task
        assert "REVIEW_PASSED" in task

    def test_parse_result_review_passed(self):
        role = SelfEvolutionReviewerRole()
        msg = MagicMock()
        msg.content = "Tests pass. REVIEW_PASSED"
        type(msg).__name__ = "AIMessage"
        mock = MagicMock()
        mock.messages = [msg]
        mock.success = True
        result = role.parse_result(mock, working_dir="/tmp")
        assert result["review_passed"] is True

    def test_parse_result_review_failed(self):
        role = SelfEvolutionReviewerRole()
        msg = MagicMock()
        msg.content = "Issues found.\n- Test failure in test_coder.py\nREVIEW_FAILED"
        type(msg).__name__ = "AIMessage"
        mock = MagicMock()
        mock.messages = [msg]
        mock.success = True
        result = role.parse_result(mock, working_dir="/tmp")
        assert result["review_passed"] is False
        assert len(result["review_issues"]) >= 1

    def test_parse_result_agent_not_successful(self):
        role = SelfEvolutionReviewerRole()
        mock = MagicMock()
        mock.messages = []
        mock.success = False
        result = role.parse_result(mock, working_dir="/tmp")
        assert result["review_passed"] is False
        assert "not complete" in result["review_issues"][0].lower()


# ============================================================
# Writer role
# ============================================================

class TestWriterRole:
    def test_agent_name(self):
        role = SelfEvolutionWriterRole()
        assert role.agent_name == "self_evolution_writer"

    def test_max_steps(self):
        role = SelfEvolutionWriterRole()
        assert role.max_steps == 8

    def test_tools_write_only(self):
        role = SelfEvolutionWriterRole()
        tools = role.tools_for_backend("deepseek")
        names = {t["name"] for t in tools}
        assert "write_file" in names
        assert "read_file" not in names

    def test_build_task_with_files_and_review(self):
        role = SelfEvolutionWriterRole()
        task = role.build_task("deepseek", working_dir="/tmp",
                               changed_files=["pipeline/research.py"],
                               review_passed=True,
                               test_results="204 passed")
        assert "pipeline/research.py" in task
        assert "PASSED" in task
        assert "204 passed" in task

    def test_build_task_no_files(self):
        role = SelfEvolutionWriterRole()
        task = role.build_task("deepseek", working_dir="/tmp")
        assert "evolution_report.md" in task

    def test_fallback_report_creates_file(self, tmpdir):
        result = SelfEvolutionWriterRole._fallback_report(tmpdir)
        assert os.path.exists(result["evolution_report"])
        content = open(result["evolution_report"]).read()
        assert "Self-Evolution" in content
        assert "# UMAF" in content

    def test_parse_result_reads_disk(self, tmpdir):
        with open(os.path.join(tmpdir, "evolution_report.md"), "w") as f:
            f.write("# Evolution Report")
        role = SelfEvolutionWriterRole()
        mock = MagicMock()
        mock.messages = []
        result = role.parse_result(mock, working_dir=tmpdir)
        assert "evolution_report" in result

    def test_parse_result_falls_back(self, tmpdir):
        role = SelfEvolutionWriterRole()
        mock = MagicMock()
        mock.messages = []
        result = role.parse_result(mock, working_dir=tmpdir)
        assert "evolution_report" in result
        assert os.path.exists(result["evolution_report"])


# ============================================================
# SelfEvolutionPipeline
# ============================================================

class TestSelfEvolutionPipeline:
    def test_pipeline_name(self):
        p = SelfEvolutionPipeline(working_dir="/tmp/test")
        assert p.name == "self_evolution"

    def test_default_output_dir(self):
        assert SelfEvolutionPipeline.default_output_dir == "self_evolution_output"

    def test_decompose_returns_single_task(self):
        p = SelfEvolutionPipeline(working_dir="/tmp/test")
        result = p._decompose("Improve test coverage")
        assert len(result) == 1
        assert result[0]["title"] == "Self-Evolution"

    def test_build_initial_state(self):
        p = SelfEvolutionPipeline(working_dir="/tmp/test", backend="claude_cli")
        state = p._build_initial_state("Improve tests", [{"id": 1}])
        assert state["input_spec"] == "Improve tests"
        assert state["working_dir"] == "/tmp/test"
        assert state["backend"] == "claude_cli"
        assert state["status"] == "initialized"
        assert state["iteration"] == 0
        assert state["review_passed"] is False
        assert state["changed_files"] == []
        assert state["analysis_report"] == {}
        assert state["implementation_plan"] == {}

    def test_build_graph_compiles(self):
        p = SelfEvolutionPipeline(working_dir="/tmp/test")
        graph = p._build_graph()
        assert graph is not None

    def test_display_decomposition(self, capsys):
        p = SelfEvolutionPipeline(working_dir="/tmp/test")
        p._display_decomposition([
            {"id": 1, "title": "Self-Evolution", "description": "Add tests"},
        ])
        out = capsys.readouterr().out
        assert "Add tests" in out

    def test_manage_output_dir(self, tmpdir):
        p = SelfEvolutionPipeline(working_dir=tmpdir)
        p.manage_output_dir()
        assert os.path.isdir(tmpdir)

    def test_constants(self):
        assert SelfEvolutionPipeline.MAX_ITERATIONS == 3
        assert SelfEvolutionPipeline.name == "self_evolution"


# ============================================================
# Pipeline graph nodes (mocked LLM calls)
# ============================================================

class TestSelfEvolutionGraphNodes:
    """Test graph node behavior by mocking role.execute() calls."""

    def test_graph_analyzer_to_planner_transition(self, tmpdir):
        """Analyzer -> Planner transition works."""
        analysis = {
            "project_overview": {"total_python_files": 50},
            "improvement_opportunities": [
                {"id": "SEO-001", "category": "test_gaps", "title": "More tests",
                 "severity": "medium"},
            ],
        }
        plan = {"improvements": [], "estimated_impact": "Better"}

        with patch.object(SelfEvolutionAnalyzerRole, "execute", return_value=analysis), \
             patch.object(SelfEvolutionPlannerRole, "execute", return_value=plan), \
             patch.object(SelfEvolutionCoderRole, "execute", return_value={"changed_files": [], "success": True}), \
             patch.object(SelfEvolutionReviewerRole, "execute",
                          return_value={"review_passed": True, "review_issues": []}):
            p = SelfEvolutionPipeline(working_dir=tmpdir, backend="deepseek", yes=True)
            graph = p._build_graph()

            state = p._build_initial_state("Improve UMAF", [{"id": 1}])
            result = graph.invoke(state)
            assert result["analysis_report"]["improvement_opportunities"][0]["id"] == "SEO-001"
            assert "improvements" in result["implementation_plan"]

    def test_router_max_iterations_stops_at_limit(self, tmpdir):
        """When iteration >= MAX_ITERATIONS, writer is called even on failure."""
        from self_evolution.analyzer import SelfEvolutionAnalyzerRole
        from self_evolution.planner import SelfEvolutionPlannerRole
        from self_evolution.coder import SelfEvolutionCoderRole
        from self_evolution.reviewer import SelfEvolutionReviewerRole
        from self_evolution.writer import SelfEvolutionWriterRole

        analysis = {"improvement_opportunities": [], "project_overview": {}, "log_analysis": {}}
        plan = {"improvements": [], "estimated_impact": "None"}
        writer_report = {"evolution_report": os.path.join(tmpdir, "evolution_report.md")}

        with patch.object(SelfEvolutionAnalyzerRole, "execute", return_value=analysis), \
             patch.object(SelfEvolutionPlannerRole, "execute", return_value=plan), \
             patch.object(SelfEvolutionCoderRole, "execute", return_value={"changed_files": [], "success": True}), \
             patch.object(SelfEvolutionReviewerRole, "execute",
                          return_value={"review_passed": False, "review_issues": ["Still failing"]}), \
             patch.object(SelfEvolutionWriterRole, "execute", return_value=writer_report):
            p = SelfEvolutionPipeline(working_dir=tmpdir, backend="deepseek", yes=True)
            graph = p._build_graph()

            state = p._build_initial_state("Improve UMAF", [{"id": 1}])
            result = graph.invoke(state, {"recursion_limit": 50})
            # After MAX_ITERATIONS retries, should exit with evolution_report
            assert result["iteration"] <= SelfEvolutionPipeline.MAX_ITERATIONS + 3
            assert "evolution_report" in result


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))