"""Tests for Feature Pipeline v2 — 5-node graph with coder/reviewer loop."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pipeline import FeaturePipeline, FeatureState
from feature.scanner import FeatureScannerRole
from feature.planner import FeaturePlannerRole
from feature.coder import FeatureCoderRole
from feature.reviewer import FeatureReviewerRole
from feature.writer import FeatureReportWriterRole
from tools import ToolRegistry
from utils import extract_json_object, safe_read

# Load tools_config.json so tool methods return configured tools
_config_path = Path(__file__).resolve().parent.parent / "tools_config.json"
if _config_path.exists():
    with open(_config_path) as f:
        ToolRegistry.set_tool_config(json.load(f))


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _make_agent_result(messages: list[dict], success: bool = True):
    """Build a mock AgentResult with given messages."""
    mock_msgs = []
    for m in messages:
        mm = MagicMock()
        mm.content = m.get("content", "")
        type(mm).__name__ = m.get("type", "AIMessage")
        mock_msgs.append(mm)
    result = MagicMock()
    result.messages = mock_msgs
    result.success = success
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers tests
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractJsonObject:
    def test_simple_json(self):
        assert extract_json_object('{"key": "value"}') == '{"key": "value"}'

    def test_json_in_text(self):
        result = extract_json_object('Some text {"a": 1} more text')
        assert result == '{"a": 1}'

    def test_nested_json(self):
        result = extract_json_object('{"outer": {"inner": [1,2,3]}}')
        assert '"inner"' in result

    def test_no_json(self):
        assert extract_json_object("no json here") is None

    def test_unclosed_json(self):
        assert extract_json_object('{"a": 1') is None

    def test_multiple_json_objects_returns_first(self):
        result = extract_json_object('{"first": 1} {"second": 2}')
        assert result == '{"first": 1}'


class TestSafeRead:
    def test_reads_file(self, tmpdir):
        path = os.path.join(tmpdir, "test.txt")
        with open(path, "w") as f:
            f.write("hello")
        assert safe_read(path) == "hello"

    def test_returns_default_on_missing(self, tmpdir):
        assert safe_read(os.path.join(tmpdir, "nope.txt"), "fallback") == "fallback"

    def test_default_default_is_empty_string(self, tmpdir):
        assert safe_read(os.path.join(tmpdir, "nope.txt")) == ""


# ═══════════════════════════════════════════════════════════════════════════
# Scanner role tests
# ═══════════════════════════════════════════════════════════════════════════

class TestScannerRole:
    def test_agent_name(self):
        role = FeatureScannerRole()
        assert role.agent_name == "feature_scanner"

    def test_max_steps(self):
        role = FeatureScannerRole()
        assert role.max_steps == 15

    def test_tools_for_backend(self):
        role = FeatureScannerRole()
        tools = role.tools_for_backend("deepseek")
        assert len(tools) >= 2
        names = {t["name"] for t in tools}
        assert "read_file" in names
        assert "write_file" in names

    def test_build_task_deepseek(self):
        role = FeatureScannerRole()
        task = role.build_task("deepseek", working_dir="/tmp", project_dir="myproj")
        assert "project_context.json" in task
        assert "myproj" in task
        assert "conventions" in task

    def test_build_task_claude_cli(self):
        role = FeatureScannerRole()
        task = role.build_task("claude_cli", working_dir="/tmp", project_dir=".")
        assert "project_context.json" in task
        assert "TASK_COMPLETE" in task

    def test_parse_result_from_messages(self):
        role = FeatureScannerRole()
        mock = _make_agent_result([
            {"content": '{"file_manifest": [{"path": "a.py"}], "conventions": {"naming": {}}}',
             "type": "AIMessage"},
        ])
        result = role.parse_result(mock, working_dir="/tmp")
        assert len(result["file_manifest"]) == 1

    def test_parse_result_fallback(self, tmpdir):
        role = FeatureScannerRole()
        mock = _make_agent_result([])
        # Should use deterministic fallback
        result = role.parse_result(mock, working_dir=tmpdir, project_dir=".")
        assert "_fallback" in result
        assert "file_manifest" in result

    def test_fallback_scanner_produces_valid_output(self, tmpdir):
        # Create a minimal project
        os.makedirs(os.path.join(tmpdir, "src"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "tests"), exist_ok=True)
        with open(os.path.join(tmpdir, "src", "main.py"), "w") as f:
            f.write("def hello(): return 'world'\n")
        with open(os.path.join(tmpdir, "tests", "test_main.py"), "w") as f:
            f.write("def test_hello(): pass\n")

        scan = FeatureScannerRole._fallback_scanner(tmpdir, tmpdir)
        assert scan.get("_fallback") is True
        assert scan.get("language") == "python"
        assert len(scan.get("file_manifest", [])) >= 2
        assert "file_manifest" in scan
        assert "conventions" in scan


# ═══════════════════════════════════════════════════════════════════════════
# Planner role tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPlannerRole:
    def test_agent_name(self):
        role = FeaturePlannerRole()
        assert role.agent_name == "feature_planner"

    def test_max_steps(self):
        role = FeaturePlannerRole()
        assert role.max_steps == 12

    def test_tools_are_read_write(self):
        role = FeaturePlannerRole()
        tools = role.tools_for_backend("deepseek")
        names = {t["name"] for t in tools}
        assert "read_file" in names
        assert "write_file" in names

    def test_build_task_includes_files_to_modify(self):
        role = FeaturePlannerRole()
        task = role.build_task("deepseek", working_dir="/tmp",
                               feature_description="Add hello function")
        assert "files_to_modify" in task
        assert "files_to_create" in task
        assert "section" in task  # modification instructions
        assert "Add hello function" in task

    def test_parse_result_from_messages(self):
        role = FeaturePlannerRole()
        mock = _make_agent_result([
            {"content": '{"files_to_create": [{"path": "x.py"}], '
                        '"files_to_modify": [{"path": "y.py", "section": "end"}]}',
             "type": "AIMessage"},
        ])
        result = role.parse_result(mock, working_dir="/tmp")
        assert len(result["files_to_create"]) == 1
        assert len(result["files_to_modify"]) == 1

    def test_parse_result_reads_from_disk(self, tmpdir):
        plan = {"files_to_create": [{"path": "a.py"}], "files_to_modify": []}
        with open(os.path.join(tmpdir, "implementation_plan.json"), "w") as f:
            json.dump(plan, f)

        role = FeaturePlannerRole()
        mock = _make_agent_result([])
        result = role.parse_result(mock, working_dir=tmpdir)
        assert len(result["files_to_create"]) == 1

    def test_fallback_plan(self):
        plan = FeaturePlannerRole._fallback_plan("/tmp", "Add X")
        assert plan["_fallback"] is True
        assert "files_to_create" in plan
        assert "files_to_modify" in plan
        assert plan["feature"] == "Add X"


# ═══════════════════════════════════════════════════════════════════════════
# Coder role tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCoderRole:
    def test_agent_name(self):
        role = FeatureCoderRole()
        assert role.agent_name == "feature_coder"

    def test_max_steps(self):
        role = FeatureCoderRole()
        assert role.max_steps == 25

    def test_tools_include_all(self):
        role = FeatureCoderRole()
        tools = role.tools_for_backend("deepseek")
        names = {t["name"] for t in tools}
        assert "read_file" in names
        assert "write_file" in names
        assert "run_command" in names

    def test_build_task_references_context_and_plan(self, tmpdir):
        ctx = {"language": "python", "conventions": {"naming": {"functions": "snake_case"}}}
        plan = {"files_to_create": [{"path": "x.py"}], "files_to_modify": [{"path": "y.py"}]}
        with open(os.path.join(tmpdir, "project_context.json"), "w") as f:
            json.dump(ctx, f)
        with open(os.path.join(tmpdir, "implementation_plan.json"), "w") as f:
            json.dump(plan, f)

        role = FeatureCoderRole()
        task = role.build_task("deepseek", working_dir=tmpdir)
        assert "snake_case" in task
        assert "x.py" in task
        assert "files_to_modify" in task
        assert "Read" in task  # instructions to read existing files

    def test_build_task_with_review_issues(self, tmpdir):
        with open(os.path.join(tmpdir, "project_context.json"), "w") as f:
            json.dump({}, f)
        with open(os.path.join(tmpdir, "implementation_plan.json"), "w") as f:
            json.dump({}, f)

        role = FeatureCoderRole()
        task = role.build_task("deepseek", working_dir=tmpdir,
                               review_issues=["import x is missing", "tests fail"])
        assert "import x is missing" in task
        assert "FIX THESE" in task

    def test_parse_result_counts_changed_files(self, tmpdir):
        plan = {
            "files_to_create": [{"path": "src/new.py"}],
            "files_to_modify": [{"path": "src/old.py"}],
            "test_files": [{"path": "tests/test_new.py"}],
        }
        with open(os.path.join(tmpdir, "implementation_plan.json"), "w") as f:
            json.dump(plan, f)
        # Create the files on disk
        os.makedirs(os.path.join(tmpdir, "src"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "tests"), exist_ok=True)
        with open(os.path.join(tmpdir, "src", "new.py"), "w") as f:
            f.write("def foo(): pass")

        role = FeatureCoderRole()
        mock = _make_agent_result([])
        result = role.parse_result(mock, working_dir=tmpdir)
        assert len(result["changed_files"]) >= 1
        assert "src/new.py" in result["changed_files"]


# ═══════════════════════════════════════════════════════════════════════════
# Reviewer role tests
# ═══════════════════════════════════════════════════════════════════════════

class TestReviewerRole:
    def test_agent_name(self):
        role = FeatureReviewerRole()
        assert role.agent_name == "feature_reviewer"

    def test_max_steps(self):
        role = FeatureReviewerRole()
        assert role.max_steps == 10

    def test_tools_read_and_run(self):
        role = FeatureReviewerRole()
        tools = role.tools_for_backend("deepseek")
        names = {t["name"] for t in tools}
        assert "read_file" in names
        assert "run_command" in names
        assert "write_file" not in names

    def test_build_task_includes_review_dimensions(self, tmpdir):
        with open(os.path.join(tmpdir, "project_context.json"), "w") as f:
            json.dump({}, f)
        with open(os.path.join(tmpdir, "implementation_plan.json"), "w") as f:
            json.dump({}, f)
        role = FeatureReviewerRole()
        task = role.build_task("deepseek", working_dir=tmpdir,
                               changed_files=["a.py"],
                               feature_description="Add X")
        assert "Completeness" in task
        assert "Correctness" in task
        assert "Convention Compliance" in task
        assert "REVIEW_PASSED" in task
        assert "REVIEW_FAILED" in task

    def test_parse_result_review_passed(self):
        role = FeatureReviewerRole()
        mock = _make_agent_result([
            {"content": "Checks complete. REVIEW_PASSED", "type": "AIMessage"},
        ])
        result = role.parse_result(mock)
        assert result["review_passed"] is True
        assert result["review_issues"] == []

    def test_parse_result_review_failed(self):
        role = FeatureReviewerRole()
        mock = _make_agent_result([
            {"content": "Issues found.\n- import missing\n- tests fail\nREVIEW_FAILED",
             "type": "AIMessage"},
        ])
        result = role.parse_result(mock)
        assert result["review_passed"] is False
        assert len(result["review_issues"]) >= 2

    def test_parse_result_no_clear_verdict(self):
        role = FeatureReviewerRole()
        mock = _make_agent_result([
            {"content": "Just some text without the token", "type": "AIMessage"},
        ])
        result = role.parse_result(mock)
        assert result["review_passed"] is False


# ═══════════════════════════════════════════════════════════════════════════
# Writer role tests
# ═══════════════════════════════════════════════════════════════════════════

class TestWriterRole:
    def test_agent_name(self):
        role = FeatureReportWriterRole()
        assert role.agent_name == "feature_report_writer"

    def test_max_steps(self):
        role = FeatureReportWriterRole()
        assert role.max_steps == 5

    def test_tools_write_only(self):
        role = FeatureReportWriterRole()
        tools = role.tools_for_backend("deepseek")
        names = {t["name"] for t in tools}
        assert "write_file" in names
        assert "read_file" not in names

    def test_build_task_includes_report_structure(self):
        role = FeatureReportWriterRole()
        task = role.build_task("deepseek", working_dir="/tmp",
                               changed_files=["a.py"],
                               review_passed=True,
                               feature_description="Add X")
        assert "Summary" in task
        assert "feature_report.md" in task

    def test_parse_result_reads_disk(self, tmpdir):
        with open(os.path.join(tmpdir, "feature_report.md"), "w") as f:
            f.write("# Report")
        role = FeatureReportWriterRole()
        mock = _make_agent_result([])
        result = role.parse_result(mock, working_dir=tmpdir)
        assert "feature_report" in result

    def test_fallback_report_writes_file(self, tmpdir):
        result = FeatureReportWriterRole._fallback_report(
            tmpdir, changed_files=["a.py"], review_passed=True,
            feature_description="Test feature",
        )
        assert os.path.isfile(result["feature_report"])
        content = safe_read(result["feature_report"])
        assert "Test feature" in content


# ═══════════════════════════════════════════════════════════════════════════
# FeatureState tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFeatureState:
    def test_all_required_fields(self):
        """FeatureState should have the 12 essential fields."""
        fields = FeatureState.__annotations__
        required = [
            "input_spec", "working_dir", "backend", "status",
            "iteration", "project_context", "implementation_plan",
            "changed_files", "review_passed", "review_issues", "feature_report",
        ]
        for field in required:
            assert field in fields, f"Missing field: {field}"

    def test_is_typed_dict(self):
        assert issubclass(FeatureState, dict) or hasattr(FeatureState, "__annotations__")


# ═══════════════════════════════════════════════════════════════════════════
# FeaturePipeline integration tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFeaturePipeline:
    def test_pipeline_name(self):
        p = FeaturePipeline(working_dir="/tmp/test")
        assert p.name == "feature"

    def test_default_output_dir(self):
        assert FeaturePipeline.default_output_dir == "feature_output"

    def test_build_initial_state(self):
        p = FeaturePipeline(working_dir="/tmp/test", backend="claude_cli")
        state = p._build_initial_state("Add feature X", [])
        assert state["input_spec"] == "Add feature X"
        assert state["working_dir"] == "/tmp/test"
        assert state["backend"] == "claude_cli"
        assert state["status"] == "initialized"
        assert state["iteration"] == 0
        assert state["review_passed"] is False
        assert state["changed_files"] == []

    def test_build_graph_compiles(self):
        p = FeaturePipeline(working_dir="/tmp/test")
        graph = p._build_graph()
        assert graph is not None
        # Graph should have nodes and entry point
        assert hasattr(graph, "nodes") or hasattr(graph, "builder")

    def test_decompose_returns_input(self):
        p = FeaturePipeline()
        result = p._decompose("Test feature")
        assert len(result) == 1
        assert result[0]["description"] == "Test feature"

    def test_manage_output_dir(self, tmpdir):
        p = FeaturePipeline(working_dir=tmpdir)
        p.manage_output_dir()
        assert os.path.isdir(tmpdir)

    def test_run_with_mock(self, tmpdir):
        """End-to-end test with mocked agent execution."""
        import feature.scanner as fscan
        import feature.planner as fplan
        import feature.coder as fcode
        import feature.reviewer as frev
        import feature.writer as fwrt

        wd = tmpdir
        # Pre-create context and plan so fallbacks are not triggered
        ctx = {
            "language": "python", "total_files": 3,
            "conventions": {"naming": {"functions": "snake_case"}},
            "test_patterns": {"framework": "pytest"},
            "file_manifest": [{"path": "main.py", "role": "entry_point"}],
        }
        with open(os.path.join(wd, "project_context.json"), "w") as f:
            json.dump(ctx, f)

        plan = {
            "feature": "Add hello",
            "files_to_create": [{"path": "hello.py", "description": "Hello module"}],
            "files_to_modify": [],
            "test_files": [],
        }
        with open(os.path.join(wd, "implementation_plan.json"), "w") as f:
            json.dump(plan, f)

        # Mock all role execute methods
        with patch.object(fscan.FeatureScannerRole, "execute",
                          return_value=ctx) as mock_scan, \
             patch.object(fplan.FeaturePlannerRole, "execute",
                          return_value=plan) as mock_plan, \
             patch.object(fcode.FeatureCoderRole, "execute",
                          return_value={"changed_files": ["hello.py"], "success": True}) as mock_coder, \
             patch.object(frev.FeatureReviewerRole, "execute",
                          return_value={"review_passed": True, "review_issues": []}) as mock_review, \
             patch.object(fwrt.FeatureReportWriterRole, "execute",
                          return_value={"feature_report": os.path.join(wd, "feature_report.md")}) as mock_writer:

            p = FeaturePipeline(working_dir=wd, backend="deepseek", yes=True)
            p.run("Add hello function")

        assert mock_scan.called
        assert mock_plan.called
        assert mock_coder.called
        assert mock_review.called
        assert mock_writer.called


# ═══════════════════════════════════════════════════════════════════════════
# FeaturePipeline import from main
# ═══════════════════════════════════════════════════════════════════════════

class TestFeaturePipelineImport:
    def test_import_from_feature_pipeline(self):
        from pipeline import FeaturePipeline, FeatureState
        assert FeaturePipeline.name == "feature"
        assert FeatureState is not None

    def test_feature_in_pipelines_dict(self):
        """Feature key is registered in PIPELINES in main.py."""
        import main
        assert "feature" in main.PIPELINES
        assert main.PIPELINES["feature"].name == "feature"

    def test_main_imports_feature_pipeline(self):
        """main.py imports FeaturePipeline from feature_pipeline."""
        import main
        cls = main.PIPELINES["feature"]
        # Should be the FeaturePipeline class from feature_pipeline
        assert cls.__module__ == "pipeline.feature"
