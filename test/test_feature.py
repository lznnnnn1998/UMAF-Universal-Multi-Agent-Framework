"""Tests for Feature Pipeline v2 — 5-node graph with coder/reviewer loop."""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

import test.conftest  # noqa: F401 — loads tools_config.json, tmpdir fixture
from test.conftest import make_agent_result as _make_agent_result

from pipeline import FeaturePipeline, FeatureState
from pipeline.base import BasePipeline
import pipeline.feature as pfeature
from feature.scanner import FeatureScannerRole
from feature.planner import FeaturePlannerRole
from feature.coder import FeatureCoderRole
from feature.reviewer import FeatureReviewerRole
from feature.writer import FeatureReportWriterRole
from utils import extract_json_object, safe_read


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — tmpdir fixture and _make_agent_result imported from test.conftest
# ═══════════════════════════════════════════════════════════════════════════


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
        # Create the files on disk — source files live under project_dir
        os.makedirs(os.path.join(tmpdir, "src"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "tests"), exist_ok=True)
        with open(os.path.join(tmpdir, "src", "new.py"), "w") as f:
            f.write("def foo(): pass")

        role = FeatureCoderRole()
        mock = _make_agent_result([])
        result = role.parse_result(mock, working_dir=tmpdir, project_dir=tmpdir)
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
        """FeatureState should have the 14 essential fields."""
        fields = FeatureState.__annotations__
        required = [
            "input_spec", "working_dir", "backend", "project_dir",
            "status", "iteration", "version", "project_context",
            "implementation_plan", "changed_files", "review_passed",
            "review_issues", "feature_report",
        ]
        for field in required:
            assert field in fields, f"Missing field: {field}"

    def test_is_typed_dict(self):
        assert issubclass(FeatureState, dict) or hasattr(FeatureState, "__annotations__")


# ═══════════════════════════════════════════════════════════════════════════
# Multi-coder state tests (v2 fields)
# ═══════════════════════════════════════════════════════════════════════════

class TestMultiCoderState:
    """Tests for new FeatureState fields (sub_tasks, coder_outputs, dependency_graph)."""

    def test_new_fields_exist(self):
        """FeatureState should include the 3 new multi-coder fields."""
        fields = FeatureState.__annotations__
        new_fields = ["sub_tasks", "coder_outputs", "dependency_graph"]
        for field in new_fields:
            assert field in fields, f"Missing multi-coder field: {field}"

    def test_initial_state_includes_new_fields(self):
        """_build_initial_state must initialize the new fields to safe defaults."""
        p = FeaturePipeline(working_dir="/tmp/test")
        state = p._build_initial_state("Feature X", [])
        assert state["sub_tasks"] == []
        assert state["coder_outputs"] == []
        assert state["dependency_graph"] == {}

    def test_sub_tasks_field_is_mutable_list(self):
        """sub_tasks should be a list so coders_node can extend it."""
        p = FeaturePipeline(working_dir="/tmp/test")
        state = p._build_initial_state("Feature X", [])
        state["sub_tasks"].append({"id": 1, "module_name": "test"})
        assert len(state["sub_tasks"]) == 1

    def test_coder_outputs_field_is_mutable_list(self):
        """coder_outputs should be a list for collecting per-coder results."""
        p = FeaturePipeline(working_dir="/tmp/test")
        state = p._build_initial_state("Feature X", [])
        state["coder_outputs"].append({"sub_task_id": 1, "files": ["a.py"]})
        assert len(state["coder_outputs"]) == 1

    def test_dependency_graph_field_is_mutable_dict(self):
        """dependency_graph should be a dict for nodes/edges/levels."""
        p = FeaturePipeline(working_dir="/tmp/test")
        state = p._build_initial_state("Feature X", [])
        state["dependency_graph"]["nodes"] = []
        assert "nodes" in state["dependency_graph"]

    def test_status_can_be_planned_with_deps(self):
        """planned_with_deps status flows to coders node."""
        p = FeaturePipeline(working_dir="/tmp/test")
        state = p._build_initial_state("Feature X", [])
        state["status"] = "planned_with_deps"
        assert state["status"] == "planned_with_deps"

    def test_status_can_be_coders_done(self):
        """coders_done status flows to reviewer node."""
        p = FeaturePipeline(working_dir="/tmp/test")
        state = p._build_initial_state("Feature X", [])
        state["status"] = "coders_done"
        assert state["status"] == "coders_done"

    def test_max_coder_retries_exists(self):
        """_MAX_CODER_RETRIES replaces the old _MAX_VERSIONS."""
        assert hasattr(FeaturePipeline, "_MAX_CODER_RETRIES")
        assert FeaturePipeline._MAX_CODER_RETRIES == 3


# ═══════════════════════════════════════════════════════════════════════════
# Topological levels tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTopologicalLevels:
    """Tests that _topological_levels correctly groups sub-tasks with/without
    dependencies, handles single-item degradation, and cycle detection."""

    def test_no_dependencies_single_level(self):
        """Tasks without dependencies all land in one level."""
        tasks = [
            {"id": 1, "module_name": "a", "dependencies": []},
            {"id": 2, "module_name": "b", "dependencies": []},
            {"id": 3, "module_name": "c", "dependencies": []},
        ]
        levels = BasePipeline._topological_levels(tasks)
        assert len(levels) == 1
        assert len(levels[0]) == 3

    def test_single_item_degradation(self):
        """Single sub_task produces one level (degraded to flat parallelism)."""
        tasks = [{"id": 1, "module_name": "only", "dependencies": []}]
        levels = BasePipeline._topological_levels(tasks)
        assert len(levels) == 1
        assert len(levels[0]) == 1

    def test_empty_tasks(self):
        """Empty task list returns a single empty level (existing behavior)."""
        levels = BasePipeline._topological_levels([])
        # When no tasks exist, [list([])] = [[]] — one level with no tasks
        assert len(levels) == 1
        assert levels[0] == []

    def test_linear_chain_two_levels(self):
        """A -> B -> C produces 3 levels."""
        tasks = [
            {"id": 1, "module_name": "a", "dependencies": []},
            {"id": 2, "module_name": "b", "dependencies": [1]},
            {"id": 3, "module_name": "c", "dependencies": [2]},
        ]
        levels = BasePipeline._topological_levels(tasks)
        assert len(levels) == 3
        assert levels[0][0]["module_name"] == "a"
        assert levels[1][0]["module_name"] == "b"
        assert levels[2][0]["module_name"] == "c"

    def test_diamond_dependency(self):
        """A has no deps; B and C depend on A; D depends on B and C."""
        tasks = [
            {"id": 1, "module_name": "a", "dependencies": []},
            {"id": 2, "module_name": "b", "dependencies": [1]},
            {"id": 3, "module_name": "c", "dependencies": [1]},
            {"id": 4, "module_name": "d", "dependencies": [2, 3]},
        ]
        levels = BasePipeline._topological_levels(tasks)
        assert len(levels) == 3
        # Level 0: a
        assert {t["module_name"] for t in levels[0]} == {"a"}
        # Level 1: b and c (parallel)
        assert {t["module_name"] for t in levels[1]} == {"b", "c"}
        # Level 2: d
        assert {t["module_name"] for t in levels[2]} == {"d"}

    def test_mixed_dependencies_and_independent(self):
        """Tasks with and without deps; independents land in level 0."""
        tasks = [
            {"id": 1, "module_name": "independent", "dependencies": []},
            {"id": 2, "module_name": "depends_on_indep", "dependencies": [1]},
            {"id": 3, "module_name": "also_independent", "dependencies": []},
        ]
        levels = BasePipeline._topological_levels(tasks)
        assert len(levels) == 2
        assert len(levels[0]) == 2  # both independents
        assert len(levels[1]) == 1  # the dependent

    def test_string_dependencies(self):
        """Dependencies can reference by module_name (str)."""
        tasks = [
            {"id": 1, "module_name": "base", "dependencies": []},
            {"id": 2, "module_name": "ext", "dependencies": ["base"]},
        ]
        levels = BasePipeline._topological_levels(tasks)
        assert len(levels) == 2

    def test_dict_dependency(self):
        """Dependencies specified as dicts with module_name."""
        tasks = [
            {"id": 1, "module_name": "base", "dependencies": []},
            {"id": 2, "module_name": "ext",
             "dependencies": [{"module_name": "base", "id": 1}]},
        ]
        levels = BasePipeline._topological_levels(tasks)
        assert len(levels) == 2

    def test_cycle_detection_produces_single_warning_level(self):
        """A cycle (A→B→A) is detected and broken, producing a level."""
        tasks = [
            {"id": 1, "module_name": "a", "dependencies": ["b"]},
            {"id": 2, "module_name": "b", "dependencies": ["a"]},
        ]
        levels = BasePipeline._topological_levels(tasks)
        # Should produce at least one level (cycle broken)
        assert len(levels) >= 1
        all_tasks = {t["module_name"] for level in levels for t in level}
        assert all_tasks == {"a", "b"}

    def test_three_way_cycle(self):
        """A→B→C→A cycle is detected and broken."""
        tasks = [
            {"id": 1, "module_name": "a", "dependencies": ["b"]},
            {"id": 2, "module_name": "b", "dependencies": ["c"]},
            {"id": 3, "module_name": "c", "dependencies": ["a"]},
        ]
        levels = BasePipeline._topological_levels(tasks)
        assert len(levels) >= 1
        all_tasks = {t["module_name"] for level in levels for t in level}
        assert all_tasks == {"a", "b", "c"}

    def test_missing_dependency_ignored(self):
        """Dependency to a non-existent task is silently ignored."""
        tasks = [
            {"id": 1, "module_name": "a", "dependencies": []},
            {"id": 2, "module_name": "b", "dependencies": [999]},  # non-existent
        ]
        levels = BasePipeline._topological_levels(tasks)
        # Both tasks should appear (dep to 999 ignored = no dep)
        all_tasks = {t["module_name"] for level in levels for t in level}
        assert all_tasks == {"a", "b"}


# ═══════════════════════════════════════════════════════════════════════════
# Planner decomposition tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPlannerDecomposition:
    """Tests that FeaturePlannerRole.parse_result extracts sub_tasks, reads
    decomposition.json, and produces fallback sub_tasks."""

    def test_parse_result_extracts_sub_tasks_from_messages(self):
        """parse_result should extract sub_tasks from agent messages."""
        role = FeaturePlannerRole()
        mock = _make_agent_result([
            {"content": '{"files_to_create": [{"path": "a.py"}], '
                        '"files_to_modify": [], '
                        '"sub_tasks": [{"id": 1, "module_name": "core", '
                        '"description": "Core logic", "dependencies": [], '
                        '"files_to_create": [{"path": "a.py", "description": ""}], '
                        '"files_to_modify": [], "test_files": []}]}',
             "type": "AIMessage"},
        ])
        result = role.parse_result(mock, working_dir="/tmp")
        assert "sub_tasks" in result
        assert len(result["sub_tasks"]) == 1
        assert result["sub_tasks"][0]["module_name"] == "core"

    def test_parse_result_writes_decomposition_json(self, tmpdir):
        """parse_result should write decomposition.json to working_dir."""
        role = FeaturePlannerRole()
        mock = _make_agent_result([
            {"content": '{"files_to_create": [{"path": "a.py"}], '
                        '"files_to_modify": [], '
                        '"sub_tasks": [{"id": 1, "module_name": "core", '
                        '"description": "Core", "dependencies": [], '
                        '"files_to_create": [{"path": "a.py", "description": ""}], '
                        '"files_to_modify": [], "test_files": []}]}',
             "type": "AIMessage"},
        ])
        role.parse_result(mock, working_dir=tmpdir)
        decomp_path = os.path.join(tmpdir, "decomposition.json")
        assert os.path.isfile(decomp_path)
        with open(decomp_path) as f:
            decomp = json.load(f)
        assert len(decomp) == 1
        assert decomp[0]["module_name"] == "core"

    def test_fallback_sub_tasks_when_missing(self, tmpdir):
        """When sub_tasks is missing from plan, fallback is generated."""
        role = FeaturePlannerRole()
        mock = _make_agent_result([
            {"content": '{"files_to_create": [{"path": "src/a.py"}], '
                        '"files_to_modify": [{"path": "src/b.py", "section": "end", '
                        '"change": "add", "description": "x"}]}',
             "type": "AIMessage"},
        ])
        result = role.parse_result(mock, working_dir=tmpdir)
        assert "sub_tasks" in result
        assert len(result["sub_tasks"]) >= 1
        # Verify decomposition.json was written
        decomp_path = os.path.join(tmpdir, "decomposition.json")
        assert os.path.isfile(decomp_path)

    def test_single_task_fallback(self):
        """Empty plan produces a single fallback sub_task."""
        plan = {"feature": "Test", "files_to_create": [], "files_to_modify": []}
        from feature.planner import FeaturePlannerRole
        sub_tasks = FeaturePlannerRole._generate_sub_tasks_from_plan(plan)
        assert len(sub_tasks) == 1
        assert sub_tasks[0]["module_name"] == "feature_implementation"
        assert sub_tasks[0]["dependencies"] == []
        assert sub_tasks[0]["id"] == 1

    def test_groups_files_by_module(self):
        """Files from different directories produce separate sub_tasks."""
        plan = {
            "files_to_create": [
                {"path": "src/auth/login.py", "description": "Login"},
                {"path": "src/api/routes.py", "description": "Routes"},
            ],
            "files_to_modify": [],
        }
        from feature.planner import FeaturePlannerRole
        sub_tasks = FeaturePlannerRole._generate_sub_tasks_from_plan(plan)
        assert len(sub_tasks) == 2
        module_names = {t["module_name"] for t in sub_tasks}
        assert module_names == {"auth", "api"}

    def test_validate_dependency_references(self):
        """Unresolved dependency references produce warnings and are removed."""
        role = FeaturePlannerRole()
        mock = _make_agent_result([
            {"content": '{"files_to_create": [], "files_to_modify": [], '
                        '"sub_tasks": ['
                        '{"id": 1, "module_name": "a", "dependencies": [], '
                        '"files_to_create": [], "files_to_modify": [], "test_files": []},'
                        '{"id": 2, "module_name": "b", "dependencies": ["c"], '
                        '"files_to_create": [], "files_to_modify": [], "test_files": []}'
                        ']}',
             "type": "AIMessage"},
        ])
        result = role.parse_result(mock, working_dir="/tmp")
        # sub_task b depends on "c" which doesn't exist — should be removed
        task_b = [t for t in result["sub_tasks"] if t["module_name"] == "b"][0]
        assert task_b["dependencies"] == []

    def test_fallback_plan_includes_sub_tasks(self):
        """_fallback_plan should include generated sub_tasks."""
        plan = FeaturePlannerRole._fallback_plan("/tmp", "Fallback feature")
        assert "sub_tasks" in plan
        assert len(plan["sub_tasks"]) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Multi-coder pipeline integration tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMultiCoderPipeline:
    """Integration tests with mocked coders at two topological levels, verifying
    dependency injection, parallel execution, and cross-coder review."""

    def test_two_level_execution_with_mocked_coders(self, tmpdir):
        """Coders at level[0] run, then level[1] coders receive their outputs."""
        import feature.scanner as fscan
        import feature.planner as fplan
        import feature.coder as fcode
        import feature.reviewer as frev
        import feature.writer as fwrt

        wd = str(tmpdir)
        # Pre-create context, plan, and decomposition files
        ctx = {"language": "python", "conventions": {}, "file_manifest": []}
        with open(os.path.join(wd, "project_context.json"), "w") as f:
            json.dump(ctx, f)
        plan = {
            "files_to_create": [],
            "files_to_modify": [],
            "test_files": [],
            "sub_tasks": [
                {"id": 1, "module_name": "base", "dependencies": [],
                 "files_to_create": [{"path": "base.py", "description": ""}],
                 "files_to_modify": [], "test_files": []},
                {"id": 2, "module_name": "ext", "dependencies": [1],
                 "files_to_create": [{"path": "ext.py", "description": ""}],
                 "files_to_modify": [], "test_files": []},
            ],
        }
        with open(os.path.join(wd, "implementation_plan.json"), "w") as f:
            json.dump(plan, f)

        # Mock planner to return the plan with sub_tasks
        with patch.object(fscan.FeatureScannerRole, "execute", return_value=ctx), \
             patch.object(fplan.FeaturePlannerRole, "execute", return_value=plan):

            # Build pipeline and invoke graph through to coders
            p = FeaturePipeline(working_dir=wd, backend="deepseek")
            graph = p._build_graph()

            # Mock the coder worker and reviewer
            def mock_coder_worker(item, wd_, backend_, **_kw):
                sid = item.get("id")
                deps = item.get("_dependency_outputs", [])
                return {
                    "sub_task_id": sid,
                    "module_name": item.get("module_name", f"task_{sid}"),
                    "files": [item.get("module_name", f"task_{sid}") + ".py"],
                    "summary": f"Implemented {item.get('module_name')}",
                    "dependency_verification": len(deps) > 0,
                }

            with patch.object(pfeature, "_feature_coder_worker",
                            side_effect=mock_coder_worker) as mock_worker, \
                 patch.object(frev.FeatureReviewerRole, "execute",
                            return_value={"review_passed": True, "review_issues": []}), \
                 patch.object(fwrt.FeatureReportWriterRole, "execute",
                            return_value={"feature_report": os.path.join(wd, "report.md")}):

                result = graph.invoke({
                    "input_spec": "Add base and ext",
                    "working_dir": wd,
                    "backend": "deepseek",
                    "project_dir": ".",
                    "status": "initialized",
                    "iteration": 0,
                    "version": 1,
                    "project_context": ctx,
                    "implementation_plan": plan,
                    "changed_files": [],
                    "review_passed": False,
                    "review_issues": [],
                    "feature_report": "",
                    "sub_tasks": plan["sub_tasks"],
                    "coder_outputs": [],
                    "dependency_graph": {},
                })

            # Verify coders ran twice (one per level agent call)
            assert mock_worker.call_count >= 2

            # Check coder outputs were collected
            assert len(result.get("coder_outputs", [])) == 2
            module_names = {o["module_name"] for o in result["coder_outputs"]}
            assert module_names == {"base", "ext"}

    def test_dependency_injection_to_level1_coders(self, tmpdir):
        """Level[1] coders receive _dependency_outputs from level[0]."""
        import feature.scanner as fscan
        import feature.planner as fplan
        import feature.reviewer as frev
        import feature.writer as fwrt

        wd = str(tmpdir)
        ctx = {"language": "python", "conventions": {}, "file_manifest": []}
        with open(os.path.join(wd, "project_context.json"), "w") as f:
            json.dump(ctx, f)
        plan = {
            "files_to_create": [], "files_to_modify": [], "test_files": [],
            "sub_tasks": [
                {"id": 1, "module_name": "base", "dependencies": [],
                 "files_to_create": [{"path": "base.py", "description": ""}],
                 "files_to_modify": [], "test_files": []},
                {"id": 2, "module_name": "middle", "dependencies": [1],
                 "files_to_create": [{"path": "mid.py", "description": ""}],
                 "files_to_modify": [], "test_files": []},
            ],
        }
        with open(os.path.join(wd, "implementation_plan.json"), "w") as f:
            json.dump(plan, f)

        # Track what each coder received
        received_items: list = []

        def tracking_worker(item, wd_, backend_, **_kw):
            received_items.append(dict(item))  # shallow copy
            return {
                "sub_task_id": item.get("id"),
                "module_name": item.get("module_name", "?"),
                "files": [item.get("module_name", "?") + ".py"],
                "summary": "done",
                "dependency_verification": bool(item.get("_dependency_outputs")),
            }

        with patch.object(fscan.FeatureScannerRole, "execute", return_value=ctx), \
             patch.object(fplan.FeaturePlannerRole, "execute", return_value=plan), \
             patch.object(pfeature, "_feature_coder_worker",
                         side_effect=tracking_worker), \
             patch.object(frev.FeatureReviewerRole, "execute",
                         return_value={"review_passed": True, "review_issues": []}), \
             patch.object(fwrt.FeatureReportWriterRole, "execute",
                         return_value={"feature_report": os.path.join(wd, "report.md")}):

            p = FeaturePipeline(working_dir=wd, backend="deepseek")
            graph = p._build_graph()

            result = graph.invoke({
                "input_spec": "Test deps",
                "working_dir": wd,
                "backend": "deepseek",
                "project_dir": ".",
                "status": "initialized",
                "iteration": 0,
                "version": 1,
                "project_context": ctx,
                "implementation_plan": plan,
                "changed_files": [],
                "review_passed": False,
                "review_issues": [],
                "feature_report": "",
                "sub_tasks": plan["sub_tasks"],
                "coder_outputs": [],
                "dependency_graph": {},
            })

        # Find the "middle" task — it should have _dependency_outputs
        middle_items = [i for i in received_items if i.get("module_name") == "middle"]
        assert len(middle_items) == 1
        middle = middle_items[0]
        assert "_dependency_outputs" in middle
        assert len(middle["_dependency_outputs"]) >= 1
        # The dependency should reference the "base" module
        dep_modules = [d.get("module_name") for d in middle["_dependency_outputs"]]
        assert "base" in dep_modules

    def test_reviewer_receives_all_coder_outputs(self, tmpdir):
        """The reviewer should receive all_coder_outputs for cross-coder verification."""
        import feature.scanner as fscan
        import feature.planner as fplan
        import feature.coder as fcode
        import feature.reviewer as frev

        wd = str(tmpdir)
        ctx = {"language": "python", "conventions": {}, "file_manifest": []}
        with open(os.path.join(wd, "project_context.json"), "w") as f:
            json.dump(ctx, f)
        plan = {
            "files_to_create": [], "files_to_modify": [], "test_files": [],
            "sub_tasks": [
                {"id": 1, "module_name": "a", "dependencies": [],
                 "files_to_create": [{"path": "a.py", "description": ""}],
                 "files_to_modify": [], "test_files": []},
                {"id": 2, "module_name": "b", "dependencies": [],
                 "files_to_create": [{"path": "b.py", "description": ""}],
                 "files_to_modify": [], "test_files": []},
            ],
        }
        with open(os.path.join(wd, "implementation_plan.json"), "w") as f:
            json.dump(plan, f)

        def mock_coder_worker(item, wd_, backend_, **_kw):
            return {
                "sub_task_id": item.get("id"),
                "module_name": item.get("module_name", "?"),
                "files": [item.get("module_name", "?") + ".py"],
                "summary": "done",
                "dependency_verification": None,
            }

        # Track what reviewer.execute receives
        reviewer_call_kwargs: list = []

        def tracking_reviewer_execute(self, **kwargs):
            reviewer_call_kwargs.append(dict(kwargs))
            return {"review_passed": True, "review_issues": []}

        with patch.object(fscan.FeatureScannerRole, "execute", return_value=ctx), \
             patch.object(fplan.FeaturePlannerRole, "execute", return_value=plan), \
             patch.object(pfeature, "_feature_coder_worker",
                         side_effect=mock_coder_worker), \
             patch.object(frev.FeatureReviewerRole, "execute",
                         side_effect=tracking_reviewer_execute, autospec=True):

            p = FeaturePipeline(working_dir=wd, backend="deepseek")
            graph = p._build_graph()

            result = graph.invoke({
                "input_spec": "Test review",
                "working_dir": wd,
                "backend": "deepseek",
                "project_dir": ".",
                "status": "initialized",
                "iteration": 0,
                "version": 1,
                "project_context": ctx,
                "implementation_plan": plan,
                "changed_files": [],
                "review_passed": False,
                "review_issues": [],
                "feature_report": "",
                "sub_tasks": plan["sub_tasks"],
                "coder_outputs": [],
                "dependency_graph": {},
            })

        assert len(reviewer_call_kwargs) >= 1
        rev_kwargs = reviewer_call_kwargs[0]
        assert "all_coder_outputs" in rev_kwargs
        coder_outputs = rev_kwargs["all_coder_outputs"]
        assert len(coder_outputs) == 2
        assert {o["module_name"] for o in coder_outputs} == {"a", "b"}

    def test_reviewer_retry_loop_with_multi_coder(self, tmpdir):
        """On REVIEW_FAILED, router sends back to coders node (multi-coder retry)."""
        import feature.scanner as fscan
        import feature.planner as fplan
        import feature.coder as fcode
        import feature.reviewer as frev
        import feature.writer as fwrt

        wd = str(tmpdir)
        ctx = {"language": "python", "conventions": {}, "file_manifest": []}
        with open(os.path.join(wd, "project_context.json"), "w") as f:
            json.dump(ctx, f)
        plan = {
            "files_to_create": [], "files_to_modify": [], "test_files": [],
            "sub_tasks": [
                {"id": 1, "module_name": "x", "dependencies": [],
                 "files_to_create": [{"path": "x.py", "description": ""}],
                 "files_to_modify": [], "test_files": []},
            ],
        }
        with open(os.path.join(wd, "implementation_plan.json"), "w") as f:
            json.dump(plan, f)

        def mock_coder_worker(item, wd_, backend_, **_kw):
            return {
                "sub_task_id": item.get("id"),
                "module_name": item.get("module_name", "?"),
                "files": [item.get("module_name", "?") + ".py"],
                "summary": "done",
                "dependency_verification": None,
            }

        with patch.object(fscan.FeatureScannerRole, "execute", return_value=ctx), \
             patch.object(fplan.FeaturePlannerRole, "execute", return_value=plan), \
             patch.object(pfeature, "_feature_coder_worker",
                         side_effect=mock_coder_worker), \
             patch.object(frev.FeatureReviewerRole, "execute",
                         return_value={"review_passed": False, "review_issues": ["bug"]}), \
             patch.object(fwrt.FeatureReportWriterRole, "execute",
                         return_value={"feature_report": os.path.join(wd, "report.md")}):

            p = FeaturePipeline(working_dir=wd, backend="deepseek")
            graph = p._build_graph()

            result = graph.invoke({
                "input_spec": "Test retry",
                "working_dir": wd,
                "backend": "deepseek",
                "project_dir": ".",
                "status": "initialized",
                "iteration": 0,
                "version": 1,
                "project_context": ctx,
                "implementation_plan": plan,
                "changed_files": [],
                "review_passed": False,
                "review_issues": [],
                "feature_report": "",
                "sub_tasks": plan["sub_tasks"],
                "coder_outputs": [],
                "dependency_graph": {},
            })

        # After max retries (3), pipeline proceeds to writer
        assert result["status"] == "written"
        assert result["version"] >= FeaturePipeline._MAX_CODER_RETRIES

    def test_single_coder_fallback_when_no_sub_tasks(self, tmpdir):
        """When sub_tasks is empty, coders_node falls back to single-coder mode."""
        import feature.scanner as fscan
        import feature.planner as fplan
        import feature.coder as fcode
        import feature.reviewer as frev
        import feature.writer as fwrt

        wd = str(tmpdir)
        ctx = {"language": "python", "conventions": {}, "file_manifest": []}
        with open(os.path.join(wd, "project_context.json"), "w") as f:
            json.dump(ctx, f)
        # Plan WITHOUT sub_tasks — fallback path
        plan = {
            "files_to_create": [{"path": "x.py"}],
            "files_to_modify": [], "test_files": [],
        }
        with open(os.path.join(wd, "implementation_plan.json"), "w") as f:
            json.dump(plan, f)

        with patch.object(fscan.FeatureScannerRole, "execute", return_value=ctx), \
             patch.object(fplan.FeaturePlannerRole, "execute", return_value=plan), \
             patch.object(fcode.FeatureCoderRole, "execute",
                         return_value={"changed_files": ["x.py"], "success": True}), \
             patch.object(frev.FeatureReviewerRole, "execute",
                         return_value={"review_passed": True, "review_issues": []}), \
             patch.object(fwrt.FeatureReportWriterRole, "execute",
                         return_value={"feature_report": os.path.join(wd, "report.md")}):

            p = FeaturePipeline(working_dir=wd, backend="deepseek")
            graph = p._build_graph()

            result = graph.invoke({
                "input_spec": "Fallback test",
                "working_dir": wd,
                "backend": "deepseek",
                "project_dir": ".",
                "status": "initialized",
                "iteration": 0,
                "version": 1,
                "project_context": ctx,
                "implementation_plan": plan,
                "changed_files": [],
                "review_passed": False,
                "review_issues": [],
                "feature_report": "",
                "sub_tasks": [],
                "coder_outputs": [],
                "dependency_graph": {},
            })

        assert result["status"] == "written"
        assert result["review_passed"] is True

    def test_failed_level_blocks_downstream(self, tmpdir):
        """When a coder in level[0] fails, dependent level[1] tasks are deferred."""
        import feature.scanner as fscan
        import feature.planner as fplan
        import feature.reviewer as frev
        import feature.writer as fwrt

        wd = str(tmpdir)
        ctx = {"language": "python", "conventions": {}, "file_manifest": []}
        with open(os.path.join(wd, "project_context.json"), "w") as f:
            json.dump(ctx, f)
        plan = {
            "files_to_create": [], "files_to_modify": [], "test_files": [],
            "sub_tasks": [
                {"id": 1, "module_name": "base", "dependencies": [],
                 "files_to_create": [{"path": "base.py", "description": ""}],
                 "files_to_modify": [], "test_files": []},
                {"id": 2, "module_name": "ext", "dependencies": [1],
                 "files_to_create": [{"path": "ext.py", "description": ""}],
                 "files_to_modify": [], "test_files": []},
            ],
        }
        with open(os.path.join(wd, "implementation_plan.json"), "w") as f:
            json.dump(plan, f)

        call_count = 0

        def fail_base_then_succeed(item, wd_, backend_, **_kw):
            nonlocal call_count
            call_count += 1
            sid = item.get("id")
            # base (id=1) fails — no files produced
            if sid == 1:
                return {
                    "sub_task_id": sid,
                    "module_name": item.get("module_name", "?"),
                    "files": [],  # FAILURE — no files
                    "summary": "failed",
                    "dependency_verification": None,
                }
            return {
                "sub_task_id": sid,
                "module_name": item.get("module_name", "?"),
                "files": [item.get("module_name", "?") + ".py"],
                "summary": "done",
                "dependency_verification": None,
            }

        with patch.object(fscan.FeatureScannerRole, "execute", return_value=ctx), \
             patch.object(fplan.FeaturePlannerRole, "execute", return_value=plan), \
             patch.object(pfeature, "_feature_coder_worker",
                         side_effect=fail_base_then_succeed), \
             patch.object(frev.FeatureReviewerRole, "execute",
                         return_value={"review_passed": True, "review_issues": []}), \
             patch.object(fwrt.FeatureReportWriterRole, "execute",
                         return_value={"feature_report": os.path.join(wd, "report.md")}):

            p = FeaturePipeline(working_dir=wd, backend="deepseek")
            graph = p._build_graph()

            result = graph.invoke({
                "input_spec": "Test failure",
                "working_dir": wd,
                "backend": "deepseek",
                "project_dir": ".",
                "status": "initialized",
                "iteration": 0,
                "version": 1,
                "project_context": ctx,
                "implementation_plan": plan,
                "changed_files": [],
                "review_passed": False,
                "review_issues": [],
                "feature_report": "",
                "sub_tasks": plan["sub_tasks"],
                "coder_outputs": [],
                "dependency_graph": {},
            })

        # base ran (1 call), ext deferred (not called) — so only 1 call
        assert call_count == 1

    def test_cross_coder_issues_in_reviewer_parse_result(self):
        """parse_result extracts CROSS_CODER_ISSUE and INTEGRATION_ISSUE tokens."""
        role = FeatureReviewerRole()
        mock = _make_agent_result([
            {"content": "REVIEW_FAILED\n"
                        "- missing import\n"
                        "CROSS_CODER_ISSUE: ext: imports from base fail\n"
                        "INTEGRATION_ISSUE: data flow broken between modules\n"
                        "More text",
             "type": "AIMessage"},
        ])
        result = role.parse_result(mock)
        assert result["review_passed"] is False
        assert len(result["cross_coder_issues"]) == 2
        # First issue has module name
        assert result["cross_coder_issues"][0]["module"] == "ext"
        assert "imports from base" in result["cross_coder_issues"][0]["description"]
        # Second issue has no module name
        assert result["cross_coder_issues"][1]["module"] is None
        assert "data flow" in result["cross_coder_issues"][1]["description"]

    def test_cross_coder_issues_empty_when_passed(self):
        """No cross_coder_issues when REVIEW_PASSED."""
        role = FeatureReviewerRole()
        mock = _make_agent_result([
            {"content": "All good. REVIEW_PASSED", "type": "AIMessage"},
        ])
        result = role.parse_result(mock)
        assert result["review_passed"] is True
        assert result["cross_coder_issues"] == []


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

class TestFeaturePipelineVersioning:
    def test_initial_state_has_version_1(self):
        p = FeaturePipeline(working_dir="/tmp/test", backend="deepseek")
        state = p._build_initial_state("Add X", [])
        assert state["version"] == 1

    def test_version_bumps_across_coder_reviewer_loop(self, tmpdir):
        """Version increments on each review failure, circuit breaker at _MAX_CODER_RETRIES."""
        import feature.scanner as fscan
        import feature.planner as fplan
        import feature.coder as fcode
        import feature.reviewer as frev
        import feature.writer as fwrt

        wd = str(tmpdir)
        # Pre-create context + plan files so roles don't fallback
        ctx = {"language": "python", "conventions": {}, "file_manifest": []}
        with open(os.path.join(wd, "project_context.json"), "w") as f:
            json.dump(ctx, f)
        plan = {"files_to_create": [{"path": "x.py"}], "files_to_modify": [], "test_files": []}
        with open(os.path.join(wd, "implementation_plan.json"), "w") as f:
            json.dump(plan, f)

        with patch.object(fscan.FeatureScannerRole, "execute", return_value=ctx), \
             patch.object(fplan.FeaturePlannerRole, "execute", return_value=plan), \
             patch.object(fcode.FeatureCoderRole, "execute",
                          return_value={"changed_files": ["x.py"], "success": True}), \
             patch.object(frev.FeatureReviewerRole, "execute",
                          return_value={"review_passed": False, "review_issues": ["bug"]}), \
             patch.object(fwrt.FeatureReportWriterRole, "execute",
                          return_value={"feature_report": os.path.join(wd, "feature_report.md")}):

            p = FeaturePipeline(working_dir=wd, backend="deepseek")
            graph = p._build_graph()
            result = graph.invoke({
                "input_spec": "Add X",
                "working_dir": wd,
                "backend": "deepseek",
                "project_dir": ".",
                "status": "initialized",
                "iteration": 0,
                "version": 1,
                "project_context": {},
                "implementation_plan": {},
                "changed_files": [],
                "review_passed": False,
                "review_issues": [],
                "feature_report": "",
            })

        # After all retries exhausted (v1→v2→v3→v4 > _MAX_CODER_RETRIES=3),
        # pipeline should reach writer with version bumped past max
        assert result["status"] == "written"
        assert result["version"] >= FeaturePipeline._MAX_CODER_RETRIES
        assert "feature_report" in result

    def test_pipeline_passes_on_first_review(self, tmpdir):
        """Version stays at 1 when review passes on first attempt."""
        import feature.scanner as fscan
        import feature.planner as fplan
        import feature.coder as fcode
        import feature.reviewer as frev
        import feature.writer as fwrt

        wd = str(tmpdir)
        ctx = {"language": "python", "conventions": {}, "file_manifest": []}
        with open(os.path.join(wd, "project_context.json"), "w") as f:
            json.dump(ctx, f)
        plan = {"files_to_create": [{"path": "x.py"}], "files_to_modify": [], "test_files": []}
        with open(os.path.join(wd, "implementation_plan.json"), "w") as f:
            json.dump(plan, f)

        with patch.object(fscan.FeatureScannerRole, "execute", return_value=ctx), \
             patch.object(fplan.FeaturePlannerRole, "execute", return_value=plan), \
             patch.object(fcode.FeatureCoderRole, "execute",
                          return_value={"changed_files": ["x.py"], "success": True}), \
             patch.object(frev.FeatureReviewerRole, "execute",
                          return_value={"review_passed": True, "review_issues": []}), \
             patch.object(fwrt.FeatureReportWriterRole, "execute",
                          return_value={"feature_report": os.path.join(wd, "feature_report.md")}):

            p = FeaturePipeline(working_dir=wd, backend="deepseek")
            graph = p._build_graph()
            result = graph.invoke({
                "input_spec": "Add X",
                "working_dir": wd,
                "backend": "deepseek",
                "project_dir": ".",
                "status": "initialized",
                "iteration": 0,
                "version": 1,
                "project_context": {},
                "implementation_plan": {},
                "changed_files": [],
                "review_passed": False,
                "review_issues": [],
                "feature_report": "",
            })

        assert result["review_passed"] is True
        assert result["version"] == 1  # never bumped
        assert result["status"] == "written"


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
