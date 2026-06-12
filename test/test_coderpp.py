"""Tests for CoderPPPipeline — head → workers → reviewer → organizer."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import test.conftest  # noqa: F401 — loads tools_config.json

from pipeline.coderpp import (
    CoderPPPipeline, CoderPPState,
    CPP_HEAD_TIMEOUT, CPP_WORKER_TIMEOUT, CPP_MAX_VERSIONS, CPP_MAX_WORKER_RETRIES,
)
from coderpp.head_agent import CoderPPDecomposerRole, ObserverRole, _fallback_decompose
from coderpp.organizer import OrganizerRole


# ═══════════════════════════════════════════════════════════════════════════
# CoderPPState
# ═══════════════════════════════════════════════════════════════════════════

class TestCoderPPState:
    def test_all_required_fields(self):
        fields = CoderPPState.__annotations__
        required = [
            "input_spec", "working_dir", "backend", "sub_tasks",
            "worker_outputs", "reviewed_modules", "project_dir", "status",
            "worker_stats", "version", "environment",
        ]
        for f in required:
            assert f in fields, f"Missing field: {f}"


# ═══════════════════════════════════════════════════════════════════════════
# Decomposer role
# ═══════════════════════════════════════════════════════════════════════════

class TestDecomposerRole:
    def test_agent_name(self):
        role = CoderPPDecomposerRole()
        assert role.agent_name == "coderpp_head"

    def test_max_steps(self):
        role = CoderPPDecomposerRole()
        assert role.max_steps == 20

    def test_tools_for_backend(self):
        role = CoderPPDecomposerRole()
        for backend in ("deepseek", "claude_cli"):
            tools = role.tools_for_backend(backend)
            assert isinstance(tools, list)
            assert len(tools) > 0

    def test_json_template_has_required_fields(self):
        tmpl = CoderPPDecomposerRole._json_template()
        assert "id" in tmpl
        assert "module_name" in tmpl
        assert "description" in tmpl
        assert "files_to_create" in tmpl

    def test_role_prompt_detects_latex(self):
        role = CoderPPDecomposerRole()
        prompt = role._role_prompt(r"\section{a}\section{b} research proposal")
        assert "research proposal" in prompt.lower() or "latex" in prompt.lower()

    def test_role_prompt_standard(self):
        role = CoderPPDecomposerRole()
        prompt = role._role_prompt("Build a web server")
        assert "software architect" in prompt.lower()

    def test_sizing_guide_includes_size_ranges(self):
        guide = CoderPPDecomposerRole()._sizing_guide()
        assert "2-3" in guide or "4-5" in guide

    def test_parse_result_extracts_json_from_messages(self):
        """parse_result extracts JSON array from agent messages."""
        role = CoderPPDecomposerRole()
        msg = MagicMock()
        msg.content = json.dumps([
            {"id": 1, "module_name": "core", "description": "Core module", "files_to_create": ["core.py"]},
            {"id": 2, "module_name": "api", "description": "API layer", "files_to_create": ["api.py"], "dependencies": ["core"]},
        ])
        result = MagicMock()
        result.messages = [msg]
        parsed = role.parse_result(result, working_dir="/tmp", input_spec="Build")
        assert len(parsed) == 2
        assert parsed[0]["module_name"] == "core"
        assert parsed[1]["module_name"] == "api"

    def test_parse_result_reads_from_disk(self, tmpdir):
        """When messages have no JSON, reads from decomposition files on disk."""
        role = CoderPPDecomposerRole()
        decompositions = [{"id": 1, "module_name": "main", "description": "Main", "files_to_create": ["main.py"]}]
        with open(os.path.join(tmpdir, "decomposition.json"), "w") as f:
            json.dump(decompositions, f)

        msg = MagicMock()
        msg.content = "No JSON here"
        result = MagicMock()
        result.messages = [msg]
        parsed = role.parse_result(result, working_dir=tmpdir, input_spec="Test")
        assert len(parsed) == 1
        assert parsed[0]["module_name"] == "main"

    def test_parse_result_falls_back(self):
        """When no JSON in messages or disk, uses _fallback_decompose."""
        role = CoderPPDecomposerRole()
        msg = MagicMock()
        msg.content = "Can't decompose"
        result = MagicMock()
        result.messages = [msg]
        parsed = role.parse_result(result, working_dir="/nonexistent", input_spec="core, api")
        assert isinstance(parsed, list)
        assert len(parsed) >= 2


# ═══════════════════════════════════════════════════════════════════════════
# Fallback decompose
# ═══════════════════════════════════════════════════════════════════════════

class TestFallbackDecompose:
    def test_produces_valid_modules(self):
        result = _fallback_decompose("core, api, utils")
        assert isinstance(result, list)
        assert len(result) >= 2
        for m in result:
            assert "id" in m
            assert "module_name" in m
            assert "description" in m
            assert "files_to_create" in m
            assert isinstance(m["files_to_create"], list)

    def test_always_includes_main_module(self):
        result = _fallback_decompose("web_server")
        names = [m["module_name"] for m in result]
        assert "main" in names, f"Should include main module, got: {names}"

    def test_latex_extracts_sections(self):
        result = _fallback_decompose(
            r"\documentclass{article}\section{Introduction}\section{Methods}"
        )
        assert len(result) >= 2
        names = [m["module_name"] for m in result]
        assert "introduction" in names or "methods" in names

    def test_top_level_entry_depends_on_all(self):
        result = _fallback_decompose("core_utils")
        main = next(m for m in result if m["module_name"] == "main")
        assert len(main.get("dependencies", [])) > 0

    def test_respects_20_module_limit(self):
        many = ", ".join(f"mod_{i}" for i in range(30))
        result = _fallback_decompose(many)
        assert len(result) <= 20

    def test_string_dependencies_maintained(self):
        """Dependencies should be module_name strings, not integer IDs."""
        result = _fallback_decompose("core, api")
        main = next(m for m in result if m["module_name"] == "main")
        deps = main.get("dependencies", [])
        assert all(isinstance(d, str) for d in deps), f"All deps should be strings, got: {deps}"


# ═══════════════════════════════════════════════════════════════════════════
# Observer role
# ═══════════════════════════════════════════════════════════════════════════

class TestObserverRole:
    def test_agent_name(self):
        role = ObserverRole()
        assert role.agent_name == "coderpp_observer"

    def test_max_steps(self):
        role = ObserverRole()
        assert role.max_steps == 8

    def test_tools_for_backend(self):
        role = ObserverRole()
        tools = role.tools_for_backend("deepseek")
        assert len(tools) > 0

    def test_build_task_includes_workers(self):
        role = ObserverRole()
        task = role.build_task(
            "deepseek",
            worker_outputs=[
                {"module_name": "core", "files": ["core.py"], "summary": "Done"},
            ],
            sub_tasks=[],
        )
        assert "core" in task
        assert "OBSERVATIONS.md" in task

    def test_parse_result_returns_path_when_file_exists(self, tmpdir):
        """Returns path to OBSERVATIONS.md when the file exists on disk."""
        role = ObserverRole()
        with open(os.path.join(tmpdir, "OBSERVATIONS.md"), "w") as f:
            f.write("# Observations\nWorker 1: OK\nWorker 2: Issues found")
        mock = MagicMock()
        mock.messages = []
        result = role.parse_result(mock, working_dir=tmpdir)
        assert result.endswith("OBSERVATIONS.md")

    def test_parse_result_returns_empty_when_no_file(self, tmpdir):
        role = ObserverRole()
        mock = MagicMock()
        mock.messages = []
        assert role.parse_result(mock, working_dir=tmpdir) == ""

    def test_build_task_includes_sub_tasks(self):
        role = ObserverRole()
        task = role.build_task(
            "deepseek",
            worker_outputs=[{"module_name": "core", "files": ["core.py"], "summary": "Done"}],
            sub_tasks=[{"module_name": "core", "description": "Core logic"}],
        )
        assert "core" in task
        assert "OBSERVATIONS.md" in task


# ═══════════════════════════════════════════════════════════════════════════
# Organizer role
# ═══════════════════════════════════════════════════════════════════════════

class TestOrganizerRole:
    def test_agent_name(self):
        role = OrganizerRole()
        assert role.agent_name == "coderpp_organizer"

    def test_max_steps(self):
        role = OrganizerRole()
        assert role.max_steps == 15

    def test_tools_for_backend(self):
        role = OrganizerRole()
        tools = role.tools_for_backend("deepseek")
        assert len(tools) > 0

    def test_build_task_includes_modules_and_project_dir(self):
        role = OrganizerRole()
        task = role.build_task(
            "deepseek",
            reviewed_modules=[
                {"module_name": "core", "files": ["core.py"], "passed": True},
            ],
            input_spec="Build a CLI",
        )
        assert "core" in task
        assert "project" in task
        assert "Build a CLI" in task

    def test_build_task_claude_cli_uses_native_tool_names(self):
        role = OrganizerRole()
        task = role.build_task(
            "claude_cli",
            reviewed_modules=[{"module_name": "app", "files": ["app.py"], "passed": True}],
            input_spec="Build app",
        )
        assert "app" in task
        assert "project" in task

    def test_parse_result_detects_project_dir(self, tmpdir):
        import os
        role = OrganizerRole()
        os.makedirs(os.path.join(tmpdir, "project"))
        mock = MagicMock()
        result = role.parse_result(mock, working_dir=tmpdir)
        assert result == "project"

    def test_parse_result_no_project_dir(self, tmpdir):
        role = OrganizerRole()
        mock = MagicMock()
        result = role.parse_result(mock, working_dir=tmpdir)
        assert result == ""


# ═══════════════════════════════════════════════════════════════════════════
# Worker role (tested via coderpp.worker_agent module)
# ═══════════════════════════════════════════════════════════════════════════

class TestWorkerRoleBehavior:
    def test_parse_result_scans_module_directory(self, tmpdir):
        """Worker parse_result scans the module directory for created files."""
        from coderpp.worker_agent import CoderPPWorkerRole

        role = CoderPPWorkerRole()
        # Set up module directory with files
        module_dir = os.path.join(tmpdir, "modules", "test_mod")
        os.makedirs(module_dir)
        with open(os.path.join(module_dir, "core.py"), "w") as f:
            f.write("def main(): pass\n")
        with open(os.path.join(module_dir, "test_core.py"), "w") as f:
            f.write("def test_main(): pass\n")

        sub_task = {"id": 1, "module_name": "test_mod", "description": "Test module"}
        result = MagicMock()
        result.success = True
        result.messages = []

        parsed = role.parse_result(result, working_dir=tmpdir, sub_task=sub_task)
        assert parsed["module_name"] == "test_mod"
        assert len(parsed["files"]) >= 2
        file_names = [os.path.basename(f) for f in parsed["files"]]
        assert "core.py" in file_names
        assert "test_core.py" in file_names

    def test_parse_result_agent_not_successful(self):
        """When agent result.success=False, returns empty files."""
        from coderpp.worker_agent import CoderPPWorkerRole

        role = CoderPPWorkerRole()
        sub_task = {"id": 1, "module_name": "test_mod"}
        result = MagicMock()
        result.success = False
        result.messages = []

        parsed = role.parse_result(result, working_dir="/tmp", sub_task=sub_task)
        assert parsed["files"] == []
        assert "did not complete successfully" in parsed["summary"]

    def test_parse_result_extracts_summary_from_messages(self, tmpdir):
        """Extracts a summary from the last AIMessage content (must be > 100 chars)."""
        from coderpp.worker_agent import CoderPPWorkerRole

        role = CoderPPWorkerRole()
        os.makedirs(os.path.join(tmpdir, "modules", "mod_a"))
        from unittest.mock import MagicMock as MM
        msg = MM()
        msg.content = (
            "Module complete. Implemented core logic with comprehensive error handling, "
            "input validation, type hints, and edge case coverage. All tests pass. TASK_COMPLETE"
        )
        sub_task = {"id": 1, "module_name": "mod_a"}
        result = MM()
        result.success = True
        result.messages = [msg]

        parsed = role.parse_result(result, working_dir=str(tmpdir), sub_task=sub_task)
        assert "core logic" in parsed.get("summary", "")


# ═══════════════════════════════════════════════════════════════════════════
# Reviewer role (tested via coderpp.reviewer_agent module)
# ═══════════════════════════════════════════════════════════════════════════

class TestReviewerRoleBehavior:
    def test_parse_result_detects_review_passed(self, tmpdir):
        """Reviewer parse_result detects REVIEW_PASSED in reversed AIMessages."""
        from coderpp.reviewer_agent import CoderPPReviewerRole

        role = CoderPPReviewerRole()
        worker_output = {"sub_task_id": 1, "module_name": "core", "files": ["modules/core/core.py"]}

        msg = MagicMock()
        msg.content = "All checks passed. REVIEW_PASSED"
        type(msg).__name__ = "AIMessage"

        result = MagicMock()
        result.success = True
        result.messages = [msg]

        parsed = role.parse_result(result, working_dir=tmpdir, worker_output=worker_output)
        assert parsed["passed"] is True
        assert parsed["module_name"] == "core"

    def test_parse_result_detects_review_failed(self, tmpdir):
        """Reviewer parse_result detects REVIEW_FAILED in reversed AIMessages."""
        from coderpp.reviewer_agent import CoderPPReviewerRole

        role = CoderPPReviewerRole()
        worker_output = {"sub_task_id": 1, "module_name": "core", "files": ["modules/core/core.py"]}

        msg = MagicMock()
        msg.content = "Issues found.\n- Missing error handling\nREVIEW_FAILED"
        type(msg).__name__ = "AIMessage"

        result = MagicMock()
        result.success = True
        result.messages = [msg]

        parsed = role.parse_result(result, working_dir=tmpdir, worker_output=worker_output)
        assert parsed["passed"] is False

    def test_parse_result_review_failed_beats_passed_in_same_message(self, tmpdir):
        """REVIEW_FAILED takes precedence when both tokens appear in the same message."""
        from coderpp.reviewer_agent import CoderPPReviewerRole

        role = CoderPPReviewerRole()
        worker_output = {"sub_task_id": 1, "module_name": "core", "files": []}

        msg = MagicMock()
        msg.content = "Checks: REVIEW_PASSED for style, but REVIEW_FAILED for tests"
        type(msg).__name__ = "AIMessage"

        result = MagicMock()
        result.success = True
        result.messages = [msg]

        parsed = role.parse_result(result, working_dir=tmpdir, worker_output=worker_output)
        # REVIEW_PASSED detected first in forward scan but REVIEW_FAILED also present
        # The condition: "REVIEW_PASSED" in content and "REVIEW_FAILED" not in content
        # Since both are present, neither branch triggers → passed stays False
        assert parsed["passed"] is False

    def test_parse_result_reads_review_md_as_authoritative(self, tmpdir):
        """When review.md exists, its verdict takes precedence over message content."""
        from coderpp.reviewer_agent import CoderPPReviewerRole

        role = CoderPPReviewerRole()
        module_dir = os.path.join(tmpdir, "modules", "core")
        os.makedirs(module_dir)
        with open(os.path.join(module_dir, "review.md"), "w") as f:
            f.write("## Review\n\nAll checks passed. REVIEW_PASSED\n\nApproved.")

        worker_output = {"sub_task_id": 1, "module_name": "core", "files": ["modules/core/core.py"]}

        # Message says REVIEW_FAILED but review.md says REVIEW_PASSED
        msg = MagicMock()
        msg.content = "REVIEW_FAILED"  # wrong verdict
        type(msg).__name__ = "AIMessage"

        result = MagicMock()
        result.success = True
        result.messages = [msg]

        parsed = role.parse_result(result, working_dir=tmpdir, worker_output=worker_output)
        # review.md overrides message verdict
        assert parsed["passed"] is True

    def test_parse_result_agent_not_successful(self):
        """When agent fails, returns passed=False regardless of content."""
        from coderpp.reviewer_agent import CoderPPReviewerRole

        role = CoderPPReviewerRole()
        worker_output = {"sub_task_id": 1, "module_name": "core", "files": []}

        result = MagicMock()
        result.success = False
        result.messages = []

        parsed = role.parse_result(result, working_dir="/tmp", worker_output=worker_output)
        assert parsed["passed"] is False


# ═══════════════════════════════════════════════════════════════════════════
# CoderPPPipeline
# ═══════════════════════════════════════════════════════════════════════════

class TestCoderPPPipeline:
    def test_pipeline_name(self):
        p = CoderPPPipeline(working_dir="/tmp/test")
        assert p.name == "coderpp"

    def test_default_output_dir(self):
        assert CoderPPPipeline.default_output_dir == "coderpp_output"

    def test_build_initial_state(self):
        p = CoderPPPipeline(working_dir="/tmp/test", backend="claude_cli")
        sub_tasks = [{"id": 1, "module_name": "core", "description": "Core logic"}]
        state = p._build_initial_state("/tmp/spec.tex", sub_tasks)
        assert state["input_spec"] == "/tmp/spec.tex"
        assert state["working_dir"] == "/tmp/test"
        assert state["backend"] == "claude_cli"
        assert state["status"] == "decomposed"
        assert state["version"] == 1
        assert state["environment"] == ""

    def test_build_graph_compiles(self):
        p = CoderPPPipeline(working_dir="/tmp/test")
        graph = p._build_graph()
        assert graph is not None

    def test_constants(self):
        assert CPP_HEAD_TIMEOUT == 500
        assert CPP_WORKER_TIMEOUT == 1200
        assert CPP_MAX_VERSIONS == 5
        assert CPP_MAX_WORKER_RETRIES == 5

    def test_decompose_reads_tex_file(self, tmpdir):
        tex_path = os.path.join(tmpdir, "spec.tex")
        with open(tex_path, "w") as f:
            f.write(r"\documentclass{article}\section{Intro}hello\section{Methods}")
        p = CoderPPPipeline(working_dir=tmpdir, backend="deepseek")
        # Mock decompose_to_modules to avoid real LLM call
        with patch("pipeline.coderpp.decompose_to_modules",
                   return_value=[{"id": 1, "module_name": "core",
                                  "description": "test", "files_to_create": []}]):
            result = p._decompose(tex_path)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_decompose_reads_md_file(self, tmpdir):
        md_path = os.path.join(tmpdir, "spec.md")
        with open(md_path, "w") as f:
            f.write("# Specification\nImplement this pipeline.")
        p = CoderPPPipeline(working_dir=tmpdir, backend="deepseek")
        with patch("pipeline.coderpp.decompose_to_modules",
                   return_value=[{"id": 1, "module_name": "api",
                                  "description": "test", "files_to_create": []}]):
            result = p._decompose(md_path)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_decompose_direct_text(self):
        p = CoderPPPipeline(working_dir="/tmp/test", backend="deepseek")
        with patch("pipeline.coderpp.decompose_to_modules",
                   return_value=[{"id": 1, "module_name": "server",
                                  "description": "test", "files_to_create": []}]):
            result = p._decompose("Build a web server")
        assert isinstance(result, list)
        assert len(result) == 1

    def test_display_decomposition_with_deps(self, capsys):
        p = CoderPPPipeline(working_dir="/tmp/test")
        sub_tasks = [
            {"id": 1, "module_name": "core", "description": "Core module",
             "files_to_create": ["core.py"], "dependencies": []},
            {"id": 2, "module_name": "api", "description": "API layer",
             "files_to_create": ["api.py"], "dependencies": ["core"]},
        ]
        p._display_decomposition(sub_tasks)
        out = capsys.readouterr().out
        assert "core" in out
        assert "api" in out
        assert "depends on:" in out

    def test_decompose_non_existent_md_file(self, tmpdir):
        """Decompose handles non-existent file paths gracefully (by passing as text)."""
        p = CoderPPPipeline(working_dir=tmpdir, backend="deepseek")
        with patch("pipeline.coderpp.decompose_to_modules",
                   return_value=[{"id": 1, "module_name": "mod",
                                  "description": "test", "files_to_create": []}]):
            result = p._decompose(os.path.join(tmpdir, "nonexistent.md"))
        assert isinstance(result, list)

    def test_manage_output_dir(self, tmpdir):
        p = CoderPPPipeline(working_dir=tmpdir)
        p.manage_output_dir()
        assert os.path.isdir(tmpdir)


# ═══════════════════════════════════════════════════════════════════════════
# Flow dict & status routing
# ═══════════════════════════════════════════════════════════════════════════

class TestFlowRouting:
    def test_graph_compiles_with_all_transitions(self):
        """The graph compiles, validating all flow transitions."""
        p = CoderPPPipeline(working_dir="/tmp/test")
        graph = p._build_graph()
        assert graph is not None

    def test_flow_routes_decomposed_to_workers(self):
        from pipeline.base import BasePipeline
        flow = {"decomposed": "workers"}
        router = BasePipeline._status_router(flow)
        assert router({"status": "decomposed"}) == "workers"

    def test_flow_routes_worker_all_success_to_observer(self):
        from pipeline.base import BasePipeline
        flow = {"worker_all_success": "observer"}
        router = BasePipeline._status_router(flow)
        assert router({"status": "worker_all_success"}) == "observer"

    def test_flow_routes_reviewed_all_passed_to_organizer(self):
        from pipeline.base import BasePipeline
        flow = {"reviewed_all_passed": "organizer"}
        router = BasePipeline._status_router(flow)
        assert router({"status": "reviewed_all_passed"}) == "organizer"

    def test_flow_routes_reviewed_retry_to_workers(self):
        from pipeline.base import BasePipeline
        flow = {"reviewed_retry": "workers"}
        router = BasePipeline._status_router(flow)
        assert router({"status": "reviewed_retry"}) == "workers"

    def test_flow_routes_assembled_to_end(self):
        from pipeline.base import BasePipeline
        from langgraph.graph import END
        flow = {"assembled": END}
        router = BasePipeline._status_router(flow)
        assert router({"status": "assembled"}) == END


# ═══════════════════════════════════════════════════════════════════════════
# Resume state reconstruction
# ═══════════════════════════════════════════════════════════════════════════

class TestResumeState:
    def test_no_decomposition_file(self, tmpdir):
        """Returns None when no decomposition.json exists."""
        p = CoderPPPipeline(working_dir=tmpdir, resume=True)
        assert p._try_load_resume_state("Test") is None

    def test_with_decomposition_file(self, tmpdir):
        """Reconstructs state from decomposition.json."""
        sub_tasks = [
            {"id": 1, "module_name": "core", "description": "Core"},
            {"id": 2, "module_name": "api", "description": "API"},
        ]
        with open(os.path.join(tmpdir, "decomposition.json"), "w") as f:
            json.dump(sub_tasks, f)

        p = CoderPPPipeline(working_dir=tmpdir, resume=True)
        state = p._try_load_resume_state("Test")
        assert state is not None
        assert state["status"] == "decomposed"
        assert len(state["sub_tasks"]) == 2

    def test_with_environment_file(self, tmpdir):
        """Reads ENVIRONMENT.md from disk."""
        sub_tasks = [{"id": 1, "module_name": "core", "description": "Core"}]
        with open(os.path.join(tmpdir, "decomposition.json"), "w") as f:
            json.dump(sub_tasks, f)
        with open(os.path.join(tmpdir, "ENVIRONMENT.md"), "w") as f:
            f.write("- Path: /usr/local\n- Version: 3.11\n")

        p = CoderPPPipeline(working_dir=tmpdir, resume=True)
        state = p._try_load_resume_state("Test")
        assert state is not None
        assert "/usr/local" in state.get("environment", "")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
