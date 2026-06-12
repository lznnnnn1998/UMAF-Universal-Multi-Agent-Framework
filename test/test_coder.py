"""Tests for CoderPipeline — coder ↔ reviewer loop (max 5 cycles)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import test.conftest  # noqa: F401 — loads tools_config.json

from pipeline.coder import CoderPipeline, CoderRole, ReviewerRole, MultiAgentState


# ═══════════════════════════════════════════════════════════════════════════
# MultiAgentState
# ═══════════════════════════════════════════════════════════════════════════

class TestMultiAgentState:
    def test_all_required_fields(self):
        fields = MultiAgentState.__annotations__
        required = [
            "messages", "current_agent", "requirement", "working_dir",
            "review_passed", "iteration", "backend", "coder_files",
        ]
        for f in required:
            assert f in fields, f"Missing field: {f}"


# ═══════════════════════════════════════════════════════════════════════════
# CoderRole
# ═══════════════════════════════════════════════════════════════════════════

class TestCoderRole:
    def test_agent_name(self):
        role = CoderRole()
        assert role.agent_name == "coder"

    def test_max_steps(self):
        role = CoderRole()
        assert role.max_steps == 15

    def test_tools_for_backend(self):
        role = CoderRole()
        for backend in ("deepseek", "claude_cli"):
            tools = role.tools_for_backend(backend)
            assert isinstance(tools, list)
            assert len(tools) > 0
            names = {t["name"] for t in tools}
            assert "read_file" in names
            assert "write_file" in names

    def test_build_task_includes_guidelines(self):
        role = CoderRole()
        task = role.build_task("deepseek", requirement="Build a CLI tool")
        assert "Build a CLI tool" in task
        assert "CLI arguments" in task or "sys.argv" in task or "argparse" in task
        assert "TASK_COMPLETE" in task

    def test_build_task_requirement_is_required(self):
        """build_task embeds the requirement into the design-guidelines prompt."""
        role = CoderRole()
        task = role.build_task("deepseek", requirement="Create an API server")
        assert "Create an API server" in task
        assert "testable functions" in task.lower()

    def test_default_parse_result_passthrough(self):
        """CoderRole uses AgentRole default parse_result — returns result as-is."""
        role = CoderRole()
        result = MagicMock()
        parsed = role.parse_result(result, working_dir="/tmp")
        assert parsed is result


# ═══════════════════════════════════════════════════════════════════════════
# ReviewerRole
# ═══════════════════════════════════════════════════════════════════════════

class TestReviewerRole:
    def test_agent_name(self):
        role = ReviewerRole()
        assert role.agent_name == "reviewer"

    def test_max_steps(self):
        role = ReviewerRole()
        assert role.max_steps == 10

    def test_tools_do_not_include_write(self):
        role = ReviewerRole()
        tools = role.tools_for_backend("deepseek")
        names = {t["name"] for t in tools}
        assert "read_file" in names
        # Reviewer from tools_config has run_command, call_claude, web_search, web_fetch
        assert "write_file" not in names

    def test_build_task_no_files_section(self):
        role = ReviewerRole()
        task = role.build_task("deepseek", requirement="Build X")
        assert "REVIEW_PASSED" in task
        assert "REVIEW_FAILED" in task
        assert "Build X" in task

    def test_build_task_with_coder_files(self):
        role = ReviewerRole()
        task = role.build_task("deepseek", requirement="Build X",
                               coder_files=["main.py", "test_main.py", "utils.py"])
        assert "main.py" in task
        assert "test_main.py" in task
        assert "utils.py" in task
        assert "Files Produced by Coder" in task

    def test_build_task_coder_files_truncation(self):
        """When coder_files > 50, the list is truncated with a count message."""
        role = ReviewerRole()
        files = [f"file_{i:03d}.py" for i in range(75)]
        task = role.build_task("deepseek", requirement="Build X", coder_files=files)
        assert "file_000.py" in task
        assert "file_049.py" in task
        assert "file_050.py" not in task  # 51st file (0-indexed 50) truncated
        assert "25 more files" in task

    def test_default_parse_result_passthrough(self):
        """ReviewerRole uses AgentRole default parse_result — returns result as-is."""
        role = ReviewerRole()
        result = MagicMock()
        parsed = role.parse_result(result, working_dir="/tmp")
        assert parsed is result


# ═══════════════════════════════════════════════════════════════════════════
# CoderPipeline — graph node behavior (mocked LLM calls)
# ═══════════════════════════════════════════════════════════════════════════

class TestCoderPipelineNodes:
    """Test the graph node logic by mocking role.execute() calls.

    The coder pipeline graph runs to completion: coder ↔ reviewer loop until
    review_passed=True or iteration >= 5. All tests here mock BOTH roles so
    the graph terminates deterministically.
    """

    def _make_mock_messages(self, content: str):
        """Build a mock AgentResult with one AIMessage."""
        msg = MagicMock()
        msg.content = content
        type(msg).__name__ = "AIMessage"
        result = MagicMock()
        result.messages = [msg]
        result.success = True
        return result

    def test_coder_node_scans_working_dir_for_files(self, tmpdir):
        """After coder runs, coder_files lists all non-hidden files in working_dir."""
        with open(os.path.join(tmpdir, "main.py"), "w") as f:
            f.write("print('hello')")
        with open(os.path.join(tmpdir, "test_main.py"), "w") as f:
            f.write("def test(): pass")
        os.makedirs(os.path.join(tmpdir, "subpkg"))
        with open(os.path.join(tmpdir, "subpkg", "utils.py"), "w") as f:
            f.write("def helper(): pass")

        from pipeline.coder import CoderRole, ReviewerRole
        coder_result = self._make_mock_messages("TASK_COMPLETE")
        review_result = self._make_mock_messages("REVIEW_PASSED")

        with patch.object(CoderRole, "execute", return_value=coder_result), \
             patch.object(ReviewerRole, "execute", return_value=review_result):
            p = CoderPipeline(working_dir=tmpdir, backend="deepseek", yes=True)
            graph = p._build_graph()

            state: MultiAgentState = {
                "messages": [], "current_agent": "coder", "requirement": "Build X",
                "working_dir": tmpdir, "review_passed": False, "iteration": 0,
                "backend": "deepseek", "coder_files": [],
            }
            result = graph.invoke(state)
            assert "main.py" in result["coder_files"]
            assert "test_main.py" in result["coder_files"]
            assert "subpkg/utils.py" in result["coder_files"]

    def test_reviewer_node_detects_review_passed(self, tmpdir):
        """When reviewer returns REVIEW_PASSED, the graph terminates with review_passed=True."""
        from pipeline.coder import CoderRole, ReviewerRole
        result_mock = self._make_mock_messages("All checks pass. REVIEW_PASSED")
        with patch.object(CoderRole, "execute", return_value=self._make_mock_messages("TASK_COMPLETE")), \
             patch.object(ReviewerRole, "execute", return_value=result_mock):
            p = CoderPipeline(working_dir=tmpdir, backend="deepseek", yes=True)
            graph = p._build_graph()

            state: MultiAgentState = {
                "messages": [], "current_agent": "reviewer", "requirement": "Build X",
                "working_dir": tmpdir, "review_passed": False, "iteration": 1,
                "backend": "deepseek", "coder_files": [],
            }
            result = graph.invoke(state)
            assert result["review_passed"] is True

    def test_reviewer_node_detects_review_failed(self, tmpdir):
        """When reviewer returns REVIEW_FAILED, router routes back to coder."""
        from pipeline.coder import CoderRole, ReviewerRole
        fail_review = self._make_mock_messages("Issues found.\n- Missing imports\nREVIEW_FAILED")
        pass_review = self._make_mock_messages("Fixed. REVIEW_PASSED")

        with patch.object(CoderRole, "execute", return_value=self._make_mock_messages("TASK_COMPLETE")), \
             patch.object(ReviewerRole, "execute", side_effect=[fail_review, pass_review]):
            p = CoderPipeline(working_dir=tmpdir, backend="deepseek", yes=True)
            graph = p._build_graph()

            state: MultiAgentState = {
                "messages": [], "current_agent": "reviewer", "requirement": "Build X",
                "working_dir": tmpdir, "review_passed": False, "iteration": 1,
                "backend": "deepseek", "coder_files": [],
            }
            result = graph.invoke(state)
            # First review failed → routes to coder → coder runs → review again → passes
            assert result["review_passed"] is True

    def test_reviewer_reverse_scan_uses_last_verdict(self, tmpdir):
        """When multiple AIMessages have verdicts, the most recent (last) wins."""
        from pipeline.coder import CoderRole, ReviewerRole

        msg1 = MagicMock()
        msg1.content = "Let me check... REVIEW_FAILED might be needed."
        type(msg1).__name__ = "AIMessage"
        msg2 = MagicMock()
        msg2.content = "Actually everything looks good. REVIEW_PASSED"
        type(msg2).__name__ = "AIMessage"

        result_mock = MagicMock()
        result_mock.messages = [msg1, msg2]
        result_mock.success = True

        with patch.object(CoderRole, "execute", return_value=self._make_mock_messages("TASK_COMPLETE")), \
             patch.object(ReviewerRole, "execute", return_value=result_mock):
            p = CoderPipeline(working_dir=tmpdir, backend="deepseek", yes=True)
            graph = p._build_graph()

            state: MultiAgentState = {
                "messages": [], "current_agent": "reviewer", "requirement": "Build X",
                "working_dir": tmpdir, "review_passed": False, "iteration": 1,
                "backend": "deepseek", "coder_files": [],
            }
            result = graph.invoke(state)
            # Last AIMessage says REVIEW_PASSED → should be True
            assert result["review_passed"] is True

    def test_router_ends_at_max_iterations(self, tmpdir):
        """When iteration reaches 5 (max), the router returns END."""
        from pipeline.coder import CoderRole

        result_mock = self._make_mock_messages("TASK_COMPLETE")
        with patch.object(CoderRole, "execute", return_value=result_mock):
            p = CoderPipeline(working_dir=tmpdir, backend="deepseek", yes=True)
            graph = p._build_graph()

            state: MultiAgentState = {
                "messages": [], "current_agent": "coder", "requirement": "Build X",
                "working_dir": tmpdir, "review_passed": False, "iteration": 4,
                "backend": "deepseek", "coder_files": [],
            }
            result = graph.invoke(state)
            assert result["iteration"] == 5

    def test_full_loop_review_fails_then_passes(self, tmpdir):
        """Coder runs → reviewer fails → coder runs again → reviewer passes."""
        from pipeline.coder import CoderRole, ReviewerRole

        coder_result = self._make_mock_messages("Code written. TASK_COMPLETE")
        fail_review = self._make_mock_messages("Issue found.\nREVIEW_FAILED")
        pass_review = self._make_mock_messages("All good. REVIEW_PASSED")

        with patch.object(CoderRole, "execute", return_value=coder_result) as mock_coder, \
             patch.object(ReviewerRole, "execute", side_effect=[fail_review, pass_review]) as mock_review:
            p = CoderPipeline(working_dir=tmpdir, backend="deepseek", yes=True)
            graph = p._build_graph()

            state: MultiAgentState = {
                "messages": [], "current_agent": "coder", "requirement": "Build X",
                "working_dir": tmpdir, "review_passed": False, "iteration": 0,
                "backend": "deepseek", "coder_files": [],
            }
            result = graph.invoke(state)
            assert result["review_passed"] is True
            assert mock_coder.call_count >= 1
            assert mock_review.call_count >= 1


# ═══════════════════════════════════════════════════════════════════════════
# CoderPipeline — structure and lifecycle
# ═══════════════════════════════════════════════════════════════════════════

class TestCoderPipeline:
    def test_pipeline_name(self):
        p = CoderPipeline(working_dir="/tmp/test")
        assert p.name == "coder"

    def test_default_output_dir(self):
        assert CoderPipeline.default_output_dir == "coder_output"

    def test_decompose_returns_single_task(self):
        p = CoderPipeline(working_dir="/tmp/test")
        result = p._decompose("Implement a fast sort")
        assert len(result) == 1
        assert result[0]["id"] == 1
        assert result[0]["title"] == "Requirement"
        assert result[0]["description"] == "Implement a fast sort"

    def test_display_decomposition_shows_requirement(self, capsys):
        p = CoderPipeline(working_dir="/tmp/test")
        p._display_decomposition([{"id": 1, "title": "R", "description": "Test req"}])
        out = capsys.readouterr().out
        assert "Test req" in out

    def test_build_initial_state(self):
        p = CoderPipeline(working_dir="/tmp/test", backend="claude_cli")
        state = p._build_initial_state("Build X", [{"id": 1}])
        assert state["requirement"] == "Build X"
        assert state["working_dir"] == "/tmp/test"
        assert state["backend"] == "claude_cli"
        assert state["review_passed"] is False
        assert state["iteration"] == 0
        assert state["current_agent"] == "coder"
        assert state["coder_files"] == []

    def test_build_graph_compiles(self):
        p = CoderPipeline(working_dir="/tmp/test")
        graph = p._build_graph()
        assert graph is not None

    def test_manage_output_dir(self, tmpdir):
        p = CoderPipeline(working_dir=tmpdir)
        p.manage_output_dir()
        assert os.path.isdir(tmpdir)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
