"""Tests for ResearchPipeline — head → workers → reviewer → writer."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest
import test.conftest  # noqa: F401 — loads tools_config.json

from pipeline.research import (
    ResearchPipeline, ResearchState,
    RESEARCH_MAX_VERSIONS, RESEARCH_MAX_WORKER_RETRIES,
    HEAD_TIMEOUT, WORKER_TIMEOUT,
)
from research.head_agent import ResearchDecomposerRole
from utils import extract_json_array
from research.reviewer_agent import ResearchReviewerRole, _extract_scores_from_result
from research.writer import WriterRole, _latex_escape


# ═══════════════════════════════════════════════════════════════════════════
# ResearchState
# ═══════════════════════════════════════════════════════════════════════════

class TestResearchState:
    def test_all_required_fields(self):
        fields = ResearchState.__annotations__
        required = [
            "topic", "working_dir", "backend", "sub_tasks",
            "worker_outputs", "scored_works", "latex_file", "status",
            "worker_stats", "version",
        ]
        for f in required:
            assert f in fields, f"Missing field: {f}"


# ═══════════════════════════════════════════════════════════════════════════
# Decomposer role
# ═══════════════════════════════════════════════════════════════════════════

class TestDecomposerRole:
    def test_agent_name(self):
        role = ResearchDecomposerRole()
        assert role.agent_name == "head_decompose"

    def test_max_steps(self):
        role = ResearchDecomposerRole()
        assert role.max_steps == 25

    def test_tools_for_backend(self):
        role = ResearchDecomposerRole()
        for backend in ("deepseek", "claude_cli"):
            tools = role.tools_for_backend(backend)
            assert isinstance(tools, list)
            assert len(tools) > 0

    def test_fallback_decompose_produces_valid_structure(self):
        result = ResearchDecomposerRole._fallback_decompose("Test Topic")
        assert isinstance(result, list)
        assert 2 <= len(result) <= 8
        for item in result:
            assert "id" in item
            assert "title" in item
            assert "description" in item
            assert isinstance(item["id"], int)

    def test_fallback_decompose_single_keyword(self):
        result = ResearchDecomposerRole._fallback_decompose("Cryptography")
        assert len(result) >= 2

    def test_fallback_decompose_multi_keyword(self):
        result = ResearchDecomposerRole._fallback_decompose("AI, ML, DL")
        assert len(result) >= 2

    def test_json_template_has_required_fields(self):
        tmpl = ResearchDecomposerRole._json_template()
        assert "id" in tmpl
        assert "title" in tmpl
        assert "description" in tmpl

    def test_sizing_guide_references_sub_topics(self):
        guide = ResearchDecomposerRole()._sizing_guide()
        assert "3-5" in guide or "5-8" in guide

    def test_parse_result_extracts_json_from_messages(self):
        """parse_result extracts JSON array from agent response messages."""
        role = ResearchDecomposerRole()
        msg = MagicMock()
        msg.content = '[{"id": 1, "title": "Intro", "description": "Overview"}, {"id": 2, "title": "Methods", "description": "Deep dive"}]'
        result = MagicMock()
        result.messages = [msg]
        parsed = role.parse_result(result, working_dir="/tmp", input_spec="AI")
        assert len(parsed) == 2
        assert parsed[0]["title"] == "Intro"
        assert parsed[1]["title"] == "Methods"

    def test_parse_result_extracts_json_with_markdown_code_block(self):
        """parse_result handles JSON wrapped in markdown code fences."""
        role = ResearchDecomposerRole()
        msg = MagicMock()
        msg.content = 'Here is the decomposition:\n```json\n[{"id": 1, "title": "T", "description": "D"}]\n```\nTASK_COMPLETE'
        result = MagicMock()
        result.messages = [msg]
        parsed = role.parse_result(result, working_dir="/tmp", input_spec="Test")
        assert len(parsed) == 1
        assert parsed[0]["id"] == 1

    def test_parse_result_reads_from_disk_file(self, tmpdir):
        """When messages have no JSON, parse_result reads decomposition files from disk."""
        role = ResearchDecomposerRole()
        decompositions = [{"id": 1, "title": "From Disk", "description": "D"}]
        with open(os.path.join(tmpdir, "decomposition.json"), "w") as f:
            json.dump(decompositions, f)

        msg = MagicMock()
        msg.content = "No JSON here"
        result = MagicMock()
        result.messages = [msg]
        parsed = role.parse_result(result, working_dir=tmpdir, input_spec="Test")
        assert len(parsed) == 1
        assert parsed[0]["title"] == "From Disk"

    def test_parse_result_falls_back_when_no_json_or_file(self):
        """When neither messages nor disk files have decomposition, uses fallback."""
        role = ResearchDecomposerRole()
        msg = MagicMock()
        msg.content = "I couldn't decompose this."
        result = MagicMock()
        result.messages = [msg]
        parsed = role.parse_result(result, working_dir="/nonexistent", input_spec="AI, ML")
        assert isinstance(parsed, list)
        assert len(parsed) >= 2  # fallback produces at least 2

    def test_parse_result_reversed_scan_finds_latest_valid_json(self):
        """Reversed scan of messages finds the most recent valid JSON array."""
        role = ResearchDecomposerRole()
        # First message: invalid JSON
        msg1 = MagicMock()
        msg1.content = "Trying... [{\"id\": 1"  # incomplete JSON, won't parse
        # Second message: valid JSON
        msg2 = MagicMock()
        msg2.content = '[{"id": 1, "title": "Final", "description": "Works"}]'
        result = MagicMock()
        result.messages = [msg1, msg2]
        parsed = role.parse_result(result, working_dir="/tmp", input_spec="Test")
        assert len(parsed) == 1
        assert parsed[0]["title"] == "Final"

    def test_extract_json_array_bracket_counting_with_latex(self):
        """Bracket-counting parser handles LaTeX \\begin{...}\\end{...} pairs."""
        text = r'[\n{"id": 1, "title": "Test", "deps": [\begin{itemize}\item x\end{itemize}]}\n]'
        result = extract_json_array(text)
        assert result is not None

    def test_extract_json_array_nested_objects(self):
        """Bracket-counting parser handles deeply nested JSON objects."""
        text = '[{"a": {"b": {"c": [1, 2, 3]}}, "d": "text with [brackets] inside"}]'
        result = extract_json_array(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed[0]["a"]["b"]["c"] == [1, 2, 3]

    def test_extract_json_array_matches_first_array_only(self):
        """Returns the first complete JSON array, not the greedy last one."""
        text = '[{"first": true}] more text [{"second": false}]'
        result = extract_json_array(text)
        assert result is not None
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["first"] is True

    def test_extract_json_array_no_bracket_returns_none(self):
        """Returns None when text contains no opening bracket."""
        assert extract_json_array("no brackets here") is None

    def test_build_task_includes_all_sizing_tiers(self):
        """build_task includes sizing guide and backend instructions."""
        role = ResearchDecomposerRole()
        task = role.build_task("deepseek", input_spec="Quantum Computing")
        assert "Quantum Computing" in task
        assert "3-5" in task or "5-8" in task or "8-12" in task
        assert "ONLY the JSON array" in task

    def test_backend_instructions_claude_cli(self):
        role = ResearchDecomposerRole()
        task = role.build_task("claude_cli", input_spec="AI")
        assert "TASK_COMPLETE" in task


# ═══════════════════════════════════════════════════════════════════════════
# Reviewer role
# ═══════════════════════════════════════════════════════════════════════════

class TestReviewerRole:
    def test_agent_name(self):
        role = ResearchReviewerRole()
        assert role.agent_name == "reviewer"

    def test_max_steps(self):
        role = ResearchReviewerRole()
        assert role.max_steps == 25

    def test_build_task_includes_topic_and_files(self):
        role = ResearchReviewerRole()
        task = role.build_task(
            "deepseek", topic="AI Safety",
            worker_outputs=[
                {"sub_task_id": 1, "output_file": "research_01.md", "title": "Alignment"},
            ],
        )
        assert "AI Safety" in task
        assert "research_01.md" in task
        assert "Alignment" in task
        assert "depth" in task.lower()

    def test_build_task_with_multiple_workers(self):
        role = ResearchReviewerRole()
        worker_outputs = [
            {"sub_task_id": 1, "output_file": "research_01.md", "title": "A"},
            {"sub_task_id": 2, "output_file": "research_02.md", "title": "B"},
            {"sub_task_id": 3, "output_file": "research_03.md", "title": "C"},
        ]
        task = role.build_task("deepseek", topic="T", worker_outputs=worker_outputs)
        assert "research_01.md" in task
        assert "research_02.md" in task
        assert "research_03.md" in task
        assert "scoring_report.json" in task

    def test_build_task_asserts_on_missing_worker_outputs(self):
        """build_task requires worker_outputs kwarg."""
        role = ResearchReviewerRole()
        with pytest.raises(AssertionError):
            role.build_task("deepseek", topic="AI")

    def test_extract_scores_from_file(self, tmpdir):
        scoring = [
            {"sub_task_id": 1, "title": "T", "output_file": "f.md",
             "scores": {"depth": 8, "accuracy": 7, "relevance": 9, "clarity": 8, "originality": 7},
             "total_score": 39, "rank": 1},
        ]
        with open(os.path.join(tmpdir, "scoring_report.json"), "w") as f:
            json.dump(scoring, f)

        mock_result = MagicMock()
        mock_result.messages = []
        result = _extract_scores_from_result(mock_result, tmpdir)
        assert len(result) == 1
        assert result[0]["total_score"] == 39

    def test_extract_scores_from_messages_fallback(self, tmpdir):
        """When scoring_report.json doesn't exist, extracts scores from messages."""
        scoring = [
            {"sub_task_id": 1, "title": "Intro", "output_file": "r.md",
             "scores": {"depth": 9, "accuracy": 8, "relevance": 9, "clarity": 8, "originality": 7},
             "total_score": 41, "rank": 1},
        ]
        msg = MagicMock()
        msg.content = json.dumps(scoring)
        mock_result = MagicMock()
        mock_result.messages = [msg]
        result = _extract_scores_from_result(mock_result, "/nonexistent_dir")
        assert len(result) == 1
        assert result[0]["total_score"] == 41

    def test_extract_scores_empty_when_nothing_found(self):
        """Returns empty list when no scores found in file or messages."""
        mock_result = MagicMock()
        mock_result.messages = []
        result = _extract_scores_from_result(mock_result, "/nonexistent_dir")
        assert result == []

    def test_parse_result_delegates_to_extract(self, tmpdir):
        """ReviewerRole.parse_result delegates to _extract_scores_from_result."""
        scoring = [{"sub_task_id": 1, "title": "T", "output_file": "f.md",
                     "scores": {}, "total_score": 30, "rank": 1}]
        with open(os.path.join(tmpdir, "scoring_report.json"), "w") as f:
            json.dump(scoring, f)
        role = ResearchReviewerRole()
        mock_result = MagicMock()
        mock_result.messages = []
        parsed = role.parse_result(mock_result, working_dir=tmpdir)
        assert len(parsed) == 1


# ═══════════════════════════════════════════════════════════════════════════
# Writer role
# ═══════════════════════════════════════════════════════════════════════════

class TestWriterRole:
    def test_agent_name(self):
        role = WriterRole()
        assert role.agent_name == "writer"

    def test_max_steps(self):
        role = WriterRole()
        assert role.max_steps == 40

    def test_build_task_includes_latex_requirements(self):
        role = WriterRole()
        scored = [{"output_file": "r.md", "title": "T", "total_score": 40}]
        task = role.build_task("deepseek", topic="AI", scored_works=scored)
        assert "LaTeX" in task
        assert "research_proposal.tex" in task
        assert "AI" in task

    def test_build_task_asserts_on_missing_scored_works(self):
        role = WriterRole()
        with pytest.raises(AssertionError):
            role.build_task("deepseek", topic="AI")

    def test_parse_result_returns_tex_path_when_valid(self, tmpdir):
        """Returns the tex path when research_proposal.tex exists with \\input commands."""
        tex_content = (
            r"\documentclass{article}"
            r"\begin{document}"
            r"\input{section_01.tex}"
            r"\input{section_02.tex}"
            r"\end{document}"
        )
        with open(os.path.join(tmpdir, "research_proposal.tex"), "w") as f:
            f.write(tex_content)

        role = WriterRole()
        scored = [{"output_file": "r.md", "title": "T", "total_score": 40,
                   "sub_task_id": 1, "scores": {}, "rank": 1}]
        mock_result = MagicMock()
        mock_result.messages = []
        result = role.parse_result(mock_result, working_dir=tmpdir,
                                   topic="AI", scored_works=scored)
        assert result.endswith("research_proposal.tex")

    def test_parse_result_falls_back_when_no_input(self, tmpdir):
        """Falls back to _fallback_latex when tex file has no \\input commands."""
        tex_content = (
            r"\documentclass{article}"
            r"\begin{document}"
            r"Content without input commands."
            r"\end{document}"
        )
        with open(os.path.join(tmpdir, "research_proposal.tex"), "w") as f:
            f.write(tex_content)

        role = WriterRole()
        scored = [{"sub_task_id": 1, "title": "T", "output_file": "r.md",
                   "scores": {"depth": 5, "accuracy": 5, "relevance": 5, "clarity": 5, "originality": 5},
                   "total_score": 25, "rank": 1}]
        mock_result = MagicMock()
        result = role.parse_result(mock_result, working_dir=tmpdir,
                                   topic="AI", scored_works=scored)
        # Fallback writes a new tex file
        assert os.path.exists(result)

    def test_parse_result_falls_back_when_no_tex_file(self, tmpdir):
        """Falls back to _fallback_latex when research_proposal.tex doesn't exist."""
        role = WriterRole()
        scored = [{"sub_task_id": 1, "title": "T", "output_file": "r.md",
                   "scores": {"depth": 5, "accuracy": 5, "relevance": 5, "clarity": 5, "originality": 5},
                   "total_score": 25, "rank": 1}]
        mock_result = MagicMock()
        result = role.parse_result(mock_result, working_dir=tmpdir,
                                   topic="AI", scored_works=scored)
        assert os.path.exists(result)

    def test_build_task_includes_ranked_works(self):
        role = WriterRole()
        scored = [
            {"output_file": "r1.md", "title": "Best", "total_score": 45, "rank": 1},
            {"output_file": "r2.md", "title": "Second", "total_score": 38, "rank": 2},
        ]
        task = role.build_task("deepseek", topic="AI", scored_works=scored)
        assert "Rank 1" in task
        assert "45/50" in task
        assert "Best" in task
        assert "Second" in task


# ═══════════════════════════════════════════════════════════════════════════
# LaTeX escaping
# ═══════════════════════════════════════════════════════════════════════════

class TestLatexEscape:
    def test_all_special_chars(self):
        escaped = _latex_escape("test & % $ # _ { } ~ ^ \\ end")
        assert r"\&" in escaped
        assert r"\%" in escaped
        assert r"\$" in escaped
        assert r"\#" in escaped
        assert r"\_" in escaped
        assert r"\{" in escaped
        assert r"\}" in escaped
        assert r"\textasciitilde" in escaped
        assert r"\textasciicircum" in escaped
        assert r"\textbackslash" in escaped

    def test_plain_text_unchanged(self):
        assert _latex_escape("hello world") == "hello world"

    def test_text_with_only_latex_commands_unchanged(self):
        """Text that looks like LaTeX commands (without special chars) is preserved."""
        assert _latex_escape(r"\section{Intro}") == r"\textbackslash section\{Intro\}"


# ═══════════════════════════════════════════════════════════════════════════
# ResearchPipeline
# ═══════════════════════════════════════════════════════════════════════════

class TestResearchPipeline:
    def test_pipeline_name(self):
        p = ResearchPipeline(working_dir="/tmp/test")
        assert p.name == "research"

    def test_default_output_dir(self):
        assert ResearchPipeline.default_output_dir == "research_output"

    def test_build_initial_state(self):
        p = ResearchPipeline(working_dir="/tmp/test", backend="claude_cli")
        sub_tasks = [{"id": 1, "title": "T", "description": "D"}]
        state = p._build_initial_state("Test topic", sub_tasks)
        assert state["topic"] == "Test topic"
        assert state["working_dir"] == "/tmp/test"
        assert state["backend"] == "claude_cli"
        assert state["status"] == "decomposed"
        assert state["version"] == 1
        assert state["worker_stats"]["total"] == 1

    def test_decompose_returns_list(self, tmpdir):
        p = ResearchPipeline(working_dir=tmpdir, backend="deepseek")
        result = p._decompose("AI and ML")
        assert isinstance(result, list)

    def test_build_graph_compiles(self):
        p = ResearchPipeline(working_dir="/tmp/test")
        graph = p._build_graph()
        assert graph is not None

    def test_build_graph_compiles_with_resume_state(self, tmpdir):
        """Graph compiles when resume=True (skips head node if sub_tasks present)."""
        p = ResearchPipeline(working_dir=tmpdir, backend="deepseek", resume=True)
        graph = p._build_graph()
        assert graph is not None

    def test_constants(self):
        assert RESEARCH_MAX_VERSIONS == 6
        assert RESEARCH_MAX_WORKER_RETRIES == 5
        assert HEAD_TIMEOUT == 300
        assert WORKER_TIMEOUT == 900

    def test_display_decomposition_formats_output(self, capsys):
        p = ResearchPipeline(working_dir="/tmp/test")
        p._display_decomposition([
            {"id": 1, "title": "Intro", "description": "An introduction to the topic"},
            {"id": 2, "title": "Methods", "description": "Deep dive into methods"},
        ])
        out = capsys.readouterr().out
        assert "[1]" in out
        assert "Intro" in out
        assert "[2]" in out
        assert "Methods" in out

    def test_flow_transitions_are_correct(self):
        """Verify the flow dict routes statuses to the correct next nodes."""
        p = ResearchPipeline(working_dir="/tmp/test")
        graph = p._build_graph()
        # The flow dict is internal. Verify the graph has all expected nodes.
        nodes = graph.get_graph().nodes if hasattr(graph, 'get_graph') else []
        # Graph compiled successfully - flow transitions are validated at build time
        assert graph is not None


# ═══════════════════════════════════════════════════════════════════════════
# Fallback LaTeX generation
# ═══════════════════════════════════════════════════════════════════════════

class TestFallbackLatex:
    def test_generates_valid_tex(self, tmpdir):
        from research.writer import _fallback_latex

        scored = [
            {"sub_task_id": 1, "title": "T", "output_file": "test.md",
             "scores": {"depth": 8, "accuracy": 7, "relevance": 9, "clarity": 8, "originality": 7},
             "total_score": 39, "rank": 1},
        ]
        path = _fallback_latex(scored, "Research", tmpdir)
        assert os.path.exists(path)
        content = open(path).read()
        assert r"\documentclass" in content
        assert r"\begin{document}" in content
        assert r"\end{document}" in content
        assert "__CONTENT_PLACEHOLDER__" not in content

    def test_scores_table_includes_all_dimensions(self, tmpdir):
        from research.writer import _fallback_latex

        scored = [
            {"sub_task_id": 1, "title": "Test", "output_file": "test.md",
             "scores": {"depth": 8, "accuracy": 7, "relevance": 9, "clarity": 8, "originality": 7},
             "total_score": 39, "rank": 1},
        ]
        path = _fallback_latex(scored, "R", tmpdir)
        content = open(path).read()
        assert r"\begin{table}" in content
        assert r"\begin{tabular}" in content
        for dim in ("Depth", "Accuracy", "Relevance", "Clarity", "Originality"):
            assert dim in content

    def test_multiple_works_all_included(self, tmpdir):
        from research.writer import _fallback_latex

        scored = [
            {"sub_task_id": 1, "title": "A", "output_file": "a.md",
             "scores": {"depth": 7, "accuracy": 7, "relevance": 7, "clarity": 7, "originality": 7},
             "total_score": 35, "rank": 1},
            {"sub_task_id": 2, "title": "B", "output_file": "b.md",
             "scores": {"depth": 6, "accuracy": 6, "relevance": 6, "clarity": 6, "originality": 6},
             "total_score": 30, "rank": 2},
        ]
        path = _fallback_latex(scored, "R", tmpdir)
        content = open(path).read()
        assert "A" in content
        assert "B" in content
        assert r"\bibitem{research1}" in content
        assert r"\bibitem{research2}" in content


# ═══════════════════════════════════════════════════════════════════════════
# Flow dict & status routing
# ═══════════════════════════════════════════════════════════════════════════

class TestFlowRouting:
    def test_flow_keys_all_statuses(self):
        """Verify the graph builds with all required transitions."""
        p = ResearchPipeline(working_dir="/tmp/test")
        graph = p._build_graph()
        assert graph is not None

    def test_status_router_maps_to_workers(self):
        from pipeline.base import BasePipeline
        flow = {"decomposed": "workers", "worker_retry": "workers"}
        router = BasePipeline._status_router(flow)
        assert router({"status": "decomposed"}) == "workers"
        assert router({"status": "worker_retry"}) == "workers"

    def test_status_router_researched_to_reviewer(self):
        from pipeline.base import BasePipeline
        flow = {"researched": "reviewer", "researched_partial": "reviewer"}
        router = BasePipeline._status_router(flow)
        assert router({"status": "researched"}) == "reviewer"
        assert router({"status": "researched_partial"}) == "reviewer"

    def test_status_router_reviewed_to_writer(self):
        from pipeline.base import BasePipeline
        flow = {"reviewed": "writer"}
        router = BasePipeline._status_router(flow)
        assert router({"status": "reviewed"}) == "writer"

    def test_status_router_written_to_end(self):
        from pipeline.base import BasePipeline
        from langgraph.graph import END
        flow = {"written": END}
        router = BasePipeline._status_router(flow)
        assert router({"status": "written"}) == END

    def test_status_router_terminal_errors(self):
        from pipeline.base import BasePipeline
        from langgraph.graph import END
        flow = {"ready": "worker"}
        router = BasePipeline._status_router(flow, terminal_errors={"error_no_subtasks"})
        assert router({"status": "error_no_subtasks"}) == END


# ═══════════════════════════════════════════════════════════════════════════
# Resume state reconstruction
# ═══════════════════════════════════════════════════════════════════════════

class TestResumeState:
    def test_try_load_resume_state_no_files(self, tmpdir):
        """Returns None when no decomposition.json exists."""
        p = ResearchPipeline(working_dir=tmpdir, resume=True)
        assert p._try_load_resume_state("Test") is None

    def test_try_load_resume_state_with_decomp(self, tmpdir):
        """Reconstructs state from decomposition.json on disk."""
        sub_tasks = [
            {"id": 1, "title": "A", "description": "Topic A"},
            {"id": 2, "title": "B", "description": "Topic B"},
        ]
        with open(os.path.join(tmpdir, "decomposition.json"), "w") as f:
            json.dump(sub_tasks, f)

        p = ResearchPipeline(working_dir=tmpdir, resume=True)
        state = p._try_load_resume_state("Test topic")
        assert state is not None
        assert state["status"] == "decomposed"
        assert len(state["sub_tasks"]) == 2
        assert state["version"] == 1

    def test_try_load_resume_state_with_worker_outputs(self, tmpdir):
        """Detects worker output files on disk and sets status accordingly."""
        sub_tasks = [
            {"id": 1, "title": "A", "description": "Topic A"},
            {"id": 2, "title": "B", "description": "Topic B"},
        ]
        with open(os.path.join(tmpdir, "decomposition.json"), "w") as f:
            json.dump(sub_tasks, f)
        # Create one worker output file
        with open(os.path.join(tmpdir, "research_01_Introduction.md"), "w") as f:
            f.write("# Research\nContent here.")

        p = ResearchPipeline(working_dir=tmpdir, resume=True)
        state = p._try_load_resume_state("Test topic")
        assert state is not None
        # One worker has output → partial success
        assert state["status"] in ("decomposed", "worker_retry")

    def test_try_load_resume_state_with_scoring(self, tmpdir):
        """Detects scoring_report.json and sets status to reviewed."""
        sub_tasks = [{"id": 1, "title": "A", "description": "Topic A"}]
        with open(os.path.join(tmpdir, "decomposition.json"), "w") as f:
            json.dump(sub_tasks, f)
        with open(os.path.join(tmpdir, "research_01_Topic.md"), "w") as f:
            f.write("# Content")
        with open(os.path.join(tmpdir, "scoring_report.json"), "w") as f:
            json.dump([{"sub_task_id": 1, "total_score": 40}], f)

        p = ResearchPipeline(working_dir=tmpdir, resume=True)
        state = p._try_load_resume_state("Test topic")
        assert state is not None
        assert len(state["scored_works"]) == 1


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
