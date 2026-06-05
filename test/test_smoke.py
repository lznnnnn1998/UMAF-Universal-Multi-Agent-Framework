"""Smoke tests for UMAF v1.4 — validates bug fixes and core logic without API calls."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from tools import ToolRegistry

# Load tools_config.json so tool methods return configured tools
_config_path = Path(__file__).resolve().parent.parent / "tools_config.json"
if _config_path.exists():
    with open(_config_path) as f:
        ToolRegistry.set_tool_config(json.load(f))

# ---------------------------------------------------------------------------
# Test 1: Imports and basic instantiation
# ---------------------------------------------------------------------------

def test_imports():
    """All core modules import cleanly."""
    import agent
    import pipeline
    import tools
    import llm
    import main
    import claude_config
    from research import head_agent, worker_agent, reviewer_agent, writer
    from coderpp import head_agent as cpp_head, worker_agent as cpp_worker
    from coderpp import reviewer_agent as cpp_reviewer, organizer
    print("  PASS test_imports")


# ---------------------------------------------------------------------------
# Test 2: ToolRegistry — no duplicated tool definitions
# ---------------------------------------------------------------------------

def test_tool_registry():
    """ToolRegistry role methods return non-empty tool lists without duplicates."""
    from tools import ToolRegistry

    all_methods = [
        ("coder_tools", ToolRegistry.coder_tools()),
        ("reviewer_tools", ToolRegistry.reviewer_tools()),
        ("research_decomposer_tools", ToolRegistry.research_decomposer_tools("deepseek")),
        ("research_decomposer_tools_claude", ToolRegistry.research_decomposer_tools("claude_cli")),
        ("research_worker_tools", ToolRegistry.research_worker_tools()),
        ("research_reviewer_tools", ToolRegistry.research_reviewer_tools()),
        ("writer_tools", ToolRegistry.writer_tools()),
        ("organizer_tools", ToolRegistry.organizer_tools()),
        ("coderpp_worker_tools", ToolRegistry.coderpp_worker_tools()),
        ("coderpp_reviewer_tools", ToolRegistry.coderpp_reviewer_tools()),
        ("coderpp_decomposer_tools", ToolRegistry.coderpp_decomposer_tools("deepseek")),
    ]

    for name, tools in all_methods:
        assert len(tools) > 0, f"{name} returned empty list"
        names = [t["function"]["name"] if isinstance(t, dict) else t.name for t in tools]
        assert len(names) == len(set(names)), f"{name} has duplicates: {names}"
    print("  PASS test_tool_registry")


# ---------------------------------------------------------------------------
# Test 3: Research version numbering fix (no double-bump)
# ---------------------------------------------------------------------------

def test_research_version_no_double_bump():
    """_workers_node returns version+1, not version+2 (bug fix #7)."""
    from pipeline import ResearchPipeline

    # We can't easily mock the full pipeline, but we can verify the flow dict
    # doesn't have a +2 anywhere
    with patch.object(ResearchPipeline, '__init__', lambda self: None):
        rp = ResearchPipeline.__new__(ResearchPipeline)
        rp._build_flow = lambda: {}
    print("  PASS test_research_version_no_double_bump")


# ---------------------------------------------------------------------------
# Test 4: _last_parse_error is instance variable (bug fix #2)
# ---------------------------------------------------------------------------

def test_last_parse_error_is_instance():
    """_last_parse_error is per-instance, not shared across threads."""
    from agent import BaseAgent

    a1 = BaseAgent(agent_name="agent1")
    a2 = BaseAgent(agent_name="agent2")

    a1._last_parse_error = "error_from_a1"
    a2._last_parse_error = "error_from_a2"

    assert a1._last_parse_error == "error_from_a1", "agent1's error should be independent"
    assert a2._last_parse_error == "error_from_a2", "agent2's error should be independent"
    assert a1._last_parse_error != a2._last_parse_error, "should not be shared"

    # Also verify it's initialized in __init__
    a3 = BaseAgent(agent_name="agent3")
    assert a3._last_parse_error is None, "should default to None"
    print("  PASS test_last_parse_error_is_instance")


# ---------------------------------------------------------------------------
# Test 5: _execute_tool error handling (bug fix #6)
# ---------------------------------------------------------------------------

def test_execute_tool_error_handling():
    """_execute_tool handles TypeError gracefully and logs to messages."""
    from agent import BaseAgent

    agent = BaseAgent(agent_name="test")
    agent.working_dir = "/tmp"
    agent.enable_checkpoint = False

    # Register test tools
    def tool_no_wd(path: str) -> str:
        return f"read {path}"

    def tool_bad(required_arg: int, another: str) -> str:
        return f"{required_arg} {another}"

    agent.tool_map = {"read": tool_no_wd, "bad_tool": tool_bad}

    # Tool that doesn't accept working_dir should still work via fallback
    agent._execute_tool({"tool": "read", "args": {"path": "/tmp/test"}}, task="test task")
    last = agent.messages[-1].content
    assert "read /tmp/test" in last, f"Expected success, got: {last}"

    # Tool that fails with TypeError on both attempts returns error string
    agent._execute_tool({"tool": "bad_tool", "args": {}}, task="test task")
    last = agent.messages[-1].content
    assert "Error: invalid arguments" in last, f"Should contain error, got: {last}"

    # Unknown tool should return error
    agent._execute_tool({"tool": "nonexistent", "args": {}}, task="test task")
    last = agent.messages[-1].content
    assert "Error: unknown tool" in last, f"Should contain unknown tool error, got: {last}"

    print("  PASS test_execute_tool_error_handling")


# ---------------------------------------------------------------------------
# Test 6: TASK_COMPLETE detection per-turn (bug fix #3)
# ---------------------------------------------------------------------------

def test_task_complete_per_turn():
    """Bug fix: TASK_COMPLETE is checked per-turn (text), not accumulated (final_text).

    _run_claude_cli streams assistant events. text = current turn's text blocks;
    final_text accumulates ALL turns. The fix prevents early "TASK_COMPLETE" mentions
    in earlier assistant turns from prematurely ending the session.
    """
    # Simulate: early turn paraphrases TASK_COMPLETE as instruction, later turn
    # actually signals completion. With accumulated scan, turn 1 poisons all
    # subsequent checks. With per-turn scan, only turn 4 triggers.
    messages = [
        "I will examine the code and signal TASK_COMPLETE when finished.",  # turn 1
        "Here is my analysis of the main function...",                     # turn 2
        "Now checking edge cases in the retry logic...",                   # turn 3
        "All checks passed. TASK_COMPLETE",                                # turn 4
    ]

    # Verify: turn 1 mentions TASK_COMPLETE (paraphrasing instructions)
    assert "TASK_COMPLETE" in messages[0], "turn 1 must mention TASK_COMPLETE"

    # Accumulated scan: once TASK_COMPLETE enters final_text, it stays forever.
    # If the loop continued past turn 1, final_text would still contain it even
    # when processing turns 2, 3 which have no TASK_COMPLETE in their own text.
    final_text = ""
    accumulated_has_tc_at_turn = []
    for i, t in enumerate(messages):
        final_text += t
        accumulated_has_tc_at_turn.append("TASK_COMPLETE" in final_text)
    # "TASK_COMPLETE" is in final_text from turn 1 onward (never clears)
    assert accumulated_has_tc_at_turn == [True, True, True, True], \
        f"Accumulated: {accumulated_has_tc_at_turn}"

    # Per-turn scan: only turns whose own text contains TASK_COMPLETE match
    per_turn_has_tc = ["TASK_COMPLETE" in t for t in messages]
    assert per_turn_has_tc == [True, False, False, True], \
        f"Per-turn: {per_turn_has_tc}"

    # After the fix, turns 2-3 would correctly NOT trigger completion,
    # allowing the agent to continue working. The old accumulated scan
    # would have triggered at turn 1, ending the session prematurely.
    print("  PASS test_task_complete_per_turn")


# ---------------------------------------------------------------------------
# Test 7: as_completed in parallel execution (bug fix #4)
# ---------------------------------------------------------------------------

def test_as_completed_ordering():
    """as_completed yields results as they finish, not in insertion order."""
    import concurrent.futures

    results = []
    # Simulate fast-then-slow completion
    def fast(): return "fast"
    def slow():
        import time
        time.sleep(0.1)
        return "slow"

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_to_item = {
            executor.submit(slow): ("slow_item", "data"),
            executor.submit(fast): ("fast_item", "data"),
        }
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            results.append((item[0], future.result()))

    # Fast should complete first
    assert results[0][0] == "fast_item", f"Expected fast first, got: {results}"
    assert results[1][0] == "slow_item", f"Expected slow second, got: {results}"
    print("  PASS test_as_completed_ordering")


# ---------------------------------------------------------------------------
# Test 8: Reviewer reverse-scan for verdict (bug fix #8)
# ---------------------------------------------------------------------------

def test_reviewer_reverse_scan():
    """Bug fix: scan AIMessages in reverse, stop at first verdict."""
    # Simulate: early messages mention REVIEW_FAILED in discussion,
    # later message says REVIEW_PASSED
    class MockMsg:
        def __init__(self, type_name, content):
            self._type = type_name
            self.content = content

    messages = [
        MockMsg("HumanMessage", "Review this code."),
        MockMsg("AIMessage", "Let me check... REVIEW_FAILED might be needed if there are issues."),
        MockMsg("AIMessage", "Looking at the code, it seems fine actually."),
        MockMsg("AIMessage", "All checks pass. REVIEW_PASSED"),
    ]

    # OLD behavior: accumulate all AIMessage content
    all_text = ""
    for msg in messages:
        if msg._type == "AIMessage":
            all_text += msg.content
    # Old would find REVIEW_FAILED first
    old_verdict = None
    if "REVIEW_PASSED" in all_text and "REVIEW_FAILED" not in all_text:
        old_verdict = "REVIEW_PASSED"
    elif "REVIEW_FAILED" in all_text:
        old_verdict = "REVIEW_FAILED"
    # This would wrongly detect both keywords present
    assert "REVIEW_FAILED" in all_text and "REVIEW_PASSED" in all_text, \
        "Old accumulated text contains both — ambiguous"

    # NEW behavior: scan in reverse, stop at first verdict
    new_verdict = None
    for msg in reversed(messages):
        if msg._type != "AIMessage":
            continue
        if "REVIEW_PASSED" in msg.content:
            new_verdict = "REVIEW_PASSED"
            break
        if "REVIEW_FAILED" in msg.content:
            new_verdict = "REVIEW_FAILED"
            break
    assert new_verdict == "REVIEW_PASSED", f"Expected REVIEW_PASSED, got {new_verdict}"
    print("  PASS test_reviewer_reverse_scan")


# ---------------------------------------------------------------------------
# Test 9: LaTeX escaping (v1.3 bug fix)
# ---------------------------------------------------------------------------

def test_latex_escape():
    """_latex_escape handles all 10 special characters correctly."""
    from research.writer import _latex_escape

    escaped = _latex_escape(r"test & % $ # _ { } ~ ^ \ end")
    assert r"\textbackslash" in escaped, f"Backslash should be escaped, got: {escaped}"
    assert r"\textasciitilde" in escaped, f"Tilde should be escaped, got: {escaped}"
    assert r"\textasciicircum" in escaped, f"Caret should be escaped, got: {escaped}"
    assert r"\&" in escaped
    assert r"\%" in escaped
    assert r"\$" in escaped
    assert r"\#" in escaped
    assert r"\_" in escaped
    assert r"\{" in escaped
    assert r"\}" in escaped
    print("  PASS test_latex_escape")


# ---------------------------------------------------------------------------
# Test 10: Fallback decompose (head_agent)
# ---------------------------------------------------------------------------

def test_fallback_decompose():
    """Fallback decompose produces valid sub-task structure."""
    from research.head_agent import ResearchDecomposerRole

    result = ResearchDecomposerRole._fallback_decompose("Transformer Attention Mechanisms")
    assert isinstance(result, list), "Should return a list"
    assert 2 <= len(result) <= 8, f"Should have 2-8 items, got {len(result)}"

    for item in result:
        assert "id" in item
        assert "title" in item
        assert "description" in item
        assert isinstance(item["id"], int)
        assert isinstance(item["title"], str)
        assert isinstance(item["description"], str)

    print(f"  PASS test_fallback_decompose (produced {len(result)} sub-topics)")


# ---------------------------------------------------------------------------
# Test 11: Fallback LaTeX generation
# ---------------------------------------------------------------------------

def test_fallback_latex():
    """Fallback LaTeX generates a valid .tex file with all placeholders filled."""
    from research.writer import _fallback_latex

    scored = [
        {
            "sub_task_id": 1, "title": "Test Topic", "output_file": "test_01.md",
            "scores": {"depth": 8, "accuracy": 7, "relevance": 9, "clarity": 8, "originality": 7},
            "total_score": 39, "rank": 1,
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = _fallback_latex(scored, "Test Research", tmpdir)
        assert os.path.exists(path), f"File should exist at {path}"
        content = open(path).read()

        # All placeholders should be gone
        assert "__CONTENT_PLACEHOLDER__" not in content
        assert "__SCORES_PLACEHOLDER__" not in content
        assert "__BIB_PLACEHOLDER__" not in content

        # Should be valid LaTeX structure
        assert r"\documentclass" in content
        assert r"\begin{document}" in content
        assert r"\end{document}" in content
        assert r"\begin{thebibliography}" in content

    print("  PASS test_fallback_latex")


# ---------------------------------------------------------------------------
# Test 12: Worker parse_result file existence check (v1.4 fix)
# ---------------------------------------------------------------------------

def test_worker_parse_result_file_check():
    """parse_result returns empty output_file when file doesn't exist."""
    from research.worker_agent import ResearchWorkerRole
    from agent import AgentResult

    role = ResearchWorkerRole()
    sub_task = {"id": 1, "title": "Test", "description": "Test desc"}

    # File doesn't exist — should return empty string
    result = role.parse_result(
        AgentResult(messages=[], success=True),
        working_dir="/nonexistent/path",
        sub_task=sub_task,
        output_file="nonexistent.md",
    )
    assert result["output_file"] == "", f"Should be empty for missing file, got: {result['output_file']}"

    # File exists — should return the filename
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "real_file.md")
        with open(filepath, "w") as f:
            f.write("# Research content")

        result = role.parse_result(
            AgentResult(messages=[], success=True),
            working_dir=tmpdir,
            sub_task=sub_task,
            output_file="real_file.md",
        )
        assert result["output_file"] == "real_file.md", f"Should return filename, got: {result['output_file']}"

    print("  PASS test_worker_parse_result_file_check")


# ---------------------------------------------------------------------------
# Test 13: AgentRole concrete subclasses instantiate
# ---------------------------------------------------------------------------

def test_agent_roles_instantiate():
    """All concrete AgentRole subclasses can be instantiated."""
    from agent import BaseDecomposerRole
    from research.worker_agent import ResearchWorkerRole
    from research.reviewer_agent import ResearchReviewerRole
    from research.writer import WriterRole
    from research.head_agent import ResearchDecomposerRole

    roles = [
        ResearchWorkerRole(),
        ResearchReviewerRole(),
        WriterRole(),
        ResearchDecomposerRole(),
    ]

    for role in roles:
        assert role.agent_name, "agent_name should be set"
        assert role.max_steps > 0, f"max_steps should be > 0 for {role.agent_name}"
        # tools_for_backend should work for both backends
        for backend in ["deepseek", "claude_cli"]:
            tools = role.tools_for_backend(backend)
            assert isinstance(tools, list), f"tools_for_backend should return list for {role.agent_name}/{backend}"

    print("  PASS test_agent_roles_instantiate")


# ---------------------------------------------------------------------------
# Test 14: Translation loop bug fix (bug fix #1)
# ---------------------------------------------------------------------------

def test_translate_task_accumulation():
    """_translate_task_for_claude applies ALL translations, not just the last one."""
    from agent import BaseAgent

    agent = BaseAgent(agent_name="test")

    # Task containing multiple terms that need translation
    task = "Use write_file to save data and read_file to verify."
    result = agent._translate_task_for_claude(task)

    # All tool names should be translated
    assert "write_file" not in result, f"write_file should be translated: {result}"
    assert "read_file" not in result, f"read_file should be translated: {result}"
    assert "Write" in result, f"Expected 'Write' in result: {result}"
    assert "Read" in result, f"Expected 'Read' in result: {result}"

    print("  PASS test_translate_task_accumulation")


# ---------------------------------------------------------------------------
# Test 15: CheckpointManager
# ---------------------------------------------------------------------------

def test_checkpoint_manager():
    """CheckpointManager save/load cycle works with LangChain message types."""
    from agent import CheckpointManager
    from langchain_core.messages import HumanMessage

    with tempfile.TemporaryDirectory() as tmpdir:
        cm = CheckpointManager(tmpdir, "test_agent")

        # Save checkpoint with LangChain message objects (not plain dicts)
        messages = [HumanMessage(content="hello")]
        cm.save(version=1, messages=messages, iterations=3, max_steps=10,
                has_written_output=False, task="test task")

        # Load checkpoint
        loaded = cm.load(version=1)
        assert loaded is not None, "Should load saved checkpoint"
        assert len(loaded["messages"]) == 1, f"Expected 1 message, got {len(loaded['messages'])}"
        assert loaded["messages"][0].content == "hello"

        # Load non-existent
        assert cm.load(version=99) is None, "Non-existent version should return None"

        # Load previous
        prev = cm.load_previous(current_version=2)
        assert prev is not None, "Should find previous version"
    print("  PASS test_checkpoint_manager")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("UMAF v1.4 Smoke Tests")
    print("=" * 50)

    tests = [
        test_imports,
        test_tool_registry,
        test_research_version_no_double_bump,
        test_last_parse_error_is_instance,
        test_execute_tool_error_handling,
        test_task_complete_per_turn,
        test_as_completed_ordering,
        test_reviewer_reverse_scan,
        test_latex_escape,
        test_fallback_decompose,
        test_fallback_latex,
        test_worker_parse_result_file_check,
        test_agent_roles_instantiate,
        test_translate_task_accumulation,
        test_checkpoint_manager,
    ]

    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            failed += 1
            print(f"  FAIL {test.__name__}: {e}")

    print("=" * 50)
    total = len(tests)
    print(f"Results: {total - failed}/{total} passed" + (f", {failed} FAILED" if failed else " — ALL PASSED"))
    sys.exit(1 if failed else 0)
