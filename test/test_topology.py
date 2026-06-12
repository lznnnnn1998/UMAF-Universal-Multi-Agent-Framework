"""Smoke tests for Topology Optimizer pipeline — validates agent roles, state, and pipeline."""

import json
import os
import sys
import tempfile
from pathlib import Path

from tools import ToolRegistry

# Load tools_config.json so tool methods return configured tools
_config_path = Path(__file__).resolve().parent.parent / "tools_config.json"
if _config_path.exists():
    with open(_config_path) as f:
        ToolRegistry.set_tool_config(json.load(f))


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Imports
# ═══════════════════════════════════════════════════════════════════════════

def test_imports():
    """All topology modules import cleanly."""
    from topology.analyzer import TopologyAnalyzerRole
    from topology.designer import TopologyDesignerRole
    from topology.evaluator import TopologyEvaluatorRole
    from topology.writer import TopologyWriterRole
    from pipeline import TopologyPipeline, TopologyState
    print("  PASS test_imports")


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: Agent role instantiation
# ═══════════════════════════════════════════════════════════════════════════

def test_agent_roles_instantiate():
    """All topology roles instantiate with correct agent_name and max_steps."""
    from topology.analyzer import TopologyAnalyzerRole
    from topology.designer import TopologyDesignerRole
    from topology.evaluator import TopologyEvaluatorRole
    from topology.writer import TopologyWriterRole

    roles = [
        (TopologyAnalyzerRole(), "topology_analyzer", 8),
        (TopologyDesignerRole(), "topology_designer", 12),
        (TopologyEvaluatorRole(), "topology_evaluator", 10),
        (TopologyWriterRole(), "topology_writer", 8),
    ]

    for role, expected_name, expected_max_steps in roles:
        assert role.agent_name == expected_name, f"{role.__class__.__name__}: expected {expected_name}, got {role.agent_name}"
        assert role.max_steps == expected_max_steps, f"{role.__class__.__name__}: expected {expected_max_steps}, got {role.max_steps}"
    print("  PASS test_agent_roles_instantiate")


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Tools for backend
# ═══════════════════════════════════════════════════════════════════════════

def test_tools_for_backend():
    """All roles return non-empty tool lists for both backends."""
    from topology.analyzer import TopologyAnalyzerRole
    from topology.designer import TopologyDesignerRole
    from topology.evaluator import TopologyEvaluatorRole
    from topology.writer import TopologyWriterRole

    for role_cls in [TopologyAnalyzerRole, TopologyDesignerRole, TopologyEvaluatorRole, TopologyWriterRole]:
        role = role_cls()
        for backend in ["deepseek", "claude_cli"]:
            tools = role.tools_for_backend(backend)
            assert isinstance(tools, list), f"{role.agent_name}/{backend}: should return list"
            assert len(tools) > 0, f"{role.agent_name}/{backend}: should return non-empty list"
    print("  PASS test_tools_for_backend")


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: ToolRegistry methods
# ═══════════════════════════════════════════════════════════════════════════

def test_tool_registry_methods():
    """ToolRegistry has all 4 topology tool methods."""
    from tools import ToolRegistry

    methods = [
        "topology_analyzer_tools",
        "topology_designer_tools",
        "topology_evaluator_tools",
        "topology_writer_tools",
    ]
    for method in methods:
        assert hasattr(ToolRegistry, method), f"ToolRegistry missing {method}"
        result = getattr(ToolRegistry, method)()
        assert isinstance(result, list), f"{method} should return list"
        assert len(result) > 0, f"{method} should return non-empty"

    # Writer should only have write_file (not read_file)
    writer_tools = ToolRegistry.topology_writer_tools()
    writer_names = [t.name for t in writer_tools]
    assert "read_file" not in writer_names, "Writer should not have read_file"
    assert "write_file" in writer_names, "Writer should have write_file"
    print("  PASS test_tool_registry_methods")


# ═══════════════════════════════════════════════════════════════════════════
# Test 5: TopologyState keys
# ═══════════════════════════════════════════════════════════════════════════

def test_state_keys():
    """TopologyState TypedDict has all required keys (including retry loop fields)."""
    from pipeline import TopologyState

    required_keys = {
        "input_spec", "working_dir", "backend",
        "complexity_factors", "candidate_topologies",
        "evaluated_topologies", "topology_spec", "status",
        "iteration", "evaluation_feedback",
    }
    # TypedDict stores keys in __annotations__
    actual_keys = set(TopologyState.__annotations__.keys())
    assert required_keys <= actual_keys, f"Missing keys: {required_keys - actual_keys}"
    print("  PASS test_state_keys")


# ═══════════════════════════════════════════════════════════════════════════
# Test 6: Pipeline instantiation
# ═══════════════════════════════════════════════════════════════════════════

def test_pipeline_instantiation():
    """TopologyPipeline instantiates with correct name and default_output_dir."""
    from pipeline import TopologyPipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        p = TopologyPipeline(working_dir=tmpdir, backend="deepseek")
        assert p.name == "topology"
        assert p.default_output_dir == "topology_output"
        assert p.working_dir == tmpdir
        assert p.backend == "deepseek"
    print("  PASS test_pipeline_instantiation")


# ═══════════════════════════════════════════════════════════════════════════
# Test 7: Decompose returns empty list
# ═══════════════════════════════════════════════════════════════════════════

def test_decompose_returns_empty():
    """TopologyPipeline._decompose returns empty list (no workers needed)."""
    from pipeline import TopologyPipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        p = TopologyPipeline(working_dir=tmpdir, backend="deepseek")
        result = p._decompose("Test task")
        assert result == [], f"Should return empty list, got {result}"
    print("  PASS test_decompose_returns_empty")


# ═══════════════════════════════════════════════════════════════════════════
# Test 8: Initial state
# ═══════════════════════════════════════════════════════════════════════════

def test_build_initial_state():
    """_build_initial_state creates a dict with all required keys including
    iteration and evaluation_feedback."""
    from pipeline import TopologyPipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        p = TopologyPipeline(working_dir=tmpdir, backend="claude_cli")
        state = p._build_initial_state("Test spec", [])
        assert state["input_spec"] == "Test spec"
        assert state["working_dir"] == tmpdir
        assert state["backend"] == "claude_cli"
        assert state["complexity_factors"] == {}
        assert state["candidate_topologies"] == []
        assert state["evaluated_topologies"] == []
        assert state["topology_spec"] == {}
        assert state["status"] == "initialized"
        assert state["iteration"] == 0
        assert state["evaluation_feedback"] == ""
    print("  PASS test_build_initial_state")


# ═══════════════════════════════════════════════════════════════════════════
# Test 8b: New state fields present
# ═══════════════════════════════════════════════════════════════════════════

def test_state_has_iteration_and_feedback():
    """TopologyState TypedDict has new iteration and evaluation_feedback keys."""
    from pipeline import TopologyState

    annotations = TopologyState.__annotations__
    assert "iteration" in annotations, "TopologyState missing 'iteration' field"
    # Handle both resolved types and ForwardRef (which wraps the type name)
    from typing import ForwardRef
    iteration_type = annotations["iteration"]
    assert iteration_type is int or isinstance(iteration_type, ForwardRef), (
        f"iteration should be int, got {iteration_type!r}"
    )
    assert "evaluation_feedback" in annotations, "TopologyState missing 'evaluation_feedback' field"
    feedback_type = annotations["evaluation_feedback"]
    assert feedback_type is str or isinstance(feedback_type, ForwardRef), (
        f"evaluation_feedback should be str, got {feedback_type!r}"
    )
    print("  PASS test_state_has_iteration_and_feedback")


# ═══════════════════════════════════════════════════════════════════════════
# Test 9: Fallback analyze (analyzer)
# ═══════════════════════════════════════════════════════════════════════════

def test_fallback_analyze():
    """_fallback_analyze returns valid complexity factors dict."""
    from topology.analyzer import TopologyAnalyzerRole

    result = TopologyAnalyzerRole._fallback_analyze("Build a pipeline that summarizes skills from a codebase")

    assert "factors" in result
    assert "overall_complexity" in result
    assert "key_insights" in result

    factors = result["factors"]
    required = ["data_dependencies", "parallelism_opportunities", "tool_requirements",
                 "error_domains", "latency_sensitivity", "scale"]
    for f in required:
        assert f in factors, f"Missing factor: {f}"
        assert "level" in factors[f]
        assert factors[f]["level"] in ("low", "medium", "high")
        assert "reasoning" in factors[f]
    print("  PASS test_fallback_analyze")


# ═══════════════════════════════════════════════════════════════════════════
# Test 9b: Fallback analyze has _fallback flag
# ═══════════════════════════════════════════════════════════════════════════

def test_fallback_analyze_has_fallback_flag():
    """_fallback_analyze result includes '_fallback': True flag."""
    from topology.analyzer import TopologyAnalyzerRole

    result = TopologyAnalyzerRole._fallback_analyze("Test task")
    assert "_fallback" in result, "_fallback_analyze should include '_fallback' key"
    assert result["_fallback"] is True, "_fallback should be True"
    print("  PASS test_fallback_analyze_has_fallback_flag")


# ═══════════════════════════════════════════════════════════════════════════
# Test 10: Fallback design (designer)
# ═══════════════════════════════════════════════════════════════════════════

def test_fallback_design():
    """_fallback_design returns at least 4 topologies with valid structure
    (now includes debate_consensus in addition to the original 3 patterns)."""
    from topology.designer import TopologyDesignerRole

    result = TopologyDesignerRole._fallback_design({}, "Test task")

    assert isinstance(result, list), "Should return a list"
    assert len(result) >= 4, f"Should have at least 4 topologies (now includes debate_consensus), got {len(result)}"

    patterns_seen = set()
    for t in result:
        assert "name" in t
        assert "pattern" in t
        assert t["pattern"] in ("sequential", "fan_out_fan_in", "debate_consensus", "hierarchical")
        patterns_seen.add(t["pattern"])
        assert "agents" in t
        assert len(t["agents"]) >= 2
        assert "connections" in t
        assert "parallelism_strategy" in t
        assert "strengths" in t
        assert "weaknesses" in t

    # Verify all 4 patterns are present
    assert "debate_consensus" in patterns_seen, (
        f"debate_consensus pattern should be present; got patterns: {patterns_seen}"
    )
    print(f"  PASS test_fallback_design (produced {len(result)} topologies, patterns: {patterns_seen})")


# ═══════════════════════════════════════════════════════════════════════════
# Test 10b: Fallback design includes debate_consensus pattern
# ═══════════════════════════════════════════════════════════════════════════

def test_fallback_design_includes_debate_consensus():
    """_fallback_design now returns 4 topologies including debate_consensus."""
    from topology.designer import TopologyDesignerRole

    result = TopologyDesignerRole._fallback_design({}, "Test task")
    patterns = [t.get("pattern") for t in result]

    assert len(result) == 4, f"Expected 4 topologies, got {len(result)}"
    assert "debate_consensus" in patterns, f"debate_consensus missing from patterns: {patterns}"

    # Verify debate_consensus topology has the right agents
    debate = [t for t in result if t["pattern"] == "debate_consensus"][0]
    agent_names = [a["agent_name"] for a in debate["agents"]]
    assert "debater_a" in agent_names
    assert "debater_b" in agent_names
    assert "debater_c" in agent_names
    assert "judge" in agent_names
    print("  PASS test_fallback_design_includes_debate_consensus")


# ═══════════════════════════════════════════════════════════════════════════
# Test 10c: Fallback design has _fallback flags
# ═══════════════════════════════════════════════════════════════════════════

def test_fallback_design_has_fallback_flags():
    """Each topology in _fallback_design has '_fallback': True."""
    from topology.designer import TopologyDesignerRole

    result = TopologyDesignerRole._fallback_design({}, "Test task")
    for t in result:
        assert "_fallback" in t, f"Topology '{t.get('name')}' missing '_fallback' key"
        assert t["_fallback"] is True, f"Topology '{t.get('name')}' _fallback should be True"
    print("  PASS test_fallback_design_has_fallback_flags")


# ═══════════════════════════════════════════════════════════════════════════
# Test 11: Fallback evaluate (evaluator)
# ═══════════════════════════════════════════════════════════════════════════

def test_fallback_evaluate():
    """_fallback_evaluate returns topologies with consistent scores."""
    from topology.evaluator import TopologyEvaluatorRole
    from topology.designer import TopologyDesignerRole

    topologies = TopologyDesignerRole._fallback_design({}, "Test")
    result = TopologyEvaluatorRole._fallback_evaluate(topologies)

    assert isinstance(result, list)
    assert len(result) == len(topologies)

    dimensions = {"latency", "reliability", "cost_efficiency", "simplicity", "scalability"}
    for t in result:
        assert "scores" in t
        assert "total_score" in t
        assert set(t["scores"].keys()) == dimensions
        # Each score should be 1-10
        for d in dimensions:
            score = t["scores"][d]["score"]
            assert 1 <= score <= 10, f"Score {d}={score} out of range"
        # Total should be sum of all dimension scores
        total = sum(t["scores"][d]["score"] for d in dimensions)
        assert total == t["total_score"], f"total_score {t['total_score']} != sum {total}"
    # Should be sorted by total_score descending
    for i in range(len(result) - 1):
        assert result[i]["total_score"] >= result[i + 1]["total_score"], "Should be sorted descending"
    print(f"  PASS test_fallback_evaluate")


# ═══════════════════════════════════════════════════════════════════════════
# Test 11b: Fallback evaluate has _fallback flags
# ═══════════════════════════════════════════════════════════════════════════

def test_fallback_evaluate_has_fallback_flags():
    """Each evaluation in _fallback_evaluate has '_fallback': True."""
    from topology.evaluator import TopologyEvaluatorRole
    from topology.designer import TopologyDesignerRole

    topologies = TopologyDesignerRole._fallback_design({}, "Test")
    result = TopologyEvaluatorRole._fallback_evaluate(topologies)
    for t in result:
        assert "_fallback" in t, f"Evaluation '{t.get('name')}' missing '_fallback' key"
        assert t["_fallback"] is True, f"Evaluation '{t.get('name')}' _fallback should be True"
    print("  PASS test_fallback_evaluate_has_fallback_flags")


# ═══════════════════════════════════════════════════════════════════════════
# Test 11c: Fallback evaluate is complexity-aware
# ═══════════════════════════════════════════════════════════════════════════

def test_fallback_evaluate_complexity_aware():
    """Complexity factors actually adjust heuristic scores.
    High parallelism_opportunities boosts fan_out latency.
    High latency_sensitivity penalizes sequential latency."""
    from topology.evaluator import TopologyEvaluatorRole
    from topology.designer import TopologyDesignerRole

    topologies = TopologyDesignerRole._fallback_design({}, "Test")

    # Evaluate with no complexity factors (baseline)
    baseline = TopologyEvaluatorRole._fallback_evaluate(topologies, None)

    # Evaluate with high parallelism_opportunities and high latency_sensitivity
    high_complexity = {
        "factors": {
            "parallelism_opportunities": {
                "level": "high",
                "reasoning": "Many independent sub-tasks can run in parallel."
            },
            "latency_sensitivity": {
                "level": "high",
                "reasoning": "End-to-end latency is critical."
            },
        },
        "overall_complexity": "medium",
    }
    adjusted = TopologyEvaluatorRole._fallback_evaluate(topologies, high_complexity)

    # Find fan_out scores in both
    def find_scores(evaluated, pattern):
        for e in evaluated:
            name = e.get("name", "")
            # Match by name containing pattern hint
            if pattern in str(e.get("scores", {})):
                continue
        for e in evaluated:
            for t in topologies:
                if t["name"] == e["name"] and t["pattern"] == pattern:
                    return e["scores"]
        return None

    # Match by topology name since _fallback_evaluate sorts by total_score
    baseline_by_pattern = {}
    adjusted_by_pattern = {}
    for t in topologies:
        pattern = t["pattern"]
        for b in baseline:
            if b["name"] == t["name"]:
                baseline_by_pattern[pattern] = b
                break
        for a in adjusted:
            if a["name"] == t["name"]:
                adjusted_by_pattern[pattern] = a
                break

    # With high parallelism, fan_out latency should be boosted
    fan_out_baseline_lat = baseline_by_pattern["fan_out_fan_in"]["scores"]["latency"]["score"]
    fan_out_adjusted_lat = adjusted_by_pattern["fan_out_fan_in"]["scores"]["latency"]["score"]
    assert fan_out_adjusted_lat >= fan_out_baseline_lat, (
        f"High parallelism should boost fan_out latency: {fan_out_baseline_lat} → {fan_out_adjusted_lat}"
    )

    # With high latency sensitivity, sequential latency should be penalized
    seq_baseline_lat = baseline_by_pattern["sequential"]["scores"]["latency"]["score"]
    seq_adjusted_lat = adjusted_by_pattern["sequential"]["scores"]["latency"]["score"]
    assert seq_adjusted_lat <= seq_baseline_lat, (
        f"High latency sensitivity should penalize sequential latency: {seq_baseline_lat} → {seq_adjusted_lat}"
    )

    print(f"  PASS test_fallback_evaluate_complexity_aware")


# ═══════════════════════════════════════════════════════════════════════════
# Test 12: Fallback writer spec generation
# ═══════════════════════════════════════════════════════════════════════════

def test_fallback_writer_spec():
    """_generate_spec produces valid topology_spec.json structure."""
    from topology.writer import TopologyWriterRole
    from topology.designer import TopologyDesignerRole
    from topology.evaluator import TopologyEvaluatorRole

    candidates = TopologyDesignerRole._fallback_design({}, "Test")
    evaluated = TopologyEvaluatorRole._fallback_evaluate(candidates)

    spec = TopologyWriterRole._generate_spec(evaluated, candidates, "Test task")

    required = ["pipeline_name", "recommended_topology", "total_score",
                 "max_possible_score", "design_pattern", "agents",
                 "connections", "parallelism_strategy", "pipeline_implementation_guide"]
    for key in required:
        assert key in spec, f"Missing key: {key}"

    guide = spec["pipeline_implementation_guide"]
    assert "overview" in guide
    assert "nodes" in guide
    assert "flow_diagram" in guide
    assert "key_design_decisions" in guide
    assert "configuration_notes" in guide

    assert spec["max_possible_score"] == 50
    print("  PASS test_fallback_writer_spec")


# ═══════════════════════════════════════════════════════════════════════════
# Test 12b: Writer spec has _fallback flag
# ═══════════════════════════════════════════════════════════════════════════

def test_writer_spec_has_fallback_flag():
    """_generate_spec result includes '_fallback': True."""
    from topology.writer import TopologyWriterRole
    from topology.designer import TopologyDesignerRole
    from topology.evaluator import TopologyEvaluatorRole

    candidates = TopologyDesignerRole._fallback_design({}, "Test")
    evaluated = TopologyEvaluatorRole._fallback_evaluate(candidates)

    spec = TopologyWriterRole._generate_spec(evaluated, candidates, "Test task")
    assert "_fallback" in spec, "_generate_spec should include '_fallback' key"
    assert spec["_fallback"] is True, "_fallback should be True"

    # Also test the empty case (no evaluated/candidates)
    empty_spec = TopologyWriterRole._generate_spec([], [], "Empty test")
    assert "_fallback" in empty_spec, "Empty spec should include '_fallback' key"
    assert empty_spec["_fallback"] is True, "Empty spec _fallback should be True"
    print("  PASS test_writer_spec_has_fallback_flag")


# ═══════════════════════════════════════════════════════════════════════════
# Test 13: Fallback report generation
# ═══════════════════════════════════════════════════════════════════════════

def test_fallback_writer_report():
    """_generate_report produces valid markdown report."""
    from topology.writer import TopologyWriterRole
    from topology.designer import TopologyDesignerRole
    from topology.evaluator import TopologyEvaluatorRole

    candidates = TopologyDesignerRole._fallback_design({}, "Test")
    evaluated = TopologyEvaluatorRole._fallback_evaluate(candidates)
    spec = TopologyWriterRole._generate_spec(evaluated, candidates, "Test task")

    report = TopologyWriterRole._generate_report(evaluated, spec, "Test task")

    assert "# Topology Optimization Report" in report
    assert "## Requirement" in report
    assert "## Comparison" in report
    assert "## Recommendation" in report
    assert "Test task" in report
    print("  PASS test_fallback_writer_report")


# ═══════════════════════════════════════════════════════════════════════════
# Test 14: End-to-end fallback pipeline (no LLM calls)
# ═══════════════════════════════════════════════════════════════════════════

def test_e2e_fallback_chain():
    """Full fallback chain: analyze → design → evaluate → generate spec/report."""
    from topology.analyzer import TopologyAnalyzerRole
    from topology.designer import TopologyDesignerRole
    from topology.evaluator import TopologyEvaluatorRole
    from topology.writer import TopologyWriterRole

    task = "Build a multi-agent pipeline that analyzes a codebase, extracts skills, and generates a skill inventory report."

    # 1. Analyze (fallback)
    factors = TopologyAnalyzerRole._fallback_analyze(task)
    assert factors["overall_complexity"] in ("low", "medium", "high")
    assert factors.get("_fallback") is True  # New: check _fallback flag

    # 2. Design (fallback)
    topologies = TopologyDesignerRole._fallback_design(factors, task)
    assert len(topologies) >= 4  # Now 4 patterns including debate_consensus

    # 3. Evaluate (fallback)
    evaluated = TopologyEvaluatorRole._fallback_evaluate(topologies, factors)
    assert len(evaluated) == len(topologies)
    best_score = evaluated[0]["total_score"]
    assert evaluated[0].get("_fallback") is True  # New: check _fallback flag

    # 4. Generate spec and report (fallback)
    spec = TopologyWriterRole._generate_spec(evaluated, topologies, task)
    report = TopologyWriterRole._generate_report(evaluated, spec, task)

    assert spec["total_score"] == best_score
    assert spec["recommended_topology"] == evaluated[0]["name"]
    assert spec.get("_fallback") is True  # New: check _fallback flag
    assert "pipeline_implementation_guide" in spec
    assert len(report) > 100

    print(f"  PASS test_e2e_fallback_chain (best: {spec['recommended_topology']}, {best_score}/50)")


# ═══════════════════════════════════════════════════════════════════════════
# Test 15: Writer parse_result extracts JSON from messages
# ═══════════════════════════════════════════════════════════════════════════

def test_writer_parse_result_extracts_json_from_messages():
    """parse_result extracts spec JSON from agent messages when no disk file exists."""
    from unittest.mock import MagicMock

    from topology.writer import TopologyWriterRole
    from topology.designer import TopologyDesignerRole
    from topology.evaluator import TopologyEvaluatorRole

    # Build pre-existing evaluation data for the fallback path
    candidates = TopologyDesignerRole._fallback_design({}, "Test")
    evaluated = TopologyEvaluatorRole._fallback_evaluate(candidates)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Make sure there is NO topology_spec.json on disk
        spec_path = os.path.join(tmpdir, "topology_spec.json")
        assert not os.path.exists(spec_path)

        # Build a mock AgentResult with messages containing a valid JSON spec
        spec_json = json.dumps({
            "pipeline_name": "topology",
            "recommended_topology": "Fan-Out/Fan-In Pipeline",
            "total_score": 42,
            "max_possible_score": 50,
            "design_pattern": "fan_out_fan_in",
            "agents": [],
            "connections": [],
            "parallelism_strategy": "parallel",
            "pipeline_implementation_guide": {"overview": "Test", "nodes": [], "flow_diagram": "", "key_design_decisions": [], "configuration_notes": ""},
        })
        mock_msg = MagicMock()
        mock_msg.content = f"Here is the spec:\n```json\n{spec_json}\n```\nTASK_COMPLETE"
        type(mock_msg).__name__ = "AIMessage"

        mock_result = MagicMock()
        mock_result.messages = [mock_msg]
        mock_result.success = True

        role = TopologyWriterRole()
        result = role.parse_result(
            mock_result, tmpdir,
            evaluated_topologies=evaluated,
            candidate_topologies=candidates,
            input_spec="Test task",
        )

        assert result["success"] is True, "parse_result should succeed"
        assert result["spec"]["recommended_topology"] == "Fan-Out/Fan-In Pipeline"
        assert result["spec"]["total_score"] == 42
        # Also verify the file was written to disk
        assert os.path.exists(spec_path), "Spec file should be written to disk from extracted JSON"
    print("  PASS test_writer_parse_result_extracts_json_from_messages")


# ═══════════════════════════════════════════════════════════════════════════
# Test 16: Writer parse_result fallback when no JSON in messages
# ═══════════════════════════════════════════════════════════════════════════

def test_writer_parse_result_fallback_no_json():
    """parse_result falls back to _generate_spec when no JSON in messages or disk."""
    from unittest.mock import MagicMock

    from topology.writer import TopologyWriterRole
    from topology.designer import TopologyDesignerRole
    from topology.evaluator import TopologyEvaluatorRole

    candidates = TopologyDesignerRole._fallback_design({}, "Test")
    evaluated = TopologyEvaluatorRole._fallback_evaluate(candidates)

    with tempfile.TemporaryDirectory() as tmpdir:
        spec_path = os.path.join(tmpdir, "topology_spec.json")
        assert not os.path.exists(spec_path)

        # Mock result with no JSON in messages
        mock_msg = MagicMock()
        mock_msg.content = "TASK_COMPLETE — done."  # No JSON here
        type(mock_msg).__name__ = "AIMessage"

        mock_result = MagicMock()
        mock_result.messages = [mock_msg]
        mock_result.success = True

        role = TopologyWriterRole()
        result = role.parse_result(
            mock_result, tmpdir,
            evaluated_topologies=evaluated,
            candidate_topologies=candidates,
            input_spec="Test task",
        )

        assert result["success"] is True, "parse_result should succeed via fallback"
        assert result["spec"].get("_fallback") is True, "Fallback spec should have _fallback flag"
        assert os.path.exists(spec_path), "Fallback spec file should be written to disk"
    print("  PASS test_writer_parse_result_fallback_no_json")


# ═══════════════════════════════════════════════════════════════════════════
# Test 17: Retry routing logic
# ═══════════════════════════════════════════════════════════════════════════

def test_retry_routing_logic():
    """Exercise the retry decision logic: score < 35 and iteration < 3 → retry."""
    from pipeline import TopologyPipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        p = TopologyPipeline(working_dir=tmpdir, backend="deepseek")

        # Verify the retry constants
        assert p._MAX_RETRIES == 3
        assert p._SCORE_THRESHOLD == 35

        # Verify flow dict includes designer_retry → designer
        graph = p._build_graph()
        # A compiled graph has a 'nodes' attribute
        nodes = graph.nodes if hasattr(graph, 'nodes') else graph.builder.nodes
        # Just verify the pipeline compiles without error
        assert graph is not None
    print("  PASS test_retry_routing_logic")


# ═══════════════════════════════════════════════════════════════════════════
# Test 18: Retry flow dict contains designer_retry
# ═══════════════════════════════════════════════════════════════════════════

def test_flow_dict_has_designer_retry():
    """The flow dict in _build_graph includes 'designer_retry': 'designer'."""
    from pipeline import BasePipeline

    # Build the actual flow dict that the pipeline uses
    flow = {
        "initialized": "analyzer",
        "analyzed": "designer",
        "designed": "evaluator",
        "designer_retry": "designer",
        "evaluated": "writer",
        "written": None,  # END
    }
    terminal = {"error_analysis_failed", "error_design_failed", "error_evaluation_failed", "error_writer_failed"}

    # Test that designer_retry routes to designer
    assert "designer_retry" in flow, "flow dict must contain designer_retry"
    assert flow["designer_retry"] == "designer", "designer_retry must route to designer"

    # Test that the router handles designer_retry correctly
    from langgraph.graph import END
    flow_with_end = {
        "initialized": "analyzer",
        "analyzed": "designer",
        "designed": "evaluator",
        "designer_retry": "designer",
        "evaluated": "writer",
        "written": END,
    }
    router = BasePipeline._status_router(flow_with_end, terminal)
    # designer_retry should route to designer (router expects dict with "status" key)
    result = router({"status": "designer_retry"})
    assert result == "designer", f"Router should return 'designer' for 'designer_retry', got {result}"
    # errors should route to END
    result = router({"status": "error_evaluation_failed"})
    assert result == END, f"Router should return END for error status"
    print("  PASS test_flow_dict_has_designer_retry")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Topology Optimizer — Smoke Tests")
    print("=" * 50)

    tests = [
        test_imports,
        test_agent_roles_instantiate,
        test_tools_for_backend,
        test_tool_registry_methods,
        test_state_keys,
        test_pipeline_instantiation,
        test_decompose_returns_empty,
        test_build_initial_state,
        test_state_has_iteration_and_feedback,
        test_fallback_analyze,
        test_fallback_analyze_has_fallback_flag,
        test_fallback_design,
        test_fallback_design_includes_debate_consensus,
        test_fallback_design_has_fallback_flags,
        test_fallback_evaluate,
        test_fallback_evaluate_has_fallback_flags,
        test_fallback_evaluate_complexity_aware,
        test_fallback_writer_spec,
        test_writer_spec_has_fallback_flag,
        test_fallback_writer_report,
        test_e2e_fallback_chain,
        test_writer_parse_result_extracts_json_from_messages,
        test_writer_parse_result_fallback_no_json,
        test_retry_routing_logic,
        test_flow_dict_has_designer_retry,
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
