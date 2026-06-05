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
    """TopologyState TypedDict has all required keys."""
    from pipeline import TopologyState

    required_keys = {
        "input_spec", "working_dir", "backend",
        "complexity_factors", "candidate_topologies",
        "evaluated_topologies", "topology_spec", "status",
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
    """_build_initial_state creates a dict with all required keys."""
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
    print("  PASS test_build_initial_state")


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
# Test 10: Fallback design (designer)
# ═══════════════════════════════════════════════════════════════════════════

def test_fallback_design():
    """_fallback_design returns at least 2 topologies with valid structure."""
    from topology.designer import TopologyDesignerRole

    result = TopologyDesignerRole._fallback_design({}, "Test task")

    assert isinstance(result, list), "Should return a list"
    assert len(result) >= 2, f"Should have at least 2 topologies, got {len(result)}"

    for t in result:
        assert "name" in t
        assert "pattern" in t
        assert t["pattern"] in ("sequential", "fan_out_fan_in", "debate_consensus", "hierarchical")
        assert "agents" in t
        assert len(t["agents"]) >= 2
        assert "connections" in t
        assert "parallelism_strategy" in t
        assert "strengths" in t
        assert "weaknesses" in t
    print(f"  PASS test_fallback_design (produced {len(result)} topologies)")


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

    # 2. Design (fallback)
    topologies = TopologyDesignerRole._fallback_design(factors, task)
    assert len(topologies) >= 2

    # 3. Evaluate (fallback)
    evaluated = TopologyEvaluatorRole._fallback_evaluate(topologies)
    assert len(evaluated) == len(topologies)
    best_score = evaluated[0]["total_score"]

    # 4. Generate spec and report (fallback)
    spec = TopologyWriterRole._generate_spec(evaluated, topologies, task)
    report = TopologyWriterRole._generate_report(evaluated, spec, task)

    assert spec["total_score"] == best_score
    assert spec["recommended_topology"] == evaluated[0]["name"]
    assert "pipeline_implementation_guide" in spec
    assert len(report) > 100

    print(f"  PASS test_e2e_fallback_chain (best: {spec['recommended_topology']}, {best_score}/50)")


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
        test_fallback_analyze,
        test_fallback_design,
        test_fallback_evaluate,
        test_fallback_writer_spec,
        test_fallback_writer_report,
        test_e2e_fallback_chain,
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
