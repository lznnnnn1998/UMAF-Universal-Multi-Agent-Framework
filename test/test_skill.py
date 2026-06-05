"""Smoke tests for Skill Summarizer Pipeline v2 — artifact-agnostic skill detection."""

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
    from skill.scanner import SkillScannerRole
    from skill.detectors import (DomainExpertiseDetectorRole, TechnicalCraftDetectorRole,
                                  MethodologyDetectorRole, RigorDetectorRole)
    from skill.aggregator import SkillAggregatorRole
    from skill.writer import SkillReportWriterRole
    from pipeline import SkillPipeline, SkillState
    print("  PASS test_imports")


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: Agent role instantiation
# ═══════════════════════════════════════════════════════════════════════════

def test_agent_roles_instantiate():
    from skill.scanner import SkillScannerRole
    from skill.detectors import (DomainExpertiseDetectorRole, TechnicalCraftDetectorRole,
                                  MethodologyDetectorRole, RigorDetectorRole)
    from skill.aggregator import SkillAggregatorRole
    from skill.writer import SkillReportWriterRole

    roles = [
        (SkillScannerRole(), "skill_scanner", 15),
        (DomainExpertiseDetectorRole(), "domain_expertise_detector", 12),
        (TechnicalCraftDetectorRole(), "technical_craft_detector", 12),
        (MethodologyDetectorRole(), "methodology_detector", 12),
        (RigorDetectorRole(), "rigor_detector", 12),
        (SkillAggregatorRole(), "skill_aggregator", 10),
        (SkillReportWriterRole(), "skill_report_writer", 8),
    ]
    for role, expected_name, expected_max_steps in roles:
        assert role.agent_name == expected_name, f"{role.__class__.__name__}: expected {expected_name}, got {role.agent_name}"
        assert role.max_steps == expected_max_steps, f"{role.__class__.__name__}: expected {expected_max_steps}, got {role.max_steps}"
    print("  PASS test_agent_roles_instantiate")


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Tools for backend
# ═══════════════════════════════════════════════════════════════════════════

def test_tools_for_backend():
    from skill.scanner import SkillScannerRole
    from skill.detectors import (DomainExpertiseDetectorRole, TechnicalCraftDetectorRole,
                                  MethodologyDetectorRole, RigorDetectorRole)
    from skill.aggregator import SkillAggregatorRole
    from skill.writer import SkillReportWriterRole

    for role_cls in [SkillScannerRole, DomainExpertiseDetectorRole,
                     TechnicalCraftDetectorRole, MethodologyDetectorRole,
                     RigorDetectorRole, SkillAggregatorRole, SkillReportWriterRole]:
        role = role_cls()
        for backend in ["deepseek", "claude_cli"]:
            tools = role.tools_for_backend(backend)
            assert isinstance(tools, list), f"{role.agent_name}/{backend}: should return list"
            assert len(tools) > 0, f"{role.agent_name}/{backend}: should return non-empty list"
    print("  PASS test_tools_for_backend")


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: SkillState keys
# ═══════════════════════════════════════════════════════════════════════════

def test_state_keys():
    from pipeline import SkillState

    required_keys = {
        "input_spec", "working_dir", "backend",
        "project_scan", "artifact_analysis", "detector_outputs",
        "skill_inventory", "status",
    }
    actual_keys = set(SkillState.__annotations__.keys())
    assert required_keys <= actual_keys, f"Missing keys: {required_keys - actual_keys}"
    print("  PASS test_state_keys")


# ═══════════════════════════════════════════════════════════════════════════
# Test 5: Pipeline instantiation
# ═══════════════════════════════════════════════════════════════════════════

def test_pipeline_instantiation():
    from pipeline import SkillPipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        p = SkillPipeline(working_dir=tmpdir, backend="claude_cli")
        assert p.name == "skill"
        assert p.default_output_dir == "skill_output"
        assert p.working_dir == tmpdir
    print("  PASS test_pipeline_instantiation")


# ═══════════════════════════════════════════════════════════════════════════
# Test 6: Decompose returns empty list
# ═══════════════════════════════════════════════════════════════════════════

def test_decompose_returns_empty():
    from pipeline import SkillPipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        p = SkillPipeline(working_dir=tmpdir, backend="deepseek")
        result = p._decompose("Test project path")
        assert result == []
    print("  PASS test_decompose_returns_empty")


# ═══════════════════════════════════════════════════════════════════════════
# Test 7: Initial state
# ═══════════════════════════════════════════════════════════════════════════

def test_build_initial_state():
    from pipeline import SkillPipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        p = SkillPipeline(working_dir=tmpdir, backend="claude_cli")
        state = p._build_initial_state("/path/to/project", [])
        assert state["input_spec"] == "/path/to/project"
        assert state["working_dir"] == tmpdir
        assert state["backend"] == "claude_cli"
        assert state["project_scan"] == {}
        assert state["artifact_analysis"] == {}
        assert state["detector_outputs"] == []
        assert state["skill_inventory"] == {}
        assert state["status"] == "initialized"
    print("  PASS test_build_initial_state")


# ═══════════════════════════════════════════════════════════════════════════
# Test 8: Fallback scanner — surface scan (backward compatible)
# ═══════════════════════════════════════════════════════════════════════════

def test_fallback_scanner():
    """_fallback_surface_scan produces a valid project scan dict."""
    from skill.scanner import SkillScannerRole

    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "src"))
        with open(os.path.join(tmpdir, "src", "main.py"), "w") as f:
            f.write("print('hello')")
        with open(os.path.join(tmpdir, "README.md"), "w") as f:
            f.write("# Test")

        scan = SkillScannerRole._fallback_surface_scan(project_dir=tmpdir, working_dir=tmpdir)
        assert "file_categories" in scan
        assert "total_files" in scan
        assert scan["total_files"] >= 2
        assert "source" in scan["file_categories"]
        print(f"  PASS test_fallback_scanner ({scan['total_files']} files found)")


# ═══════════════════════════════════════════════════════════════════════════
# Test 9: Deep scan fallback
# ═══════════════════════════════════════════════════════════════════════════

def test_deep_scanner():
    """_fallback_deep_scanner produces artifact analysis with classification."""
    from skill.scanner import SkillScannerRole

    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "src"))
        with open(os.path.join(tmpdir, "src", "main.py"), "w") as f:
            f.write("import numpy as np\n\ndef train_model(data):\n    return np.mean(data)\n")
        with open(os.path.join(tmpdir, "README.md"), "w") as f:
            f.write("# ML Project\nA machine learning project.")
        with open(os.path.join(tmpdir, "pyproject.toml"), "w") as f:
            f.write("[project]\nname = 'ml-project'\n")

        analysis = SkillScannerRole._fallback_deep_scanner(project_dir=tmpdir, working_dir=tmpdir)

        # Check structure
        assert "artifact_type" in analysis
        assert "content_samples" in analysis
        assert "structure" in analysis
        assert "metadata" in analysis
        assert "surface_scan" in analysis

        # Check artifact classification
        at = analysis["artifact_type"]
        assert "type" in at
        assert at["type"] in ("software_project", "unknown")
        assert "confidence" in at

        # Check content sampling read actual files
        samples = analysis.get("content_samples", {})
        assert len(samples) > 0, "Should have read at least one file"

        # Check metadata
        meta = analysis["metadata"]
        assert meta.get("has_docs", False)

        print(f"  PASS test_deep_scanner (type: {at['type']}, "
              f"confidence: {at['confidence']}, samples: {len(samples)})")


# ═══════════════════════════════════════════════════════════════════════════
# Test 10: Domain Expertise fallback detect
# ═══════════════════════════════════════════════════════════════════════════

def test_domain_expertise_fallback_detect():
    """DomainExpertiseDetector._fallback_detect infers domain knowledge."""
    from skill.detectors import DomainExpertiseDetectorRole

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create artifact_analysis.json with ML content
        analysis = {
            "artifact_type": {"type": "software_project", "confidence": "high"},
            "content_samples": {
                "src/model.py": "import torch\nimport torch.nn as nn\n"
                               "class Transformer(nn.Module):\n"
                               "    def __init__(self):\n"
                               "        self.attention = nn.MultiheadAttention(512, 8)\n"
                               "        self.dropout = nn.Dropout(0.1)\n",
            },
            "surface_scan": {"file_categories": {"source": ["src/model.py"]}},
        }
        with open(os.path.join(tmpdir, "artifact_analysis.json"), "w") as f:
            json.dump(analysis, f)

        role = DomainExpertiseDetectorRole()
        result = role._fallback_detect(project_dir=tmpdir, working_dir=tmpdir)
        assert "domain" in result
        assert result["domain"] == "Domain Expertise"
        assert "inferred_skills" in result
        skills = result["inferred_skills"]
        names = [s["name"] for s in skills]
        print(f"  PASS test_domain_expertise_fallback_detect "
              f"({len(skills)} skills: {', '.join(names[:5]) if names else 'none'})")


# ═══════════════════════════════════════════════════════════════════════════
# Test 11: Methodology fallback detect
# ═══════════════════════════════════════════════════════════════════════════

def test_methodology_fallback_detect():
    """MethodologyDetector._fallback_detect detects tools and workflows."""
    from skill.detectors import MethodologyDetectorRole

    with tempfile.TemporaryDirectory() as tmpdir:
        analysis = {
            "artifact_type": {"type": "software_project", "confidence": "high"},
            "content_samples": {},
            "surface_scan": {
                "file_categories": {
                    "source": ["src/app.py", "src/models.py"],
                    "config": ["pyproject.toml", "package.json"],
                    "ci": [".github/workflows/ci.yml"],
                    "test": ["tests/test_app.py"],
                }
            },
        }
        with open(os.path.join(tmpdir, "artifact_analysis.json"), "w") as f:
            json.dump(analysis, f)

        role = MethodologyDetectorRole()
        result = role._fallback_detect(project_dir=tmpdir, working_dir=tmpdir)
        assert "domain" in result
        assert result["domain"] == "Methodology & Tooling"
        assert "detected_tools" in result
        assert "inferred_skills" in result
        tools = result["detected_tools"]
        skills = result["inferred_skills"]
        print(f"  PASS test_methodology_fallback_detect "
              f"({len(tools)} tools, {len(skills)} skills)")


# ═══════════════════════════════════════════════════════════════════════════
# Test 12: Fallback aggregation
# ═══════════════════════════════════════════════════════════════════════════

def test_fallback_aggregator():
    """_fallback_aggregator deduplicates tools and inferred skills."""
    from skill.aggregator import SkillAggregatorRole

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write new-format detector reports
        with open(os.path.join(tmpdir, "domain_expertise_report.json"), "w") as f:
            json.dump({
                "domain": "Domain Expertise",
                "inferred_skills": [
                    {"name": "Machine Learning", "proficiency": "advanced",
                     "confidence": "high", "evidence": {"signal_matches": ["transformer"]}},
                ],
                "detected_tools": [
                    {"name": "PyTorch", "category": "ML framework", "proficiency": "advanced"},
                ],
            }, f)

        with open(os.path.join(tmpdir, "methodology_report.json"), "w") as f:
            json.dump({
                "domain": "Methodology & Tooling",
                "inferred_skills": [
                    {"name": "Git Workflow Maturity", "proficiency": "intermediate",
                     "confidence": "medium"},
                ],
                "detected_tools": [
                    {"name": "Git", "category": "Version Control", "proficiency": "advanced"},
                    {"name": "GitHub Actions", "category": "CI/CD", "proficiency": "intermediate"},
                ],
            }, f)

        result = SkillAggregatorRole._fallback_aggregator(project_dir=tmpdir, working_dir=tmpdir)
        assert "tools" in result
        assert "inferred_skills" in result
        assert "summary" in result
        s = result["summary"]
        assert s.get("total_tools", 0) > 0
        print(f"  PASS test_fallback_aggregator "
              f"({s.get('total_tools', 0)} tools, {s.get('total_inferred_skills', 0)} skills)")


# ═══════════════════════════════════════════════════════════════════════════
# Test 13: Fallback writer (v2 format)
# ═══════════════════════════════════════════════════════════════════════════

def test_fallback_writer():
    """Writer fallback handles v2 inventory (tools + inferred_skills)."""
    from skill.writer import SkillReportWriterRole

    with tempfile.TemporaryDirectory() as tmpdir:
        inventory = {
            "artifact_type": "software_project",
            "tools": [
                {"name": "Python", "category": "Languages & Runtimes", "proficiency": "advanced", "sources": ["Methodology"]},
                {"name": "Docker", "category": "Containers", "proficiency": "intermediate", "sources": ["Methodology"]},
            ],
            "inferred_skills": [
                {"name": "Design Patterns", "category": "Technical Craft",
                 "proficiency": "advanced", "confidence": "high",
                 "evidence": {"description": "Factory pattern in models.py"}},
            ],
            "summary": {
                "total_tools": 2, "total_inferred_skills": 1,
                "dimensions_covered": ["Domain Expertise", "Technical Craft"],
                "proficiency_distribution": {"expert": 0, "advanced": 2, "intermediate": 1, "beginner": 0},
            },
        }

        skills_data = SkillReportWriterRole._fallback_skills_json("TestProject", inventory)
        SkillReportWriterRole._write_skills_json(tmpdir, skills_data)
        SkillReportWriterRole._fallback_report_md("TestProject", inventory, tmpdir)

        json_path = os.path.join(tmpdir, "skills.json")
        md_path = os.path.join(tmpdir, "skills_report.md")
        assert os.path.exists(json_path), "skills.json should exist"
        assert os.path.exists(md_path), "skills_report.md should exist"

        with open(json_path) as f:
            data = json.load(f)
        assert data.get("project") == "TestProject"
        assert data.get("artifact_type") == "software_project"
        all_tools = data.get("all_tools", [])
        all_skills = data.get("all_skills", [])
        assert len(all_tools) == 2
        assert len(all_skills) == 1

        with open(md_path) as f:
            md = f.read()
        assert "TestProject" in md
        assert "Design Patterns" in md or "Skill" in md
        print(f"  PASS test_fallback_writer ({os.path.getsize(json_path)}B JSON, {os.path.getsize(md_path)}B MD)")


# ═══════════════════════════════════════════════════════════════════════════
# Test 14: E2E fallback chain (no LLM calls)
# ═══════════════════════════════════════════════════════════════════════════

def test_e2e_fallback_chain():
    """Full fallback chain: deep scan → 4 detects → aggregate → write."""
    from skill.scanner import SkillScannerRole
    from skill.detectors import (DomainExpertiseDetectorRole, TechnicalCraftDetectorRole,
                                  MethodologyDetectorRole, RigorDetectorRole)
    from skill.aggregator import SkillAggregatorRole
    from skill.writer import SkillReportWriterRole

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a realistic mini-project
        os.makedirs(os.path.join(tmpdir, "src"))
        with open(os.path.join(tmpdir, "src", "app.py"), "w") as f:
            f.write("import flask\nfrom sqlalchemy import create_engine\n\n"
                    "class UserService:\n"
                    "    def __init__(self, db):\n"
                    "        self.db = db\n"
                    "    def get_user(self, user_id: int) -> dict | None:\n"
                    "        try:\n"
                    "            return self.db.query(user_id)\n"
                    "        except Exception as e:\n"
                    "            logging.error(f'Failed: {e}')\n"
                    "            return None\n")
        os.makedirs(os.path.join(tmpdir, "tests"), exist_ok=True)
        with open(os.path.join(tmpdir, "tests", "test_app.py"), "w") as f:
            f.write("import pytest\nfrom src.app import UserService\n\n"
                    "def test_get_user():\n"
                    "    db = Mock()\n"
                    "    svc = UserService(db)\n"
                    "    assert svc.get_user(1) is not None\n")
        with open(os.path.join(tmpdir, "pyproject.toml"), "w") as f:
            f.write("[project]\nname='mini-project'\n"
                    "[tool.pytest.ini_options]\ntestpaths=['tests']\n")
        with open(os.path.join(tmpdir, "README.md"), "w") as f:
            f.write("# Mini Project\nA sample project with tests and config.\n")

        # 1. Deep scan
        analysis = SkillScannerRole._fallback_deep_scanner(project_dir=tmpdir, working_dir=tmpdir)
        assert "artifact_type" in analysis

        # 2. Run all 4 detectors
        detector_classes = [
            DomainExpertiseDetectorRole,
            TechnicalCraftDetectorRole,
            MethodologyDetectorRole,
            RigorDetectorRole,
        ]
        detector_results = []
        for dc in detector_classes:
            role = dc()
            result = role._fallback_detect(project_dir=tmpdir, working_dir=tmpdir)
            detector_results.append(result)
            # Write report so aggregator can read it
            with open(os.path.join(tmpdir, role.output_file), "w") as f:
                json.dump(result, f)

        # Verify each detector produced valid output
        for dr in detector_results:
            assert "domain" in dr
            has_data = dr.get("inferred_skills") or dr.get("detected_tools")
            print(f"    {dr['domain']}: {len(dr.get('inferred_skills', []))} skills, "
                  f"{len(dr.get('detected_tools', []))} tools"
                  f"{' (has data)' if has_data else ''}")

        # 3. Aggregate
        inventory = SkillAggregatorRole._fallback_aggregator(project_dir=tmpdir, working_dir=tmpdir)
        assert inventory.get("tools") or inventory.get("inferred_skills"), \
            "Should have tools or inferred skills"

        # 4. Write
        skills_data = SkillReportWriterRole._fallback_skills_json("MiniProject", inventory)
        SkillReportWriterRole._write_skills_json(tmpdir, skills_data)
        SkillReportWriterRole._fallback_report_md("MiniProject", inventory, tmpdir)

        assert os.path.exists(os.path.join(tmpdir, "skills.json"))
        assert os.path.exists(os.path.join(tmpdir, "skills_report.md"))

        summary = inventory.get("summary", {})
        print(f"  PASS test_e2e_fallback_chain "
              f"({summary.get('total_tools', 0)} tools, "
              f"{summary.get('total_inferred_skills', 0)} skills, "
              f"dimensions: {summary.get('dimensions_covered', [])})")


# ═══════════════════════════════════════════════════════════════════════════
# Test 15: parse_result extracts JSON from Write tool-call parameters
# ═══════════════════════════════════════════════════════════════════════════

def test_parse_result_from_write_tool_call():
    """parse_result extracts and validates JSON embedded in Write tool calls."""
    from skill.detectors import RigorDetectorRole
    from unittest.mock import MagicMock

    report_json = json.dumps({
        "domain": "Depth & Rigor",
        "inferred_skills": [
            {"name": "Testing Strategy", "proficiency": "intermediate",
             "confidence": "high", "evidence": {"test_count": 97}},
            {"name": "Fallback & Resilience Design", "proficiency": "advanced",
             "confidence": "high", "evidence": {"fallback_count": 13}},
        ],
        "detected_tools": [],
    })

    write_params = json.dumps({
        "file_path": "/tmp/skill_output/rigor_report.json",
        "content": report_json,
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        role = RigorDetectorRole()
        role.output_file = "rigor_report.json"

        # Simulate: AI messages have no raw JSON, only Write tool call has it
        ai_msg = MagicMock()
        ai_msg.content = "Now I'll write the report. TASK_COMPLETE"

        tool_msg = MagicMock()
        tool_msg.content = f"[tool_call: Write {write_params}]"

        result = MagicMock()
        result.messages = [ai_msg, tool_msg]

        # Should extract from Write tool call params, not fallback
        report = role.parse_result(result, working_dir=tmpdir, project_dir=".")
        assert len(report.get("inferred_skills", [])) == 2
        assert report["inferred_skills"][0]["name"] == "Testing Strategy"
        assert report.get("_fallback") != True
        print("  PASS test_parse_result_from_write_tool_call")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Skill Summarizer v2 — Smoke Tests")
    print("=" * 50)

    tests = [
        test_imports,
        test_agent_roles_instantiate,
        test_tools_for_backend,
        test_state_keys,
        test_pipeline_instantiation,
        test_decompose_returns_empty,
        test_build_initial_state,
        test_fallback_scanner,
        test_deep_scanner,
        test_domain_expertise_fallback_detect,
        test_methodology_fallback_detect,
        test_fallback_aggregator,
        test_fallback_writer,
        test_e2e_fallback_chain,
        test_parse_result_from_write_tool_call,
    ]

    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            failed += 1
            import traceback
            print(f"  FAIL {test.__name__}: {e}")
            traceback.print_exc()

    print("=" * 50)
    total = len(tests)
    print(f"Results: {total - failed}/{total} passed" + (f", {failed} FAILED" if failed else " — ALL PASSED"))
    sys.exit(1 if failed else 0)
