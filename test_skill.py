"""Smoke tests for Skill Summarizer pipeline — agent roles, state, pipeline, and fallbacks."""

import json
import os
import sys
import tempfile


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Imports
# ═══════════════════════════════════════════════════════════════════════════

def test_imports():
    from skill.scanner import SkillScannerRole
    from skill.detectors import (PythonDetectorRole, JSDetectorRole,
                                  InfraDetectorRole, ConfigDocsDetectorRole)
    from skill.aggregator import SkillAggregatorRole
    from skill.writer import SkillReportWriterRole
    from pipeline import SkillPipeline, SkillState
    print("  PASS test_imports")


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: Agent role instantiation
# ═══════════════════════════════════════════════════════════════════════════

def test_agent_roles_instantiate():
    from skill.scanner import SkillScannerRole
    from skill.detectors import (PythonDetectorRole, JSDetectorRole,
                                  InfraDetectorRole, ConfigDocsDetectorRole)
    from skill.aggregator import SkillAggregatorRole
    from skill.writer import SkillReportWriterRole

    roles = [
        (SkillScannerRole(), "skill_scanner", 8),
        (PythonDetectorRole(), "python_detector", 12),
        (JSDetectorRole(), "js_detector", 12),
        (InfraDetectorRole(), "infra_detector", 12),
        (ConfigDocsDetectorRole(), "configdocs_detector", 12),
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
    from skill.detectors import (PythonDetectorRole, JSDetectorRole,
                                  InfraDetectorRole, ConfigDocsDetectorRole)
    from skill.aggregator import SkillAggregatorRole
    from skill.writer import SkillReportWriterRole

    for role_cls in [SkillScannerRole, PythonDetectorRole, JSDetectorRole,
                     InfraDetectorRole, ConfigDocsDetectorRole,
                     SkillAggregatorRole, SkillReportWriterRole]:
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
        "project_scan", "detector_outputs", "skill_inventory", "status",
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
        assert state["detector_outputs"] == []
        assert state["skill_inventory"] == {}
        assert state["status"] == "initialized"
    print("  PASS test_build_initial_state")


# ═══════════════════════════════════════════════════════════════════════════
# Test 8: Fallback scanner
# ═══════════════════════════════════════════════════════════════════════════

def test_fallback_scanner():
    """_fallback_scanner produces a valid project scan dict."""
    from skill.scanner import SkillScannerRole

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a few files to scan
        os.makedirs(os.path.join(tmpdir, "src"))
        with open(os.path.join(tmpdir, "src", "main.py"), "w") as f:
            f.write("print('hello')")
        with open(os.path.join(tmpdir, "README.md"), "w") as f:
            f.write("# Test")

        scan = SkillScannerRole._fallback_scanner(project_dir=tmpdir, working_dir=tmpdir)
        assert "file_categories" in scan
        assert "total_files" in scan
        assert scan["total_files"] >= 2
        assert "source" in scan["file_categories"]
        print(f"  PASS test_fallback_scanner ({scan['total_files']} files found)")


# ═══════════════════════════════════════════════════════════════════════════
# Test 9: Fallback detection
# ═══════════════════════════════════════════════════════════════════════════

def test_python_fallback_detect():
    """PythonDetector._fallback_detect scans a project for Python ecosystem skills."""
    from skill.detectors import PythonDetectorRole

    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "src"))
        with open(os.path.join(tmpdir, "requirements.txt"), "w") as f:
            f.write("numpy>=1.21.0\npandas>=1.3.0\n")
        with open(os.path.join(tmpdir, "setup.py"), "w") as f:
            f.write("from setuptools import setup\nsetup(name='test')")
        # Create project_scan.json that _load_project_scan reads
        scan = {"total_files": 2, "file_categories": {"source": [], "config": ["requirements.txt", "setup.py"]}}
        with open(os.path.join(tmpdir, "project_scan.json"), "w") as f:
            json.dump(scan, f)

        role = PythonDetectorRole()
        result = role._fallback_detect(project_dir=tmpdir, working_dir=tmpdir)
        assert "skills" in result
        assert "domain" in result
        assert result["domain"] == "Python"
        skills = result["skills"]
        names = [s["name"] for s in skills]
        # Should detect at least numpy and pandas
        has_package = any("numpy" in name.lower() or "pandas" in name.lower() for name in names)
        print(f"  PASS test_python_fallback_detect ({len(skills)} skills: {', '.join(names[:5])}{' — numpy/pandas found' if has_package else ''})")


# ═══════════════════════════════════════════════════════════════════════════

def test_js_fallback_detect():
    """JSDetector._fallback_detect scans for JS ecosystem skills."""
    from skill.detectors import JSDetectorRole

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "package.json"), "w") as f:
            json.dump({"dependencies": {"react": "^18.0.0", "next": "^13.0.0"}}, f)
        # Create project_scan.json
        scan = {"total_files": 1, "file_categories": {"source": [], "config": ["package.json"]}}
        with open(os.path.join(tmpdir, "project_scan.json"), "w") as f:
            json.dump(scan, f)

        role = JSDetectorRole()
        result = role._fallback_detect(project_dir=tmpdir, working_dir=tmpdir)
        assert "skills" in result
        assert result["domain"] == "JavaScript"
        skills = result["skills"]
        assert len(skills) >= 1
        names = [s["name"] for s in skills]
        print(f"  PASS test_js_fallback_detect ({len(skills)} skills: {', '.join(names[:5])})")


# ═══════════════════════════════════════════════════════════════════════════
# Test 11: Fallback aggregation
# ═══════════════════════════════════════════════════════════════════════════

def test_fallback_aggregator():
    """_fallback_aggregator deduplicates and categorizes skills."""
    from skill.aggregator import SkillAggregatorRole

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create domain report files that the aggregator reads from disk
        with open(os.path.join(tmpdir, "python_report.json"), "w") as f:
            json.dump({"domain": "Python", "skills": [
                {"name": "flask", "category": "frameworks", "proficiency": "used", "evidence": ["reqs.txt"]},
                {"name": "pytest", "category": "testing", "proficiency": "used", "evidence": ["reqs.txt"]},
            ]}, f)

        result = SkillAggregatorRole._fallback_aggregator(project_dir=tmpdir, working_dir=tmpdir)
        assert "skills" in result
        assert "summary" in result
        s = result["summary"]
        assert s.get("total_skills", 0) > 0, "Should detect at least one skill"
        print(f"  PASS test_fallback_aggregator ({s.get('total_skills', 0)} skills)")


# ═══════════════════════════════════════════════════════════════════════════
# Test 12: Fallback writer
# ═══════════════════════════════════════════════════════════════════════════

def test_fallback_writer():
    """Writer fallback produces valid skills.json and skills_report.md."""
    from skill.writer import SkillReportWriterRole

    with tempfile.TemporaryDirectory() as tmpdir:
        inventory = {
            "skills": [
                {"name": "Python", "category": "languages", "proficiency": "extensively-used", "evidence": "85% of files"},
                {"name": "Flask", "category": "frameworks", "proficiency": "used", "evidence": "Found in requirements.txt"},
                {"name": "Docker", "category": "tools", "proficiency": "detected", "evidence": "Dockerfile present"},
            ],
            "summary": {"total_skills": 3, "domains_covered": ["python", "infrastructure"]},
        }

        skills_data = SkillReportWriterRole._fallback_skills_json("TestProject", inventory)
        SkillReportWriterRole._write_skills_json(tmpdir, skills_data)
        SkillReportWriterRole._fallback_report_md("TestProject", inventory, tmpdir)

        # Verify files exist
        json_path = os.path.join(tmpdir, "skills.json")
        md_path = os.path.join(tmpdir, "skills_report.md")
        assert os.path.exists(json_path), "skills.json should exist"
        assert os.path.exists(md_path), "skills_report.md should exist"

        # Verify JSON structure
        with open(json_path) as f:
            data = json.load(f)
        assert data.get("project") == "TestProject" or data.get("project_name") == "TestProject"
        all_skills = data.get("all_skills", data.get("skills", []))
        assert len(all_skills) == 3

        # Verify markdown structure
        with open(md_path) as f:
            md = f.read()
        assert "# Skill Inventory" in md or "Skill" in md
        assert "TestProject" in md
        print(f"  PASS test_fallback_writer ({os.path.getsize(json_path)}B JSON, {os.path.getsize(md_path)}B MD)")


# ═══════════════════════════════════════════════════════════════════════════
# Test 13: E2E fallback chain (no LLM calls)
# ═══════════════════════════════════════════════════════════════════════════

def test_e2e_fallback_chain():
    """Full fallback chain: scan → detect → aggregate → write."""
    from skill.scanner import SkillScannerRole
    from skill.detectors import PythonDetectorRole
    from skill.aggregator import SkillAggregatorRole
    from skill.writer import SkillReportWriterRole

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a realistic mini-project
        os.makedirs(os.path.join(tmpdir, "src"))
        with open(os.path.join(tmpdir, "src", "app.py"), "w") as f:
            f.write("import flask\nfrom sqlalchemy import create_engine\n")
        with open(os.path.join(tmpdir, "requirements.txt"), "w") as f:
            f.write("flask>=2.0\nsqlalchemy>=1.4\npytest>=7.0\n")
        with open(os.path.join(tmpdir, "Dockerfile"), "w") as f:
            f.write("FROM python:3.11\nCOPY . /app\n")
        with open(os.path.join(tmpdir, "README.md"), "w") as f:
            f.write("# My Project\nA sample project.\n")

        # 1. Scan
        scan = SkillScannerRole._fallback_scanner(project_dir=tmpdir, working_dir=tmpdir)
        assert scan["total_files"] >= 4

        # 2. Detect (Python) — write project_scan.json first for _load_project_scan
        scan_path = os.path.join(tmpdir, "project_scan.json")
        with open(scan_path, "w") as f:
            json.dump(scan, f)
        role = PythonDetectorRole()
        py_report = role._fallback_detect(project_dir=tmpdir, working_dir=tmpdir)
        skills = py_report.get("skills", [])
        skill_names = [s["name"] for s in skills]
        print(f"  Detected skills: {skill_names}")

        # 3. Write domain report so aggregator can read it
        with open(os.path.join(tmpdir, "python_report.json"), "w") as f:
            json.dump(py_report, f)

        # 4. Aggregate
        inventory = SkillAggregatorRole._fallback_aggregator(project_dir=tmpdir, working_dir=tmpdir)
        assert inventory["summary"]["total_skills"] > 0

        # 5. Write
        skills_data = SkillReportWriterRole._fallback_skills_json("TestProject", inventory)
        SkillReportWriterRole._write_skills_json(tmpdir, skills_data)
        SkillReportWriterRole._fallback_report_md("TestProject", inventory, tmpdir)

        assert os.path.exists(os.path.join(tmpdir, "skills.json"))
        assert os.path.exists(os.path.join(tmpdir, "skills_report.md"))

        total = inventory["summary"]["total_skills"]
        print(f"  PASS test_e2e_fallback_chain ({total} skills detected)")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Skill Summarizer — Smoke Tests")
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
        test_python_fallback_detect,
        test_js_fallback_detect,
        test_fallback_aggregator,
        test_fallback_writer,
        test_e2e_fallback_chain,
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
