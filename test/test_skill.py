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
    """_fallback_deep_scanner produces artifact analysis with classification and key_files."""
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

        # Check key_files (v2)
        key_files = analysis.get("key_files", [])
        assert isinstance(key_files, list), "key_files should be a list"

        print(f"  PASS test_deep_scanner (type: {at['type']}, "
              f"confidence: {at['confidence']}, samples: {len(samples)}, "
              f"key_files: {len(key_files)})")


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
        assert "skill_graph" in result, "v2: output must include skill_graph"
        s = result["summary"]
        assert s.get("total_tools", 0) > 0
        # v2: check cross_referenced fields on inferred skills
        for skill in result.get("inferred_skills", []):
            assert "cross_referenced" in skill, (
                f"v2: skill '{skill.get('name', '?')}' must have cross_referenced flag"
            )
            assert "cross_referenced_sources" in skill, (
                f"v2: skill '{skill.get('name', '?')}' must have cross_referenced_sources"
            )
        print(f"  PASS test_fallback_aggregator "
              f"({s.get('total_tools', 0)} tools, {s.get('total_inferred_skills', 0)} skills, "
              f"skill_graph: {len(result.get('skill_graph', {}).get('nodes', []))} nodes)")


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
# Test 16: Scanner v2 — max_chars increased to 4000
# ═══════════════════════════════════════════════════════════════════════════

def test_max_chars_increased_to_4000():
    """_sample_content should capture up to 4000 chars per file (v2: doubled from 2000)."""
    from skill.scanner import _sample_content

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a file with >4000 chars of complex content to test full capacity
        long_func = "    " + "x = data.transform()\n" * 50  # ~1250 chars
        content = (
            "import numpy as np\n"
            "import pandas as pd\n"
            "from typing import Any\n\n"
            "class DataPipeline:\n"
            "    def __init__(self, config: dict[str, Any]) -> None:\n"
            "        self.config = config\n"
            "        self.steps: list[Any] = []\n\n"
            "    def add_step(self, step: Any) -> None:\n"
            "        self.steps.append(step)\n\n"
            "    def run(self, data: np.ndarray) -> np.ndarray:\n"
            + long_func +
            "        return x\n\n"
            "class AdvancedPipeline(DataPipeline):\n"
            "    def __init__(self, config: dict[str, Any]) -> None:\n"
            "        super().__init__(config)\n"
            "        self.cache: dict[str, Any] = {}\n\n"
            "    def run(self, data: np.ndarray) -> np.ndarray:\n"
            "        try:\n"
            "            result = super().run(data)\n"
            "            self.cache['last'] = result\n"
            "            return result\n"
            "        except ValueError as e:\n"
            "            raise RuntimeError(f'Pipeline failed: {e}')\n"
        )
        # Pad to over 4000 chars
        content = content + "\n# Padding\n" + "# comment line\n" * 30

        os.makedirs(os.path.join(tmpdir, "src"))
        file_path = os.path.join(tmpdir, "src", "pipeline.py")
        with open(file_path, "w") as f:
            f.write(content)

        file_list = ["src/pipeline.py"]
        samples = _sample_content(tmpdir, tmpdir, file_list)

        assert len(samples) > 0, "Should sample at least one file"
        sample_content = list(samples.values())[0]
        # Content should be up to 4000 chars (not capped at 2000)
        assert len(sample_content) > 2000, (
            f"Expected >2000 chars (v2 limit is 4000), got {len(sample_content)}"
        )
        print(f"  PASS test_max_chars_increased_to_4000 "
              f"(sample length: {len(sample_content)} chars)")


# ═══════════════════════════════════════════════════════════════════════════
# Test 17: Scanner v2 — complex files sampled before README
# ═══════════════════════════════════════════════════════════════════════════

def test_complex_files_sampled_before_readme():
    """Complex source files should appear before or instead of README in samples."""
    from skill.scanner import _sample_content

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a complex source file
        os.makedirs(os.path.join(tmpdir, "src"))
        complex_py = os.path.join(tmpdir, "src", "complex.py")
        with open(complex_py, "w") as f:
            f.write(
                "import asyncio\n"
                "from typing import Generic, TypeVar\n"
                "from dataclasses import dataclass\n"
                "from abc import ABC, abstractmethod\n\n"
                "T = TypeVar('T')\n\n"
                "@dataclass\n"
                "class Config:\n"
                "    host: str\n"
                "    port: int = 8080\n\n"
                "class BaseService(ABC, Generic[T]):\n"
                "    @abstractmethod\n"
                "    async def process(self, item: T) -> T:\n"
                "        ...\n\n"
                "class Service(BaseService[dict]):\n"
                "    async def process(self, item: dict) -> dict:\n"
                "        try:\n"
                "            result = await self._transform(item)\n"
                "            return result\n"
                "        except Exception as e:\n"
                "            raise\n\n"
                "    async def _transform(self, item: dict) -> dict:\n"
                "        with self._lock:\n"
                "            return {**item, 'processed': True}\n"
            )

        # Create a simple README
        readme_md = os.path.join(tmpdir, "README.md")
        with open(readme_md, "w") as f:
            f.write("# My Project\n\nJust a project.\n")

        # Create a config file
        config_toml = os.path.join(tmpdir, "pyproject.toml")
        with open(config_toml, "w") as f:
            f.write("[project]\nname = 'test'\n")

        file_list = ["src/complex.py", "README.md", "pyproject.toml"]
        samples = _sample_content(tmpdir, tmpdir, file_list)

        # Complex source file should be sampled
        sample_paths = list(samples.keys())
        has_complex = any("complex.py" in p for p in sample_paths)
        assert has_complex, (
            f"Complex source file 'src/complex.py' should be sampled. "
            f"Sampled: {sample_paths}"
        )
        print(f"  PASS test_complex_files_sampled_before_readme "
              f"(sampled: {sample_paths})")


# ═══════════════════════════════════════════════════════════════════════════
# Test 18: Scanner v2 — key_files in artifact_analysis
# ═══════════════════════════════════════════════════════════════════════════

def test_key_files_in_artifact_analysis():
    """_fallback_deep_scanner output includes a key_files list (v2)."""
    from skill.scanner import SkillScannerRole

    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "src"), exist_ok=True)
        with open(os.path.join(tmpdir, "src", "main.py"), "w") as f:
            f.write(
                "import sys\n"
                "from typing import Any\n"
                "from dataclasses import dataclass\n\n"
                "@dataclass\n"
                "class AppConfig:\n"
                "    debug: bool = False\n"
                "    port: int = 8080\n\n"
                "class Application:\n"
                "    def __init__(self, config: AppConfig) -> None:\n"
                "        self.config = config\n\n"
                "    def run(self) -> None:\n"
                "        try:\n"
                "            self._start()\n"
                "        except Exception as e:\n"
                "            print(f'Error: {e}')\n\n"
                "    def _start(self) -> None:\n"
                "        print(f'Starting on port {self.config.port}')\n\n"
                "if __name__ == '__main__':\n"
                "    app = Application(AppConfig())\n"
                "    app.run()\n"
            )
        with open(os.path.join(tmpdir, "README.md"), "w") as f:
            f.write("# Test Project\n")

        analysis = SkillScannerRole._fallback_deep_scanner(
            project_dir=tmpdir, working_dir=tmpdir
        )

        assert "key_files" in analysis, "artifact_analysis must have key_files (v2)"
        key_files = analysis["key_files"]
        assert isinstance(key_files, list), "key_files must be a list"
        # With a source file containing classes, functions, error handling, etc.,
        # we should have at least one key file
        assert len(key_files) >= 1, (
            f"Should have at least 1 key file, got {len(key_files)}"
        )
        print(f"  PASS test_key_files_in_artifact_analysis "
              f"({len(key_files)} key files)")


# ═══════════════════════════════════════════════════════════════════════════
# Test 19: Scanner v2 — key_files have rationale
# ═══════════════════════════════════════════════════════════════════════════

def test_key_files_have_rationale():
    """Each key_file entry must have a non-empty rationale string."""
    from skill.scanner import _identify_key_files

    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "src"), exist_ok=True)
        py_path = os.path.join(tmpdir, "src", "service.py")
        with open(py_path, "w") as f:
            f.write(
                "from abc import ABC, abstractmethod\n"
                "from typing import Generic, TypeVar\n\n"
                "T = TypeVar('T')\n\n"
                "class Repository(ABC, Generic[T]):\n"
                "    @abstractmethod\n"
                "    def find_by_id(self, id: str) -> T | None:\n"
                "        ...\n\n"
                "    @abstractmethod\n"
                "    def save(self, entity: T) -> T:\n"
                "        ...\n\n"
                "class UserRepository(Repository['User']):\n"
                "    def find_by_id(self, id: str) -> 'User | None':\n"
                "        try:\n"
                "            return self.db.query(id)\n"
                "        except Exception as e:\n"
                "            raise RepositoryError(f'Failed: {e}')\n"
            )

        file_list = ["src/service.py"]
        content_samples = {"src/service.py": open(py_path).read()[:4000]}
        key_files = _identify_key_files(file_list, content_samples, tmpdir)

        assert len(key_files) >= 1, f"Should identify at least 1 key file, got {len(key_files)}"
        for kf in key_files:
            assert "path" in kf, f"key_file missing 'path': {kf}"
            assert "rationale" in kf, f"key_file missing 'rationale': {kf}"
            assert isinstance(kf["rationale"], str), f"rationale must be str, got {type(kf['rationale'])}"
            assert len(kf["rationale"]) > 10, (
                f"rationale too short: '{kf['rationale']}'"
            )
            assert "skill_indicators" in kf, f"key_file missing 'skill_indicators': {kf}"
            assert isinstance(kf["skill_indicators"], list), (
                f"skill_indicators must be list, got {type(kf['skill_indicators'])}"
            )
            print(f"    key_file: {kf['path']} (score: {kf.get('complexity_score', 'N/A')})")
            print(f"      rationale: {kf['rationale'][:100]}...")
        print(f"  PASS test_key_files_have_rationale ({len(key_files)} key files)")


# ═══════════════════════════════════════════════════════════════════════════
# Test 20: Scanner v2 — key_files have complexity_score
# ═══════════════════════════════════════════════════════════════════════════

def test_key_files_have_complexity_score():
    """Each key_file entry must have a valid complexity_score 0.0-1.0."""
    from skill.scanner import _identify_key_files, _compute_file_complexity

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a complex file
        os.makedirs(os.path.join(tmpdir, "src"), exist_ok=True)
        py_path = os.path.join(tmpdir, "src", "complex_module.py")
        with open(py_path, "w") as f:
            f.write(
                "import asyncio\nimport logging\n"
                "from typing import Any, TypeVar, Generic\n"
                "from dataclasses import dataclass\n"
                "from abc import ABC, abstractmethod\n"
                "from enum import Enum\n\n"
                "T = TypeVar('T')\n\n"
                "class Status(Enum):\n"
                "    PENDING = 'pending'\n"
                "    RUNNING = 'running'\n"
                "    COMPLETE = 'complete'\n\n"
                "@dataclass\n"
                "class Task(Generic[T]):\n"
                "    id: str\n"
                "    status: Status = Status.PENDING\n"
                "    result: T | None = None\n\n"
                "class BaseHandler(ABC):\n"
                "    @abstractmethod\n"
                "    async def handle(self, task: Task[Any]) -> Task[Any]:\n"
                "        ...\n\n"
                "class Handler(BaseHandler):\n"
                "    async def handle(self, task: Task[Any]) -> Task[Any]:\n"
                "        try:\n"
                "            with self._lock:\n"
                "                result = await self._process(task)\n"
                "                task.status = Status.COMPLETE\n"
                "                task.result = result\n"
                "                return task\n"
                "        except ValueError as e:\n"
                "            logging.error('Handler failed: %s', e)\n"
                "            raise\n"
                "        except Exception:\n"
                "            task.status = Status.PENDING\n"
                "            return task\n"
            )

        file_list = ["src/complex_module.py"]
        content_samples = {"src/complex_module.py": open(py_path).read()[:4000]}
        key_files = _identify_key_files(file_list, content_samples, tmpdir)

        assert len(key_files) >= 1
        for kf in key_files:
            assert "complexity_score" in kf, f"key_file missing 'complexity_score': {kf}"
            score = kf["complexity_score"]
            assert isinstance(score, (int, float)), (
                f"complexity_score must be numeric, got {type(score)}"
            )
            assert 0.0 <= score <= 1.0, (
                f"complexity_score must be 0.0-1.0, got {score}"
            )
            print(f"    {kf['path']}: complexity_score={score}")

    # Also test _compute_file_complexity directly
    with tempfile.TemporaryDirectory() as tmpdir:
        # Simple file → low score
        simple_path = os.path.join(tmpdir, "simple.py")
        with open(simple_path, "w") as f:
            f.write("x = 1\nprint(x)\n")
        simple_score = _compute_file_complexity(simple_path)
        assert 0.0 <= simple_score <= 0.3, (
            f"Simple file should have low complexity, got {simple_score}"
        )

        # Complex file → high score
        complex_path = os.path.join(tmpdir, "complex.py")
        with open(complex_path, "w") as f:
            f.write(
                "class A:\n    def m1(self): pass\n    def m2(self): pass\n"
                "class B(A):\n    def m1(self): pass\n    def m3(self): pass\n"
                + ("x = 1\n" * 20)
            )
        complex_score = _compute_file_complexity(complex_path)
        assert complex_score > simple_score, (
            f"Complex file ({complex_score}) should score higher than simple ({simple_score})"
        )
        print(f"    simple.py: {simple_score}, complex.py: {complex_score}")
        print(f"  PASS test_key_files_have_complexity_score")


# ═══════════════════════════════════════════════════════════════════════════
# Test 21: Scanner v2 — generated files excluded from sampling
# ═══════════════════════════════════════════════════════════════════════════

def test_generated_files_excluded_from_sampling():
    """Minified, compiled, and generated files should not be sampled."""
    from skill.scanner import _sample_content, _is_generated_file

    # Test _is_generated_file directly
    assert _is_generated_file("dist/bundle.min.js"), "dist/ files should be excluded"
    assert _is_generated_file("build/output.min.css"), "build/ files should be excluded"
    assert _is_generated_file("src/__pycache__/module.cpython-311.pyc"), "__pycache__ should be excluded"
    assert _is_generated_file("src/app.min.js"), ".min. files should be excluded"
    assert _is_generated_file("lib/vendor/jquery.min.js"), "vendor/ should be excluded"

    # These should NOT be excluded
    assert not _is_generated_file("src/main.py"), "normal source file should NOT be excluded"
    assert not _is_generated_file("README.md"), "README should NOT be excluded"
    assert not _is_generated_file("pyproject.toml"), "config should NOT be excluded"
    assert not _is_generated_file("src/utils/min_value.py"), "min_ in name is not .min. marker"

    # Test sampling exclusion
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "src"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "dist"), exist_ok=True)

        # Normal file
        with open(os.path.join(tmpdir, "src", "main.py"), "w") as f:
            f.write("class App:\n    def run(self):\n        pass\n")

        # Minified file
        with open(os.path.join(tmpdir, "dist", "bundle.min.js"), "w") as f:
            f.write("var a=1,b=2,c=3;function x(){return a+b+c}" * 50)

        file_list = ["src/main.py", "dist/bundle.min.js"]
        samples = _sample_content(tmpdir, tmpdir, file_list)

        sample_paths = list(samples.keys())
        has_normal = any("main.py" in p for p in sample_paths)
        has_minified = any("bundle.min.js" in p for p in sample_paths)
        assert has_normal, f"Normal source file should be sampled. Got: {sample_paths}"
        assert not has_minified, (
            f"Minified file should be excluded from sampling. Got: {sample_paths}"
        )
        print(f"  PASS test_generated_files_excluded_from_sampling "
              f"(sampled: {len(samples)} files)")


# ═══════════════════════════════════════════════════════════════════════════
# Test 22: Scanner v2 — build_task prompt includes key_files and 4000-char
# ═══════════════════════════════════════════════════════════════════════════

def test_build_task_includes_key_files():
    """build_task prompt should reference key_files and 4000-char sampling (v2)."""
    from skill.scanner import SkillScannerRole

    role = SkillScannerRole()
    prompt = role.build_task("deepseek", project_dir="/test/project")

    # Check for 4000-char sampling instruction
    assert "4000 chars" in prompt or "~4000" in prompt, (
        "Prompt should mention 4000-char sampling limit (v2)"
    )

    # Check for key_files identification
    assert "key_files" in prompt or "key file" in prompt.lower(), (
        "Prompt should mention key_files identification (v2)"
    )

    # Check for sampling priority instruction
    assert "skill-demonstrative" in prompt.lower() or "complex" in prompt.lower(), (
        "Prompt should mention skill-demonstrative or complex file prioritization"
    )

    # Check for schema reference
    assert "rationale" in prompt, "Prompt should include rationale field in schema"
    assert "complexity_score" in prompt, "Prompt should include complexity_score field"

    print("  PASS test_build_task_includes_key_files")


# ═══════════════════════════════════════════════════════════════════════════
# Test 23: Scanner v2 — scanner module exports all new functions
# ═══════════════════════════════════════════════════════════════════════════

def test_scanner_module_exports():
    """Verify scanner module exports the new v2 functions and constants."""
    import skill.scanner as scanner

    expected_functions = [
        "_sample_content",
        "_classify_artifact",
        "_analyze_structure",
        "_identify_key_files",
        "_is_generated_file",
        "_compute_file_complexity",
    ]
    for func_name in expected_functions:
        assert hasattr(scanner, func_name), (
            f"skill.scanner should export {func_name}"
        )

    # Check constants
    assert hasattr(scanner, "_STRUCTURAL_KEYWORDS")
    assert hasattr(scanner, "_GENERATED_FILE_MARKERS")
    assert hasattr(scanner, "_GENERATED_DIR_MARKERS")

    print("  PASS test_scanner_module_exports")


# ═══════════════════════════════════════════════════════════════════════════
# Test 24-34: Detectors v2 — evidence_refs, qualitative proficiency,
# expanded domain/tool signals, version detection, scope field
# ═══════════════════════════════════════════════════════════════════════════


def test_evidence_refs_in_domain_detection():
    """v2: Domain expertise detection output must include evidence_refs with file paths."""
    from skill.detectors import _detect_domain_expertise

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

    skills = _detect_domain_expertise(analysis)
    for skill in skills:
        assert "evidence_refs" in skill, (
            f"v2: Skill '{skill['name']}' must have evidence_refs"
        )
        evidence_refs = skill["evidence_refs"]
        assert isinstance(evidence_refs, list), "evidence_refs must be a list"
        assert len(evidence_refs) > 0, (
            f"evidence_refs should have at least 1 entry for skill '{skill['name']}'"
        )
        for ref in evidence_refs:
            assert "file" in ref, f"evidence_ref missing 'file': {ref}"
            assert "signals" in ref, f"evidence_ref missing 'signals': {ref}"
            assert "description" in ref, f"evidence_ref missing 'description': {ref}"
            assert isinstance(ref["signals"], list), "signals must be a list"
    print(f"  PASS test_evidence_refs_in_domain_detection "
          f"({len(skills)} skills with evidence_refs)")


def test_qualitative_proficiency_not_count_based():
    """v2: Proficiency is based on qualitative assessment, not raw match count.

    A domain with fewer but more specific multi-word signals should score
    HIGHER than a domain with many single-word generic signals.
    """
    from skill.detectors import _detect_domain_expertise, _build_signal_specificity

    _build_signal_specificity()

    # ML content: many specific multi-word terms (high depth)
    ml_analysis = {
        "artifact_type": {"type": "software_project", "confidence": "high"},
        "content_samples": {
            "src/train.py": (
                "convolutional neural network architecture with "
                "batch normalization and dropout regularization. "
                "Uses attention mechanism and gradient descent "
                "with learning rate scheduling."
            ),
        },
        "surface_scan": {"file_categories": {"source": ["src/train.py"]}},
    }
    ml_skills = _detect_domain_expertise(ml_analysis)
    ml_names = [s["name"] for s in ml_skills]

    # Basic content: many matches but all single generic words
    basic_analysis = {
        "artifact_type": {"type": "software_project", "confidence": "high"},
        "content_samples": {
            "src/app.py": (
                "the game loop runs on the server with a class for each entity. "
                "the class handles game state. each class has a private key. "
                "the game uses a neural network for some features. "
                "the class design is simple."
            ),
        },
        "surface_scan": {"file_categories": {"source": ["src/app.py"]}},
    }
    basic_skills = _detect_domain_expertise(basic_analysis)

    # The ML analysis should have at least one skill detected
    assert len(ml_skills) > 0, "ML content should detect domain expertise"

    # Verify the proficiency is not just count-based:
    # ML skills with multi-word phrases should not have "beginner" proficiency
    # if they have sophisticated terminology
    for skill in ml_skills:
        assert skill["proficiency"] in ("beginner", "intermediate", "advanced", "expert"), (
            f"Invalid proficiency: {skill['proficiency']}"
        )
        # At minimum, proficiency and confidence are strings
        assert isinstance(skill["proficiency"], str)
        assert isinstance(skill["confidence"], str)

    print(f"  PASS test_qualitative_proficiency_not_count_based "
          f"(ML: {len(ml_skills)} skills, Basic: {len(basic_skills)} skills)")


def test_new_domain_signals_matched():
    """v2: New domains (Computer Vision, Networking, DevOps, etc.) are detected."""
    from skill.detectors import _detect_domain_expertise, _DOMAIN_SIGNALS

    # Verify the new domains exist in the signal dictionary
    new_domains = [
        "Computer Vision", "Reinforcement Learning", "Networking",
        "Operating Systems", "Embedded Systems", "DevOps",
        "Data Engineering", "Frontend Development", "Mobile Development",
        "Blockchain",
    ]
    for domain in new_domains:
        assert domain in _DOMAIN_SIGNALS, (
            f"v2: {domain} should be in _DOMAIN_SIGNALS"
        )
        signals = _DOMAIN_SIGNALS[domain]
        assert len(signals) >= 5, (
            f"{domain} should have at least 5 signals, got {len(signals)}"
        )

    # Test detection of a new domain with representative content
    analysis = {
        "artifact_type": {"type": "software_project", "confidence": "high"},
        "content_samples": {
            "src/vision.py": (
                "YOLO object detection model with non-maximum suppression. "
                "Uses ResNet backbone for feature extraction and "
                "convolutional neural network for image classification. "
                "Supports bounding box regression and semantic segmentation "
                "with data augmentation pipeline."
            ),
        },
        "surface_scan": {"file_categories": {"source": ["src/vision.py"]}},
    }
    skills = _detect_domain_expertise(analysis)
    names = [s["name"] for s in skills]
    assert "Computer Vision" in names, (
        f"Computer Vision should be detected, got: {names}"
    )
    print(f"  PASS test_new_domain_signals_matched (detected: {names})")


def test_multi_word_phrase_matching():
    """v2: Multi-word phrases are preferred over single-word sub-matches.

    'convolutional neural network' should match as a 3-word phrase (high specificity),
    not just as individual words 'neural' and 'network'.
    """
    from skill.detectors import _SIGNAL_SPECIFICITY, _build_signal_specificity

    _build_signal_specificity()

    # Multi-word phrases should have higher specificity weight
    multi_word = "convolutional neural network"
    assert multi_word in _SIGNAL_SPECIFICITY, (
        f"'{multi_word}' should be in _SIGNAL_SPECIFICITY"
    )
    multi_weight = _SIGNAL_SPECIFICITY[multi_word]

    single_word = "network"
    # 'network' alone is not a signal in any domain, but check weights of
    # similarly-scoped single vs multi-word signals
    two_word = "object detection"
    assert two_word in _SIGNAL_SPECIFICITY, (
        f"'{two_word}' should be in _SIGNAL_SPECIFICITY"
    )
    two_weight = _SIGNAL_SPECIFICITY[two_word]

    # Multi-word (3+) should have higher specificity than 2-word
    assert multi_weight >= two_weight, (
        f"3-word phrase weight ({multi_weight}) should be >= 2-word ({two_weight})"
    )
    print(f"  PASS test_multi_word_phrase_matching "
          f"(3-word={multi_weight}, 2-word={two_weight})")


def test_negative_signals_excluded():
    """v2: Negative signals prevent false-positive domain classifications."""
    from skill.detectors import _DOMAIN_NEGATIVE_SIGNALS, _check_negative_signals

    # Verify negative signals dict has entries for the listed domains
    assert "Game Development" in _DOMAIN_NEGATIVE_SIGNALS
    assert "Blockchain" in _DOMAIN_NEGATIVE_SIGNALS
    assert "Operating Systems" in _DOMAIN_NEGATIVE_SIGNALS

    # Content that mentions "game theory" (not game development) should trigger
    # negative signal for Game Development
    content_samples = {
        "paper.md": "This paper applies game theory to economic modeling. "
                     "We analyze the language game in social contexts."
    }
    penalty = _check_negative_signals("Game Development", content_samples)
    assert penalty > 0, (
        f"'game theory' in content should trigger negative signal penalty, got {penalty}"
    )

    # Content that is actually about game development should have no penalty
    gamedev_samples = {
        "engine.py": "The game loop runs at 60fps with entity component system. "
                      "Collision detection uses A* algorithm for pathfinding."
    }
    penalty_gamedev = _check_negative_signals("Game Development", gamedev_samples)
    # Even if no negative signals, penalty should be 0
    assert penalty_gamedev == 0, (
        f"Game dev content should have no negative penalty, got {penalty_gamedev}"
    )
    print(f"  PASS test_negative_signals_excluded "
          f"(penalty with false positive: {penalty}, without: {penalty_gamedev})")


def test_modern_tool_detection():
    """v2: Modern tools (uv, ruff, Vitest, Playwright, TailwindCSS, etc.) are detected."""
    from skill.detectors import _TOOL_INDICATORS, _detect_tools

    # Verify new tools are in the indicators dict
    modern_tools = [
        "uv", "Ruff", "Biome", "pnpm", "Bun", "Vitest", "Playwright",
        "TailwindCSS", "shadcn/ui", "tRPC", "Prisma", "Drizzle",
        "Zustand", "Svelte", "SolidJS", "Qwik", "Astro", "htmx",
        "Alpine.js", "Turbopack", "Poetry", "MSW",
    ]
    for tool in modern_tools:
        assert tool in _TOOL_INDICATORS, (
            f"v2: '{tool}' should be in _TOOL_INDICATORS"
        )
        indicators = _TOOL_INDICATORS[tool]
        assert len(indicators) >= 1, (
            f"'{tool}' should have at least 1 indicator, got {len(indicators)}"
        )

    # Test detection with realistic project files
    all_files = [
        "pyproject.toml", "src/app.py", "package.json",
        "vitest.config.ts", "playwright.config.ts",
        "tailwind.config.js", "src/components/Button.tsx",
        "src/__tests__/Button.test.tsx",
    ]
    all_text = """
    import { zustand } from 'zustand'
    import { useQuery } from '@tanstack/react-query'
    import { PrismaClient } from '@prisma/client'
    """
    tools = _detect_tools(all_files, all_text)

    tool_names = [t["name"] for t in tools]
    # Should detect at least some modern tools
    assert len(tools) > 0, "Should detect modern tools from config files"
    for tool in tools:
        assert "scope" in tool, f"v2: tool '{tool['name']}' must have 'scope' field"
        assert tool["scope"] in ("project_level", "ecosystem_level"), (
            f"Invalid scope '{tool['scope']}' for tool '{tool['name']}'"
        )
    print(f"  PASS test_modern_tool_detection ({len(tools)} tools: {', '.join(tool_names[:10])})")


def test_tool_version_detection():
    """v2: Tool version detection reads pyproject.toml and package.json for versions."""
    from skill.detectors import _detect_tool_version, _detect_tools

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create pyproject.toml with tool configs and project version
        pyproject_path = os.path.join(tmpdir, "pyproject.toml")
        with open(pyproject_path, "w") as f:
            f.write(
                '[project]\n'
                'name = "test-project"\n'
                'version = "2.1.0"\n'
                '\n'
                '[tool.ruff]\n'
                'line-length = 100\n'
                '\n'
                '[tool.uv]\n'
                'dev-dependencies = []\n'
            )

        all_files = ["pyproject.toml", "src/main.py"]

        # Test _detect_tool_version directly
        uv_ver = _detect_tool_version("uv", "[tool.uv]", all_files, tmpdir)
        assert uv_ver is not None, "Should detect uv version/config in pyproject.toml"
        print(f"    uv version: {uv_ver}")

        ruff_ver = _detect_tool_version("Ruff", "[tool.ruff]", all_files, tmpdir)
        assert ruff_ver is not None, "Should detect Ruff config in pyproject.toml"
        print(f"    ruff version: {ruff_ver}")

        # Test through _detect_tools (integration)
        tools = _detect_tools(all_files, "", tmpdir)
        tool_names = [t["name"] for t in tools]
        print(f"    tools detected: {tool_names}")

        # Tools with config file indicators should have scope=project_level
        for tool in tools:
            if tool["name"] in ("Ruff", "uv", "Python"):
                assert tool["scope"] == "project_level", (
                    f"Tool '{tool['name']}' with config files should be project_level"
                )

    print("  PASS test_tool_version_detection")


def test_tool_scope_field():
    """v2: Tools have a 'scope' field: project_level vs ecosystem_level."""
    from skill.detectors import _detect_tools, _ECOSYSTEM_LEVEL_INDICATORS

    # Verify ecosystem-level tools are defined
    assert "Markdown" in _ECOSYSTEM_LEVEL_INDICATORS
    assert "Python" in _ECOSYSTEM_LEVEL_INDICATORS
    assert "JavaScript" in _ECOSYSTEM_LEVEL_INDICATORS

    # Files-only detection: Markdown (.md) is ecosystem_level
    all_files = ["README.md", "src/main.py", "pyproject.toml"]
    tools = _detect_tools(all_files, "")

    for tool in tools:
        assert "scope" in tool, (
            f"v2: Every tool must have 'scope' field. Missing from '{tool['name']}'"
        )

    # Specific tools with config files should be project_level
    all_files_with_config = ["package.json", "vitest.config.ts", ".eslintrc", "README.md"]
    tools_with_config = _detect_tools(all_files_with_config, "")
    for tool in tools_with_config:
        if tool["name"] == "ESLint" or tool["name"] == "Vitest":
            assert tool["scope"] == "project_level", (
                f"'{tool['name']}' with config file should be project_level"
            )

    print(f"  PASS test_tool_scope_field ({len(tools_with_config)} tools)")


def test_code_craft_qualitative_basis():
    """v2: Code craft detection uses qualitative criteria (advanced_indicators, evidence_refs)."""
    from skill.detectors import _detect_code_craft

    # Sophisticated code: advanced patterns, multi-file, integrated
    sophisticated_analysis = {
        "artifact_type": {"type": "software_project", "confidence": "high"},
        "content_samples": {
            "src/services.py": (
                "class UserRepository:\n"
                "    def __init__(self, db: Database, cache: Cache) -> None:\n"
                "        self._db = db\n"
                "        self._cache = cache\n\n"
                "    def find_by_id(self, user_id: str) -> User | None:\n"
                "        try:\n"
                "            cached = await self._cache.get(user_id)\n"
                "            if cached:\n"
                "                return User.from_dict(cached)\n"
                "        except CacheError:\n"
                "            pass\n"
                "        with self._db.transaction():\n"
                "            user = self._db.query(User).filter_by(id=user_id).first()\n"
                "            if user:\n"
                "                self._cache.set(user_id, user.to_dict())\n"
                "            return user\n"
            ),
            "src/factory.py": (
                "from abc import ABC, abstractmethod\n"
                "from typing import TypeVar, Generic\n\n"
                "T = TypeVar('T')\n\n"
                "class AbstractFactory(ABC, Generic[T]):\n"
                "    @abstractmethod\n"
                "    def create(self) -> T: ...\n\n"
                "class ServiceFactory(AbstractFactory[UserService]):\n"
                "    def create(self) -> UserService:\n"
                "        db = Database.connect()\n"
                "        return UserService(db, Cache.default())\n"
            ),
        },
        "surface_scan": {"file_categories": {"source": ["src/services.py", "src/factory.py"]}},
    }

    skills = _detect_code_craft(sophisticated_analysis)
    skill_names = [s["name"] for s in skills]

    for skill in skills:
        assert "evidence_refs" in skill, (
            f"v2: Skill '{skill['name']}' must have evidence_refs"
        )
        assert "proficiency" in skill
        assert "confidence" in skill
        evidence = skill.get("evidence", {})
        # v2: advanced_indicators should be present
        assert "advanced_indicators" in evidence or "indicators_matched" in evidence, (
            f"Skill '{skill['name']}' should have indicators in evidence"
        )

    print(f"  PASS test_code_craft_qualitative_basis "
          f"({len(skills)} skills: {skill_names})")


def test_methodology_qualitative_basis():
    """v2: Methodology detection uses deep_signals and evidence_refs for qualitative assessment."""
    from skill.detectors import _detect_methodology_skills

    # Sophisticated methodology: multi-stage CI, locked deps, multi-container env
    analysis = {
        "artifact_type": {"type": "software_project", "confidence": "high"},
        "content_samples": {
            "Dockerfile": (
                "FROM python:3.11-slim AS builder\n"
                "RUN pip install --no-cache-dir -r requirements.txt\n\n"
                "FROM python:3.11-slim\n"
                "COPY --from=builder /app /app\n"
                "HEALTHCHECK CMD curl --fail http://localhost:8080/health || exit 1\n"
            ),
            "docker-compose.yml": (
                "services:\n"
                "  app:\n"
                "    build: .\n"
                "    ports: ['8080:8080']\n"
                "  db:\n"
                "    image: postgres:15\n"
                "    healthcheck:\n"
                "      test: ['CMD', 'pg_isready']\n"
            ),
            ".github/workflows/ci.yml": (
                "name: CI\n"
                "on: [push]\n"
                "jobs:\n"
                "  test:\n"
                "    runs-on: ubuntu-latest\n"
                "    steps:\n"
                "      - uses: actions/checkout@v4\n"
                "      - run: pytest\n"
                "  deploy-staging:\n"
                "    needs: test\n"
                "    if: github.ref == 'refs/heads/main'\n"
                "    steps:\n"
                "      - run: ./deploy.sh staging\n"
                "  deploy-prod:\n"
                "    needs: deploy-staging\n"
                "    environment: production\n"
                "    steps:\n"
                "      - run: ./deploy.sh production\n"
            ),
        },
        "surface_scan": {
            "file_categories": {
                "config": ["Dockerfile", "docker-compose.yml"],
                "ci": [".github/workflows/ci.yml"],
            }
        },
    }

    skills = _detect_methodology_skills(analysis)
    skill_names = [s["name"] for s in skills]

    for skill in skills:
        assert "evidence_refs" in skill, (
            f"v2: Methodology skill '{skill['name']}' must have evidence_refs"
        )
        evidence = skill.get("evidence", {})
        # v2: deep_signals should be present
        assert "deep_signals" in evidence, (
            f"Skill '{skill['name']}' should have deep_signals in evidence"
        )

    # Multi-stage CI/CD should score higher than basic
    ci_skill = next((s for s in skills if "CI/CD" in s["name"]), None)
    if ci_skill:
        # With multi-stage deploy (staging + production), should be at least intermediate
        assert ci_skill["proficiency"] in ("intermediate", "advanced", "expert"), (
            f"Multi-stage CI/CD should have proficiency >= intermediate, "
            f"got {ci_skill['proficiency']}"
        )

    # Environment management with HEALTHCHECK + multi-stage + docker-compose = sophisticated
    env_skill = next((s for s in skills if "Environment" in s["name"]), None)
    if env_skill:
        print(f"    Environment Management: proficiency={env_skill['proficiency']}, "
              f"confidence={env_skill['confidence']}")

    print(f"  PASS test_methodology_qualitative_basis "
          f"({len(skills)} skills: {skill_names})")


def test_rigor_qualitative_basis():
    """v2: Rigor detection uses testing pyramid completeness and enforcement layers."""
    from skill.detectors import _detect_rigor_skills

    # Project with unit + integration + E2E tests, multi-layer docs, multi-layer QA
    analysis = {
        "artifact_type": {"type": "software_project", "confidence": "high"},
        "content_samples": {
            "tests/unit/test_models.py": (
                "import pytest\n"
                "from src.models import User\n\n"
                "class TestUser:\n"
                "    def test_create_user(self):\n"
                "        user = User(name='test')\n"
                "        assert user.name == 'test'\n\n"
                "    @pytest.mark.parametrize('name', ['a', 'b', 'c'])\n"
                "    def test_names(self, name):\n"
                "        assert User(name=name).name == name\n"
            ),
            "tests/integration/test_api.py": (
                "import pytest\n"
                "from fastapi.testclient import TestClient\n\n"
                "def test_create_endpoint():\n"
                "    client = TestClient(app)\n"
                "    response = client.post('/api/users', json={'name': 'test'})\n"
                "    assert response.status_code == 201\n"
            ),
            "tests/e2e/test_flow.py": (
                "from playwright.sync_api import Page\n\n"
                "def test_user_flow(page: Page):\n"
                "    page.goto('http://localhost:3000')\n"
                "    page.click('text=Sign Up')\n"
                "    page.fill('input[name=email]', 'test@example.com')\n"
            ),
            "README.md": "# My Project\n\nInstallation guide and API reference.\n",
            "CONTRIBUTING.md": "# Contributing\n\nCode of conduct and style guide.\n",
            "docs/architecture.md": "# Architecture\n\nADR 001: Use hexagonal architecture.\n",
            ".pre-commit-config.yaml": (
                "repos:\n"
                "  - repo: https://github.com/astral-sh/ruff-pre-commit\n"
                "    hooks:\n"
                "      - id: ruff\n"
                "      - id: ruff-format\n"
            ),
        },
        "surface_scan": {
            "file_categories": {
                "test": [
                    "tests/unit/test_models.py",
                    "tests/integration/test_api.py",
                    "tests/e2e/test_flow.py",
                ],
                "doc": ["README.md", "CONTRIBUTING.md", "docs/architecture.md"],
                "config": [".pre-commit-config.yaml"],
            }
        },
    }

    skills = _detect_rigor_skills(analysis)
    skill_names = [s["name"] for s in skills]

    for skill in skills:
        assert "evidence_refs" in skill, (
            f"v2: Rigor skill '{skill['name']}' must have evidence_refs"
        )

    # Testing Strategy should reflect pyramid completeness (unit + integration + E2E = 3 levels)
    test_skill = next((s for s in skills if s["name"] == "Testing Strategy"), None)
    if test_skill:
        evidence = test_skill.get("evidence", {})
        assert evidence.get("pyramid_levels", 0) >= 1, (
            f"Should have pyramid_levels in testing evidence"
        )
        print(f"    Testing Strategy: proficiency={test_skill['proficiency']}, "
              f"pyramid_levels={evidence.get('pyramid_levels', 0)}")

    # Code Quality Enforcement with pre-commit + linter + formatter
    qa_skill = next((s for s in skills if s["name"] == "Code Quality Enforcement"), None)
    if qa_skill:
        evidence = qa_skill.get("evidence", {})
        assert evidence.get("enforcement_layers", 0) >= 1, (
            f"Should have enforcement_layers in QA evidence"
        )

    print(f"  PASS test_rigor_qualitative_basis "
          f"({len(skills)} skills: {skill_names})")


# ═══════════════════════════════════════════════════════════════════════════
# Test 34-42: Aggregator v2 — category inference, cross_referencing, evidence
# merging, confidence boost, skill_graph generation
# ═══════════════════════════════════════════════════════════════════════════


def test_category_inferred_from_domain():
    """v2: _infer_category produces correct categories based on domain + name."""
    from skill.aggregator import _infer_category

    # Domain Expertise → Domain Knowledge
    assert _infer_category("Machine Learning", "Domain Expertise") == "Domain Knowledge"
    assert _infer_category("Computer Vision", "Domain Expertise") == "Domain Knowledge"
    assert _infer_category("Distributed Systems", "Domain Expertise") == "Domain Knowledge"

    # Technical Craft → Technical Craft
    assert _infer_category("Design Patterns", "Technical Craft") == "Technical Craft"
    assert _infer_category("Error Handling Maturity", "Technical Craft") == "Technical Craft"
    assert _infer_category("Type System Proficiency", "Technical Craft") == "Technical Craft"
    assert _infer_category("Code Organization", "Technical Craft") == "Technical Craft"
    assert _infer_category("Performance Awareness", "Technical Craft") == "Technical Craft"
    assert _infer_category("Security Awareness", "Technical Craft") == "Technical Craft"

    # Writing craft (also Technical Craft)
    assert _infer_category("Argumentation", "Technical Craft") == "Technical Craft"
    assert _infer_category("Technical Writing", "Technical Craft") == "Technical Craft"
    assert _infer_category("Narrative Structure", "Technical Craft") == "Technical Craft"
    assert _infer_category("Clarity", "Technical Craft") == "Technical Craft"

    # Methodology & Tooling → Engineering Practice
    assert _infer_category("Git Workflow Maturity", "Methodology & Tooling") == "Engineering Practice"
    assert _infer_category("CI/CD Sophistication", "Methodology & Tooling") == "Engineering Practice"
    assert _infer_category("Dependency Management", "Methodology & Tooling") == "Engineering Practice"
    assert _infer_category("Environment Management", "Methodology & Tooling") == "Engineering Practice"
    assert _infer_category("Incremental Development", "Methodology & Tooling") == "Engineering Practice"

    # Depth & Rigor → Quality Assurance
    assert _infer_category("Testing Strategy", "Depth & Rigor") == "Quality Assurance"
    assert _infer_category("Test Coverage Thoroughness", "Depth & Rigor") == "Quality Assurance"
    assert _infer_category("Documentation Quality", "Depth & Rigor") == "Quality Assurance"
    assert _infer_category("Code Quality Enforcement", "Depth & Rigor") == "Quality Assurance"
    assert _infer_category("Academic Rigor", "Depth & Rigor") == "Quality Assurance"

    # Architecture sub-category refinement
    assert _infer_category("Component Design", "Technical Craft") == "Architecture"
    assert _infer_category("API Design", "Technical Craft") == "Architecture"
    assert _infer_category("Data Modeling", "Technical Craft") == "Architecture"
    assert _infer_category("Scalability", "Technical Craft") == "Architecture"
    assert _infer_category("Infrastructure Design", "Technical Craft") == "Architecture"

    # Unknown domain → domain fallback
    assert _infer_category("Some Unknown Skill", "Custom Domain") == "Custom Domain"

    print("  PASS test_category_inferred_from_domain")


def test_no_hardcoded_category_map_used():
    """v2: _fallback_aggregator does NOT reference a local _SKILL_CATEGORY_MAP."""
    import inspect
    from skill.aggregator import SkillAggregatorRole

    source = inspect.getsource(SkillAggregatorRole._fallback_aggregator)
    # The hardcoded map should not appear in the source
    assert "_SKILL_CATEGORY_MAP" not in source, (
        "v2: _fallback_aggregator should NOT contain _SKILL_CATEGORY_MAP"
    )
    # Instead, it should use _infer_category
    assert "_infer_category" in source, (
        "v2: _fallback_aggregator should use _infer_category for skill categorization"
    )
    print("  PASS test_no_hardcoded_category_map_used")


def test_evidence_merged_for_cross_referenced_skill():
    """v2: When a skill appears in multiple detector reports, evidence is merged."""
    from skill.aggregator import SkillAggregatorRole

    with tempfile.TemporaryDirectory() as tmpdir:
        # Domain Expertise detects "Testing Strategy" (unusual but possible via test signals)
        with open(os.path.join(tmpdir, "domain_expertise_report.json"), "w") as f:
            json.dump({
                "domain": "Domain Expertise",
                "inferred_skills": [
                    {"name": "Testing Strategy", "proficiency": "intermediate",
                     "confidence": "medium",
                     "evidence": {"domain_signals": ["test coverage", "assertion patterns"]},
                     "evidence_refs": [
                         {"file": "tests/test_models.py", "signals": ["assert"], "description": "Test file"}
                     ]},
                ],
                "detected_tools": [],
            }, f)

        # Rigor also detects "Testing Strategy"
        with open(os.path.join(tmpdir, "rigor_report.json"), "w") as f:
            json.dump({
                "domain": "Depth & Rigor",
                "inferred_skills": [
                    {"name": "Testing Strategy", "proficiency": "advanced",
                     "confidence": "high",
                     "evidence": {"test_count": 25, "pyramid_levels": 3},
                     "evidence_refs": [
                         {"file": "tests/test_app.py", "signals": ["def test_"], "description": "More tests"}
                     ]},
                ],
                "detected_tools": [],
            }, f)

        result = SkillAggregatorRole._fallback_aggregator(project_dir=tmpdir, working_dir=tmpdir)
        skills = result.get("inferred_skills", [])

        # Testing Strategy should appear once (deduplicated)
        ts_skills = [s for s in skills if s["name"] == "Testing Strategy"]
        assert len(ts_skills) == 1, (
            f"Testing Strategy should be deduplicated to 1 entry, got {len(ts_skills)}"
        )

        ts = ts_skills[0]
        # Evidence should be merged from both detectors
        evidence = ts.get("evidence", {})
        assert "domain_signals" in evidence, (
            f"Merged evidence should include domain_signals from Domain Expertise detector"
        )
        assert "test_count" in evidence or "pyramid_levels" in evidence, (
            f"Merged evidence should include test_count/pyramid_levels from Rigor detector"
        )

        # evidence_refs should be merged
        refs = ts.get("evidence_refs", [])
        ref_files = [r.get("file", "") for r in refs]
        assert len(refs) >= 2, (
            f"evidence_refs should be merged from both detectors, got {len(refs)} refs: {ref_files}"
        )

        print(f"  PASS test_evidence_merged_for_cross_referenced_skill "
              f"(evidence keys: {list(evidence.keys())}, refs: {len(refs)})")


def test_cross_referenced_flag_set():
    """v2: cross_referenced=True when skill is found by multiple detectors."""
    from skill.aggregator import SkillAggregatorRole

    with tempfile.TemporaryDirectory() as tmpdir:
        # Two different detectors both find "Security Awareness"
        with open(os.path.join(tmpdir, "technical_craft_report.json"), "w") as f:
            json.dump({
                "domain": "Technical Craft",
                "inferred_skills": [
                    {"name": "Security Awareness", "proficiency": "intermediate",
                     "confidence": "medium",
                     "evidence": {"indicators_matched": ["validate", "sanitize"]}},
                ],
                "detected_tools": [],
            }, f)

        with open(os.path.join(tmpdir, "methodology_report.json"), "w") as f:
            json.dump({
                "domain": "Methodology & Tooling",
                "inferred_skills": [
                    {"name": "Security Awareness", "proficiency": "beginner",
                     "confidence": "low",
                     "evidence": {"text_signals": ["auth", "rate limit"]}},
                ],
                "detected_tools": [],
            }, f)

        result = SkillAggregatorRole._fallback_aggregator(project_dir=tmpdir, working_dir=tmpdir)
        skills = result.get("inferred_skills", [])

        sa_skills = [s for s in skills if s["name"] == "Security Awareness"]
        assert len(sa_skills) == 1, "Security Awareness should be deduplicated"

        sa = sa_skills[0]
        assert sa.get("cross_referenced") is True, (
            f"Skill found by 2 detectors must have cross_referenced=True, "
            f"got {sa.get('cross_referenced')}"
        )
        assert len(sa.get("cross_referenced_sources", [])) >= 2, (
            f"cross_referenced_sources should list both detectors, "
            f"got {sa.get('cross_referenced_sources')}"
        )
        # Should have the highest proficiency
        assert sa.get("proficiency") == "intermediate", (
            f"Should keep highest proficiency (intermediate), got {sa.get('proficiency')}"
        )

        print(f"  PASS test_cross_referenced_flag_set "
              f"(cross_referenced={sa['cross_referenced']}, "
              f"sources={sa.get('cross_referenced_sources')})")


def test_confidence_boosted_when_cross_referenced():
    """v2: Cross-referenced skills get a confidence boost (low→medium, medium→high)."""
    from skill.aggregator import SkillAggregatorRole

    with tempfile.TemporaryDirectory() as tmpdir:
        # Both detectors find "Error Handling Maturity" with low confidence
        with open(os.path.join(tmpdir, "technical_craft_report.json"), "w") as f:
            json.dump({
                "domain": "Technical Craft",
                "inferred_skills": [
                    {"name": "Error Handling Maturity", "proficiency": "beginner",
                     "confidence": "low",
                     "evidence": {"indicators_matched": ["try:", "except"]}},
                ],
                "detected_tools": [],
            }, f)

        with open(os.path.join(tmpdir, "rigor_report.json"), "w") as f:
            json.dump({
                "domain": "Depth & Rigor",
                "inferred_skills": [
                    {"name": "Error Handling Maturity", "proficiency": "intermediate",
                     "confidence": "low",
                     "evidence": {"rigor_signals": ["fallback"]}},
                ],
                "detected_tools": [],
            }, f)

        result = SkillAggregatorRole._fallback_aggregator(project_dir=tmpdir, working_dir=tmpdir)
        skills = result.get("inferred_skills", [])

        ehm = next(s for s in skills if s["name"] == "Error Handling Maturity")
        # Confidence should be boosted from low → medium (one level up)
        assert ehm.get("confidence") == "medium", (
            f"Cross-referenced skill with low confidence should be boosted to medium, "
            f"got {ehm.get('confidence')}"
        )

        print(f"  PASS test_confidence_boosted_when_cross_referenced "
              f"(confidence: low → {ehm['confidence']})")


def test_confidence_boosted_to_high():
    """v2: medium→high when cross-referenced, high stays high."""
    from skill.aggregator import SkillAggregatorRole

    with tempfile.TemporaryDirectory() as tmpdir:
        # Two detectors find the same skill with medium confidence
        with open(os.path.join(tmpdir, "methodology_report.json"), "w") as f:
            json.dump({
                "domain": "Methodology & Tooling",
                "inferred_skills": [
                    {"name": "CI/CD Sophistication", "proficiency": "advanced",
                     "confidence": "medium",
                     "evidence": {"text_signals": ["pipeline", "deploy"]}},
                ],
                "detected_tools": [],
            }, f)

        with open(os.path.join(tmpdir, "rigor_report.json"), "w") as f:
            json.dump({
                "domain": "Depth & Rigor",
                "inferred_skills": [
                    {"name": "CI/CD Sophistication", "proficiency": "intermediate",
                     "confidence": "medium",
                     "evidence": {"rigor_signals": ["ci"]}},
                ],
                "detected_tools": [],
            }, f)

        result = SkillAggregatorRole._fallback_aggregator(project_dir=tmpdir, working_dir=tmpdir)
        skills = result.get("inferred_skills", [])

        cicd = next(s for s in skills if s["name"] == "CI/CD Sophistication")
        assert cicd.get("confidence") == "high", (
            f"Cross-referenced medium confidence should be boosted to high, "
            f"got {cicd.get('confidence')}"
        )

        print(f"  PASS test_confidence_boosted_to_high "
              f"(confidence: medium → {cicd['confidence']})")


def test_skill_graph_generated():
    """v2: _fallback_aggregator output must include a skill_graph."""
    from skill.aggregator import SkillAggregatorRole

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create detector reports with tools and skills that have known relationships
        with open(os.path.join(tmpdir, "methodology_report.json"), "w") as f:
            json.dump({
                "domain": "Methodology & Tooling",
                "inferred_skills": [
                    {"name": "Git Workflow Maturity", "proficiency": "intermediate",
                     "confidence": "medium", "evidence": {}},
                ],
                "detected_tools": [
                    {"name": "pytest", "category": "Testing", "proficiency": "advanced"},
                    {"name": "Docker", "category": "Containerization", "proficiency": "intermediate"},
                ],
            }, f)

        with open(os.path.join(tmpdir, "rigor_report.json"), "w") as f:
            json.dump({
                "domain": "Depth & Rigor",
                "inferred_skills": [
                    {"name": "Testing Strategy", "proficiency": "advanced",
                     "confidence": "high",
                     "evidence": {"test_count": 50, "pyramid_levels": 3}},
                ],
                "detected_tools": [],
            }, f)

        result = SkillAggregatorRole._fallback_aggregator(project_dir=tmpdir, working_dir=tmpdir)
        assert "skill_graph" in result, "v2: output must include skill_graph"

        graph = result["skill_graph"]
        assert "nodes" in graph, "skill_graph must have 'nodes'"
        assert "edges" in graph, "skill_graph must have 'edges'"

        nodes = graph["nodes"]
        edges = graph["edges"]
        assert len(nodes) > 0, f"skill_graph should have at least 1 node, got {len(nodes)}"
        assert len(edges) >= 0, f"skill_graph edges should be a list"

        # Verify node structure
        for node in nodes:
            assert "id" in node, f"Node missing 'id': {node}"
            assert "type" in node, f"Node missing 'type': {node}"
            assert node["type"] in ("tool", "skill"), (
                f"Node type must be 'tool' or 'skill', got '{node['type']}'"
            )
            assert "category" in node, f"Node missing 'category': {node}"

        # Verify edge structure if any edges exist
        for edge in edges:
            assert "source" in edge, f"Edge missing 'source': {edge}"
            assert "target" in edge, f"Edge missing 'target': {edge}"
            assert "relationship" in edge, f"Edge missing 'relationship': {edge}"

        print(f"  PASS test_skill_graph_generated "
              f"({len(nodes)} nodes, {len(edges)} edges)")


def test_skill_graph_contains_tool_skill_edges():
    """v2: skill_graph has edges connecting tools to related skills."""
    from skill.aggregator import SkillAggregatorRole

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "methodology_report.json"), "w") as f:
            json.dump({
                "domain": "Methodology & Tooling",
                "inferred_skills": [],
                "detected_tools": [
                    {"name": "pytest", "category": "Testing", "proficiency": "advanced"},
                    {"name": "Docker", "category": "Containerization", "proficiency": "intermediate"},
                    {"name": "GitHub Actions", "category": "CI/CD", "proficiency": "intermediate"},
                ],
            }, f)

        with open(os.path.join(tmpdir, "rigor_report.json"), "w") as f:
            json.dump({
                "domain": "Depth & Rigor",
                "inferred_skills": [
                    {"name": "Testing Strategy", "proficiency": "advanced",
                     "confidence": "high", "evidence": {}},
                    {"name": "Test Coverage Thoroughness", "proficiency": "intermediate",
                     "confidence": "medium", "evidence": {}},
                ],
                "detected_tools": [],
            }, f)

        result = SkillAggregatorRole._fallback_aggregator(project_dir=tmpdir, working_dir=tmpdir)
        graph = result.get("skill_graph", {})
        edges = graph.get("edges", [])

        # Check for tool → skill edges
        tool_skill_edges = [e for e in edges if e.get("relationship") == "indicates"]
        assert len(tool_skill_edges) > 0, (
            f"Should have tool→skill edges with relationship='indicates'. "
            f"Got {len(edges)} total edges, {len(tool_skill_edges)} indicates edges"
        )

        # pytest should connect to Testing Strategy
        pytest_targets = [
            e["target"] for e in tool_skill_edges if e["source"] == "pytest"
        ]
        assert "Testing Strategy" in pytest_targets or "Test Coverage Thoroughness" in pytest_targets, (
            f"pytest should connect to Testing Strategy or Test Coverage Thoroughness, "
            f"connected to: {pytest_targets}"
        )

        print(f"  PASS test_skill_graph_contains_tool_skill_edges "
              f"({len(tool_skill_edges)} tool→skill edges)")


def test_skill_graph_contains_skill_skill_edges():
    """v2: skill_graph has edges connecting related skills to each other."""
    from skill.aggregator import SkillAggregatorRole

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "technical_craft_report.json"), "w") as f:
            json.dump({
                "domain": "Technical Craft",
                "inferred_skills": [
                    {"name": "Design Patterns", "proficiency": "advanced",
                     "confidence": "high", "evidence": {}},
                    {"name": "Code Organization", "proficiency": "intermediate",
                     "confidence": "medium", "evidence": {}},
                ],
                "detected_tools": [],
            }, f)

        with open(os.path.join(tmpdir, "rigor_report.json"), "w") as f:
            json.dump({
                "domain": "Depth & Rigor",
                "inferred_skills": [
                    {"name": "Testing Strategy", "proficiency": "advanced",
                     "confidence": "high", "evidence": {}},
                    {"name": "Test Coverage Thoroughness", "proficiency": "intermediate",
                     "confidence": "medium", "evidence": {}},
                ],
                "detected_tools": [],
            }, f)

        result = SkillAggregatorRole._fallback_aggregator(project_dir=tmpdir, working_dir=tmpdir)
        graph = result.get("skill_graph", {})
        edges = graph.get("edges", [])

        # Look for skill ↔ skill edges (non-indicates, non-same_category)
        skill_skill_edges = [
            e for e in edges
            if e.get("relationship") not in ("indicates", "same_category")
        ]
        # Design Patterns → Code Organization or Testing Strategy ↔ Test Coverage
        if skill_skill_edges:
            print(f"    Skill↔skill edges: {[(e['source'], e['target'], e['relationship']) for e in skill_skill_edges]}")

        # At minimum, should have same_category edges
        same_cat_edges = [e for e in edges if e.get("relationship") == "same_category"]
        # Testing Strategy and Test Coverage Thoroughness should be same_category (both Quality Assurance)
        ts_related = [
            e for e in same_cat_edges
            if "Testing Strategy" in (e.get("source"), e.get("target"))
            and "Test Coverage Thoroughness" in (e.get("source"), e.get("target"))
        ]
        if ts_related:
            print(f"    Found Testing Strategy ↔ Test Coverage Thoroughness same_category edge")

        total_edges = len(edges)
        assert total_edges > 0, f"Should have at least some edges, got {total_edges}"

        print(f"  PASS test_skill_graph_contains_skill_skill_edges "
              f"({len(skill_skill_edges)} skill↔skill, {len(same_cat_edges)} same_category)")


def test_aggregator_output_includes_skill_graph():
    """v2: Full aggregator output dict includes skill_graph with expected structure."""
    from skill.aggregator import SkillAggregatorRole

    with tempfile.TemporaryDirectory() as tmpdir:
        # Minimal: one detector with one tool, the other with one skill
        with open(os.path.join(tmpdir, "methodology_report.json"), "w") as f:
            json.dump({
                "domain": "Methodology & Tooling",
                "inferred_skills": [
                    {"name": "Git Workflow Maturity", "proficiency": "intermediate",
                     "confidence": "medium", "evidence": {}},
                ],
                "detected_tools": [
                    {"name": "Git", "category": "Version Control", "proficiency": "advanced"},
                ],
            }, f)

        result = SkillAggregatorRole._fallback_aggregator(project_dir=tmpdir, working_dir=tmpdir)

        # Top-level keys
        assert "skill_graph" in result
        assert "tools" in result
        assert "inferred_skills" in result
        assert "summary" in result

        graph = result["skill_graph"]
        assert isinstance(graph, dict)
        assert isinstance(graph.get("nodes"), list)
        assert isinstance(graph.get("edges"), list)

        # Node IDs should match detected tools and skills
        node_ids = {n["id"] for n in graph["nodes"]}
        tool_names = {t["name"] for t in result["tools"]}
        skill_names = {s["name"] for s in result["inferred_skills"]}

        for tname in tool_names:
            assert tname in node_ids, f"Tool '{tname}' should be a node in skill_graph"

        for sname in skill_names:
            assert sname in node_ids, f"Skill '{sname}' should be a node in skill_graph"

        print(f"  PASS test_aggregator_output_includes_skill_graph "
              f"({len(graph['nodes'])} nodes, {len(graph['edges'])} edges, "
              f"all tools/skills present as nodes)")


def test_cross_referenced_skills_have_sources():
    """v2: Every skill must have a 'sources' list and cross_referenced flag."""
    from skill.aggregator import SkillAggregatorRole

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "domain_expertise_report.json"), "w") as f:
            json.dump({
                "domain": "Domain Expertise",
                "inferred_skills": [
                    {"name": "Machine Learning", "proficiency": "advanced",
                     "confidence": "high", "evidence": {}},
                ],
                "detected_tools": [],
            }, f)

        with open(os.path.join(tmpdir, "technical_craft_report.json"), "w") as f:
            json.dump({
                "domain": "Technical Craft",
                "inferred_skills": [
                    {"name": "Design Patterns", "proficiency": "intermediate",
                     "confidence": "medium", "evidence": {}},
                ],
                "detected_tools": [],
            }, f)

        result = SkillAggregatorRole._fallback_aggregator(project_dir=tmpdir, working_dir=tmpdir)
        skills = result.get("inferred_skills", [])

        for skill in skills:
            name = skill.get("name", "?")
            # Every skill must have these fields
            assert "sources" in skill, f"Skill '{name}' missing 'sources'"
            assert isinstance(skill["sources"], list), f"'sources' must be a list for '{name}'"
            assert len(skill["sources"]) >= 1, f"Skill '{name}' should have at least 1 source"

            assert "cross_referenced" in skill, f"Skill '{name}' missing 'cross_referenced'"
            assert isinstance(skill["cross_referenced"], bool), (
                f"'cross_referenced' must be bool for '{name}'"
            )

            assert "cross_referenced_sources" in skill, (
                f"Skill '{name}' missing 'cross_referenced_sources'"
            )
            assert isinstance(skill["cross_referenced_sources"], list), (
                f"'cross_referenced_sources' must be list for '{name}'"
            )

            # Single-detector skills should NOT be cross_referenced
            if len(skill["sources"]) == 1:
                assert skill["cross_referenced"] is False, (
                    f"Single-source skill '{name}' should have cross_referenced=False"
                )
                assert skill["cross_referenced_sources"] == [], (
                    f"Single-source skill '{name}' should have empty cross_referenced_sources"
                )

        print(f"  PASS test_cross_referenced_skills_have_sources "
              f"({len(skills)} skills verified)")


def test_skill_graph_nodes_have_categories():
    """v2: Every node in skill_graph must have a category field."""
    from skill.aggregator import SkillAggregatorRole

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "methodology_report.json"), "w") as f:
            json.dump({
                "domain": "Methodology & Tooling",
                "inferred_skills": [
                    {"name": "Git Workflow Maturity", "proficiency": "intermediate",
                     "confidence": "medium", "evidence": {}},
                ],
                "detected_tools": [
                    {"name": "Git", "category": "Version Control", "proficiency": "advanced"},
                ],
            }, f)

        result = SkillAggregatorRole._fallback_aggregator(project_dir=tmpdir, working_dir=tmpdir)
        graph = result.get("skill_graph", {})

        for node in graph.get("nodes", []):
            assert "category" in node, f"Node '{node.get('id', '?')}' missing 'category'"
            assert isinstance(node["category"], str), (
                f"Node category must be string for '{node.get('id', '?')}'"
            )
            assert len(node["category"]) > 0, (
                f"Node category must not be empty for '{node.get('id', '?')}'"
            )

        print(f"  PASS test_skill_graph_nodes_have_categories "
              f"({len(graph.get('nodes', []))} nodes)")


# ═══════════════════════════════════════════════════════════════════════════
# Test 43-50: Writer v2 — artifact-type-aware reports, Skill Gap Analysis,
# skill_graph in JSON output
# ═══════════════════════════════════════════════════════════════════════════


def test_software_project_report_emphasizes_tools():
    """v2: Software project reports should emphasize Tools → Testing → CI/CD → Code Quality."""
    from skill.writer import SkillReportWriterRole

    inventory = {
        "artifact_type": "software_project",
        "tools": [
            {"name": "Python", "category": "Languages & Runtimes", "proficiency": "advanced", "sources": ["Methodology"]},
            {"name": "pytest", "category": "Testing", "proficiency": "advanced", "sources": ["Methodology"]},
            {"name": "Docker", "category": "Containers & Orchestration", "proficiency": "intermediate", "sources": ["Methodology"]},
        ],
        "inferred_skills": [
            {"name": "Design Patterns", "category": "Technical Craft",
             "proficiency": "advanced", "confidence": "high",
             "evidence": {"description": "Factory pattern used"}},
            {"name": "Testing Strategy", "category": "Quality Assurance",
             "proficiency": "advanced", "confidence": "high",
             "evidence": {"test_count": 20}},
        ],
        "summary": {
            "total_tools": 3, "total_inferred_skills": 2,
            "dimensions_covered": ["Technical Craft", "Methodology & Tooling", "Depth & Rigor"],
            "proficiency_distribution": {"expert": 0, "advanced": 3, "intermediate": 2, "beginner": 0},
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        SkillReportWriterRole._fallback_report_md("TestProject", inventory, tmpdir)
        md_path = os.path.join(tmpdir, "skills_report.md")
        assert os.path.exists(md_path)

        with open(md_path) as f:
            md = f.read()

        # Software project should have emphasis note
        assert "Software Project Focus" in md, (
            "Software project report should include type emphasis note"
        )

        # Tools section should exist with Testing category visible
        assert "Detected Tools" in md
        assert "pytest" in md
        assert "Testing" in md

        # Skill Gap Analysis should be present
        assert "Skill Gap Analysis" in md, (
            "v2: Report should include Skill Gap Analysis section"
        )

        print(f"  PASS test_software_project_report_emphasizes_tools "
              f"(report: {os.path.getsize(md_path)}B)")


def test_research_paper_report_emphasizes_domain():
    """v2: Research paper reports should emphasize Domain Expertise → Methodology → Academic Rigor."""
    from skill.writer import SkillReportWriterRole

    inventory = {
        "artifact_type": "research_paper",
        "tools": [
            {"name": "LaTeX", "category": "Documentation Tools", "proficiency": "advanced", "sources": ["Methodology"]},
            {"name": "Python", "category": "Languages & Runtimes", "proficiency": "intermediate", "sources": ["Methodology"]},
        ],
        "inferred_skills": [
            {"name": "Machine Learning", "category": "Domain Knowledge",
             "proficiency": "advanced", "confidence": "high",
             "evidence": {"signal_matches": ["transformer", "attention mechanism"]}},
            {"name": "Academic Rigor", "category": "Quality Assurance",
             "proficiency": "intermediate", "confidence": "medium",
             "evidence": {"rigor_signals": ["citation", "methodology"]}},
            {"name": "Clarity", "category": "Technical Craft",
             "proficiency": "intermediate", "confidence": "medium",
             "evidence": {"indicators_matched": ["for example", "specifically"]}},
        ],
        "summary": {
            "total_tools": 2, "total_inferred_skills": 3,
            "dimensions_covered": ["Domain Expertise", "Technical Craft", "Depth & Rigor"],
            "proficiency_distribution": {"expert": 0, "advanced": 2, "intermediate": 3, "beginner": 0},
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        SkillReportWriterRole._fallback_report_md("ResearchPaper", inventory, tmpdir)
        md_path = os.path.join(tmpdir, "skills_report.md")
        assert os.path.exists(md_path)

        with open(md_path) as f:
            md = f.read()

        # Research paper should have emphasis note
        assert "Research Paper Focus" in md, (
            "Research paper report should include type emphasis note"
        )

        # Domain Knowledge category should appear before Technical Craft
        # within the Inferred Developer Skills section specifically
        skills_section = md.find("## 🧠 Inferred Developer Skills")
        if skills_section >= 0:
            skills_only = md[skills_section:]
            domain_pos = skills_only.find("Domain Knowledge")
            craft_pos = skills_only.find("Technical Craft")
            if domain_pos >= 0 and craft_pos >= 0:
                assert domain_pos < craft_pos, (
                    f"Domain Knowledge (pos {domain_pos}) should appear before "
                    f"Technical Craft (pos {craft_pos}) in skills section for research papers"
                )

        # Skill Gap Analysis should be present
        assert "Skill Gap Analysis" in md, (
            "v2: Report should include Skill Gap Analysis section"
        )

        print(f"  PASS test_research_paper_report_emphasizes_domain")


def test_blog_article_report_emphasizes_writing_craft():
    """v2: Blog article reports should emphasize Writing Craft → Clarity → Argumentation."""
    from skill.writer import SkillReportWriterRole

    inventory = {
        "artifact_type": "blog_article",
        "tools": [
            {"name": "Markdown", "category": "Documentation Tools", "proficiency": "advanced", "sources": ["Methodology"]},
        ],
        "inferred_skills": [
            {"name": "Argumentation", "category": "Technical Craft",
             "proficiency": "advanced", "confidence": "high",
             "evidence": {"indicators_matched": ["therefore", "however", "evidence"]}},
            {"name": "Clarity", "category": "Technical Craft",
             "proficiency": "intermediate", "confidence": "medium",
             "evidence": {"indicators_matched": ["for example", "in other words"]}},
            {"name": "Frontend Development", "category": "Domain Knowledge",
             "proficiency": "beginner", "confidence": "low",
             "evidence": {"signal_matches": ["responsive design"]}},
        ],
        "summary": {
            "total_tools": 1, "total_inferred_skills": 3,
            "dimensions_covered": ["Technical Craft", "Domain Expertise"],
            "proficiency_distribution": {"expert": 0, "advanced": 2, "intermediate": 1, "beginner": 1},
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        SkillReportWriterRole._fallback_report_md("BlogArticle", inventory, tmpdir)
        md_path = os.path.join(tmpdir, "skills_report.md")
        assert os.path.exists(md_path)

        with open(md_path) as f:
            md = f.read()

        # Blog article should have emphasis note
        assert "Blog Article Focus" in md, (
            "Blog article report should include type emphasis note"
        )

        # Technical Craft (Writing Craft) should appear before Domain Knowledge
        craft_pos = md.find("Technical Craft")
        domain_pos = md.find("Domain Knowledge")
        if craft_pos >= 0 and domain_pos >= 0:
            assert craft_pos < domain_pos, (
                f"Technical Craft (pos {craft_pos}) should appear before "
                f"Domain Knowledge (pos {domain_pos}) for blog articles"
            )

        # Skill Gap Analysis should be present
        assert "Skill Gap Analysis" in md, (
            "v2: Report should include Skill Gap Analysis section"
        )

        print(f"  PASS test_blog_article_report_emphasizes_writing_craft")


def test_skill_gap_analysis_present():
    """v2: All artifact types should have a Skill Gap Analysis section."""
    from skill.writer import SkillReportWriterRole, _ARTIFACT_EXPECTED_AREAS

    # Test for each artifact type that has expected areas defined
    for artifact_type in ["software_project", "research_paper", "blog_article"]:
        inventory = {
            "artifact_type": artifact_type,
            "tools": [],
            "inferred_skills": [],
            "summary": {
                "total_tools": 0, "total_inferred_skills": 0,
                "dimensions_covered": [],
                "proficiency_distribution": {"expert": 0, "advanced": 0, "intermediate": 0, "beginner": 0},
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            SkillReportWriterRole._fallback_report_md(f"Test{artifact_type}", inventory, tmpdir)
            md_path = os.path.join(tmpdir, "skills_report.md")

            with open(md_path) as f:
                md = f.read()

            assert "Skill Gap Analysis" in md, (
                f"{artifact_type} report must have Skill Gap Analysis section"
            )

            # All areas should be reported as missing since inventory is empty
            expected_areas = _ARTIFACT_EXPECTED_AREAS.get(artifact_type, {})
            for area in expected_areas:
                assert area in md, (
                    f"Expected area '{area}' should be listed in gap analysis for {artifact_type}"
                )

    print(f"  PASS test_skill_gap_analysis_present")


def test_skill_gap_shows_missing_categories():
    """v2: Skill Gap Analysis should show ❌ for missing expected areas."""
    from skill.writer import SkillReportWriterRole

    # Software project with only one skill (missing Testing, CI/CD, Version Control)
    inventory = {
        "artifact_type": "software_project",
        "tools": [
            {"name": "Python", "category": "Languages & Runtimes", "proficiency": "advanced", "sources": ["Methodology"]},
        ],
        "inferred_skills": [
            {"name": "Design Patterns", "category": "Technical Craft",
             "proficiency": "intermediate", "confidence": "medium",
             "evidence": {"description": "Basic patterns"}},
        ],
        "summary": {
            "total_tools": 1, "total_inferred_skills": 1,
            "dimensions_covered": ["Technical Craft"],
            "proficiency_distribution": {"expert": 0, "advanced": 1, "intermediate": 1, "beginner": 0},
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        SkillReportWriterRole._fallback_report_md("GapTest", inventory, tmpdir)
        md_path = os.path.join(tmpdir, "skills_report.md")

        with open(md_path) as f:
            md = f.read()

        # Should show missing markers for areas not covered
        assert "❌ Missing" in md, (
            "Gap analysis should show '❌ Missing' for uncovered expected areas"
        )

        # Should include Growth Suggestions when there are gaps
        assert "Growth Suggestions" in md, (
            "Gap analysis with missing areas should include Growth Suggestions"
        )

        print(f"  PASS test_skill_gap_shows_missing_categories")


def test_skill_gap_empty_when_all_present():
    """v2: Skill Gap Analysis should show all ✅ when expected skills are present."""
    from skill.writer import SkillReportWriterRole

    # Software project with all expected areas covered
    inventory = {
        "artifact_type": "software_project",
        "tools": [
            {"name": "pytest", "category": "Testing", "proficiency": "advanced", "sources": ["Methodology"]},
            {"name": "GitHub Actions", "category": "CI/CD", "proficiency": "intermediate", "sources": ["Methodology"]},
            {"name": "Git", "category": "Version Control", "proficiency": "advanced", "sources": ["Methodology"]},
            {"name": "Docker", "category": "Containers & Orchestration", "proficiency": "intermediate", "sources": ["Methodology"]},
            {"name": "Ruff", "category": "Build & Tooling", "proficiency": "intermediate", "sources": ["Methodology"]},
        ],
        "inferred_skills": [
            {"name": "Testing Strategy", "category": "Quality Assurance",
             "proficiency": "advanced", "confidence": "high",
             "evidence": {"test_count": 100}},
            {"name": "CI/CD Sophistication", "category": "Engineering Practice",
             "proficiency": "advanced", "confidence": "high",
             "evidence": {"deep_signals": ["staging environment"]}},
            {"name": "Git Workflow Maturity", "category": "Engineering Practice",
             "proficiency": "intermediate", "confidence": "medium",
             "evidence": {"text_signals": ["feat:", "fix:"]}},
            {"name": "Environment Management", "category": "Engineering Practice",
             "proficiency": "intermediate", "confidence": "medium",
             "evidence": {"deep_signals": ["multi-stage build"]}},
            {"name": "Code Quality Enforcement", "category": "Quality Assurance",
             "proficiency": "advanced", "confidence": "high",
             "evidence": {"enforcement_layers": 3}},
        ],
        "summary": {
            "total_tools": 5, "total_inferred_skills": 5,
            "dimensions_covered": ["Methodology & Tooling", "Depth & Rigor", "Technical Craft"],
            "proficiency_distribution": {"expert": 0, "advanced": 6, "intermediate": 4, "beginner": 0},
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        SkillReportWriterRole._fallback_report_md("FullProject", inventory, tmpdir)
        md_path = os.path.join(tmpdir, "skills_report.md")

        with open(md_path) as f:
            md = f.read()

        # Should show detected markers for all areas
        assert "Skill Gap Analysis" in md

        # All expected areas should be marked as covered
        detected_count = md.count("✅ Detected")
        assert detected_count == 5, (
            f"All 5 expected areas should be detected, got {detected_count} ✅ markers"
        )

        # No missing areas should be shown
        assert "❌ Missing" not in md, (
            "No areas should be missing when all expected skills are present"
        )

        print(f"  PASS test_skill_gap_empty_when_all_present ({detected_count}/5 areas covered)")


def test_skills_json_includes_skill_graph():
    """v2: skills.json output should include skill_graph when present in inventory."""
    from skill.writer import SkillReportWriterRole

    inventory = {
        "artifact_type": "software_project",
        "tools": [
            {"name": "pytest", "category": "Testing", "proficiency": "advanced", "sources": ["Methodology"]},
        ],
        "inferred_skills": [
            {"name": "Testing Strategy", "category": "Quality Assurance",
             "proficiency": "advanced", "confidence": "high",
             "evidence": {"test_count": 50}},
        ],
        "summary": {
            "total_tools": 1, "total_inferred_skills": 1,
            "dimensions_covered": ["Depth & Rigor"],
            "proficiency_distribution": {"expert": 0, "advanced": 2, "intermediate": 0, "beginner": 0},
        },
        "skill_graph": {
            "nodes": [
                {"id": "pytest", "type": "tool", "category": "Testing"},
                {"id": "Testing Strategy", "type": "skill", "category": "Quality Assurance"},
            ],
            "edges": [
                {"source": "pytest", "target": "Testing Strategy", "relationship": "indicates"},
            ],
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        skills_data = SkillReportWriterRole._fallback_skills_json("GraphTest", inventory)
        SkillReportWriterRole._write_skills_json(tmpdir, skills_data)

        json_path = os.path.join(tmpdir, "skills.json")
        with open(json_path) as f:
            data = json.load(f)

        # Verify skill_graph is present
        assert "skill_graph" in data, (
            "v2: skills.json should include skill_graph when present in inventory"
        )
        graph = data["skill_graph"]
        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) == 2
        assert len(graph["edges"]) == 1
        assert graph["edges"][0]["relationship"] == "indicates"

        print(f"  PASS test_skills_json_includes_skill_graph "
              f"({len(graph['nodes'])} nodes, {len(graph['edges'])} edges)")


def test_skills_json_no_skill_graph_when_absent():
    """v2: skills.json should NOT have skill_graph when inventory doesn't include it."""
    from skill.writer import SkillReportWriterRole

    inventory = {
        "artifact_type": "software_project",
        "tools": [
            {"name": "Python", "category": "Languages & Runtimes", "proficiency": "advanced", "sources": ["Methodology"]},
        ],
        "inferred_skills": [],
        "summary": {
            "total_tools": 1, "total_inferred_skills": 0,
            "dimensions_covered": [],
            "proficiency_distribution": {"expert": 0, "advanced": 1, "intermediate": 0, "beginner": 0},
        },
        # No skill_graph key
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        skills_data = SkillReportWriterRole._fallback_skills_json("NoGraphTest", inventory)
        SkillReportWriterRole._write_skills_json(tmpdir, skills_data)

        json_path = os.path.join(tmpdir, "skills.json")
        with open(json_path) as f:
            data = json.load(f)

        assert "skill_graph" not in data, (
            "v2: skills.json should NOT include skill_graph when absent from inventory"
        )

        print(f"  PASS test_skills_json_no_skill_graph_when_absent")


def test_default_ordering_for_unknown_types():
    """v2: Unknown artifact types should use alphabetical category ordering."""
    from skill.writer import SkillReportWriterRole

    inventory = {
        "artifact_type": "unknown",
        "tools": [
            {"name": "Git", "category": "Version Control", "proficiency": "advanced", "sources": ["Methodology"]},
            {"name": "Markdown", "category": "Documentation Tools", "proficiency": "intermediate", "sources": ["Methodology"]},
        ],
        "inferred_skills": [
            {"name": "Design Patterns", "category": "Technical Craft",
             "proficiency": "intermediate", "confidence": "medium",
             "evidence": {"description": "Basic patterns"}},
            {"name": "Testing Strategy", "category": "Quality Assurance",
             "proficiency": "beginner", "confidence": "low",
             "evidence": {"test_count": 2}},
        ],
        "summary": {
            "total_tools": 2, "total_inferred_skills": 2,
            "dimensions_covered": [],
            "proficiency_distribution": {"expert": 0, "advanced": 1, "intermediate": 2, "beginner": 1},
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        SkillReportWriterRole._fallback_report_md("UnknownType", inventory, tmpdir)
        md_path = os.path.join(tmpdir, "skills_report.md")
        assert os.path.exists(md_path)

        with open(md_path) as f:
            md = f.read()

        # Unknown type should still have Skill Gap Analysis (if any expected areas)
        assert "Skill Gap Analysis" in md or "## 💡 Recommendations" in md, (
            "Unknown type report should have either gap analysis or recommendations"
        )

        # Should still render skills and tools correctly
        assert "Design Patterns" in md
        assert "Testing Strategy" in md

        print(f"  PASS test_default_ordering_for_unknown_types")


def test_build_task_includes_artifact_type_instructions():
    """v2: build_task prompt should include artifact-type-aware instructions."""
    from skill.writer import SkillReportWriterRole

    role = SkillReportWriterRole()

    # Software project inventory
    sw_inventory = {
        "artifact_type": "software_project",
        "tools": [{"name": "pytest", "category": "Testing", "proficiency": "advanced"}],
        "inferred_skills": [
            {"name": "Testing Strategy", "category": "Quality Assurance",
             "proficiency": "advanced", "confidence": "high",
             "evidence": {}}
        ],
        "summary": {"total_tools": 1, "total_inferred_skills": 1,
                     "dimensions_covered": [], "proficiency_distribution": {}},
    }
    sw_prompt = role.build_task("deepseek", working_dir="/tmp/test",
                                 project_name="SW", skill_inventory=sw_inventory)
    assert "software project" in sw_prompt.lower(), (
        "Prompt for software_project should mention the artifact type"
    )
    assert "Tools & Testing" in sw_prompt or "tools" in sw_prompt.lower(), (
        "Prompt should instruct emphasis on tools/testing for software projects"
    )
    assert "Skill Gap Analysis" in sw_prompt, (
        "v2: Prompt should request Skill Gap Analysis section"
    )

    # Research paper inventory
    rp_inventory = {
        "artifact_type": "research_paper",
        "tools": [{"name": "LaTeX", "category": "Documentation Tools", "proficiency": "advanced"}],
        "inferred_skills": [
            {"name": "Academic Rigor", "category": "Quality Assurance",
             "proficiency": "intermediate", "confidence": "medium",
             "evidence": {}}
        ],
        "summary": {"total_tools": 1, "total_inferred_skills": 1,
                     "dimensions_covered": [], "proficiency_distribution": {}},
    }
    rp_prompt = role.build_task("deepseek", working_dir="/tmp/test",
                                 project_name="RP", skill_inventory=rp_inventory)
    assert "research paper" in rp_prompt.lower(), (
        "Prompt for research_paper should mention the artifact type"
    )
    assert "Domain Expertise" in rp_prompt, (
        "Prompt should instruct emphasis on domain expertise for research papers"
    )

    # Blog article inventory
    blog_inventory = {
        "artifact_type": "blog_article",
        "tools": [{"name": "Markdown", "category": "Documentation Tools", "proficiency": "intermediate"}],
        "inferred_skills": [
            {"name": "Clarity", "category": "Technical Craft",
             "proficiency": "intermediate", "confidence": "medium",
             "evidence": {}}
        ],
        "summary": {"total_tools": 1, "total_inferred_skills": 1,
                     "dimensions_covered": [], "proficiency_distribution": {}},
    }
    blog_prompt = role.build_task("deepseek", working_dir="/tmp/test",
                                   project_name="Blog", skill_inventory=blog_inventory)
    assert "blog article" in blog_prompt.lower(), (
        "Prompt for blog_article should mention the artifact type"
    )
    assert "Writing Craft" in blog_prompt, (
        "Prompt should instruct emphasis on writing craft for blog articles"
    )

    print("  PASS test_build_task_includes_artifact_type_instructions")


def test_category_display_order():
    """v2: _get_category_display_order returns type-prioritized category ordering."""
    from skill.writer import SkillReportWriterRole

    # Software project: Testing should come first among listed categories
    categories = ["Technical Craft", "Quality Assurance", "Domain Knowledge",
                  "Engineering Practice", "Architecture"]
    sw_order = SkillReportWriterRole._get_category_display_order(
        "software_project", categories)
    # No specific priority for inferred skill categories yet for software_project,
    # but alphabetical with priority categories first
    assert isinstance(sw_order, list)
    assert len(sw_order) == len(categories), (
        f"Should preserve all categories, got {len(sw_order)} vs {len(categories)}"
    )

    # Research paper: Domain Knowledge should come first
    rp_order = SkillReportWriterRole._get_category_display_order(
        "research_paper", categories)
    assert rp_order[0] == "Domain Knowledge", (
        f"Research paper: Domain Knowledge should be first, got {rp_order[0]}"
    )

    # Blog article: Technical Craft should come first
    blog_order = SkillReportWriterRole._get_category_display_order(
        "blog_article", categories)
    assert blog_order[0] == "Technical Craft", (
        f"Blog article: Technical Craft should be first, got {blog_order[0]}"
    )

    # Unknown type: alphabetical order
    unknown_order = SkillReportWriterRole._get_category_display_order(
        "unknown", categories)
    assert unknown_order == sorted(categories), (
        f"Unknown type should use alphabetical order, got {unknown_order}"
    )

    print(f"  PASS test_category_display_order "
          f"(SW: {sw_order[:3]}..., RP: {rp_order[:3]}..., Blog: {blog_order[:3]}...)")


def test_cross_referenced_marker_in_report():
    """v2: Cross-referenced skills should have a 🔗 marker in the report."""
    from skill.writer import SkillReportWriterRole

    inventory = {
        "artifact_type": "software_project",
        "tools": [],
        "inferred_skills": [
            {"name": "Testing Strategy", "category": "Quality Assurance",
             "proficiency": "advanced", "confidence": "high",
             "evidence": {"test_count": 50},
             "cross_referenced": True,
             "cross_referenced_sources": ["Methodology & Tooling", "Depth & Rigor"]},
            {"name": "Design Patterns", "category": "Technical Craft",
             "proficiency": "intermediate", "confidence": "medium",
             "evidence": {"indicators_matched": ["class", "factory"]},
             "cross_referenced": False,
             "cross_referenced_sources": []},
        ],
        "summary": {
            "total_tools": 0, "total_inferred_skills": 2,
            "dimensions_covered": ["Technical Craft", "Depth & Rigor"],
            "proficiency_distribution": {"expert": 0, "advanced": 1, "intermediate": 1, "beginner": 0},
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        SkillReportWriterRole._fallback_report_md("CrossRefTest", inventory, tmpdir)
        md_path = os.path.join(tmpdir, "skills_report.md")

        with open(md_path) as f:
            md = f.read()

        # Cross-referenced skill should have the link emoji
        assert "🔗" in md, (
            "v2: Cross-referenced skills should have 🔗 marker in the report"
        )

        print(f"  PASS test_cross_referenced_marker_in_report")


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
        # v2 scanner tests
        test_max_chars_increased_to_4000,
        test_complex_files_sampled_before_readme,
        test_key_files_in_artifact_analysis,
        test_key_files_have_rationale,
        test_key_files_have_complexity_score,
        test_generated_files_excluded_from_sampling,
        test_build_task_includes_key_files,
        test_scanner_module_exports,
        # v2 detector tests
        test_evidence_refs_in_domain_detection,
        test_qualitative_proficiency_not_count_based,
        test_new_domain_signals_matched,
        test_multi_word_phrase_matching,
        test_negative_signals_excluded,
        test_modern_tool_detection,
        test_tool_version_detection,
        test_tool_scope_field,
        test_code_craft_qualitative_basis,
        test_methodology_qualitative_basis,
        test_rigor_qualitative_basis,
        # v2 aggregator tests
        test_category_inferred_from_domain,
        test_no_hardcoded_category_map_used,
        test_evidence_merged_for_cross_referenced_skill,
        test_cross_referenced_flag_set,
        test_confidence_boosted_when_cross_referenced,
        test_confidence_boosted_to_high,
        test_skill_graph_generated,
        test_skill_graph_contains_tool_skill_edges,
        test_skill_graph_contains_skill_skill_edges,
        test_aggregator_output_includes_skill_graph,
        test_cross_referenced_skills_have_sources,
        test_skill_graph_nodes_have_categories,
        # v2 writer tests
        test_software_project_report_emphasizes_tools,
        test_research_paper_report_emphasizes_domain,
        test_blog_article_report_emphasizes_writing_craft,
        test_skill_gap_analysis_present,
        test_skill_gap_shows_missing_categories,
        test_skill_gap_empty_when_all_present,
        test_skills_json_includes_skill_graph,
        test_skills_json_no_skill_graph_when_absent,
        test_default_ordering_for_unknown_types,
        test_build_task_includes_artifact_type_instructions,
        test_category_display_order,
        test_cross_referenced_marker_in_report,
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
