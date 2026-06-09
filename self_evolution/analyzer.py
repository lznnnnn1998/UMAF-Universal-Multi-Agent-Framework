"""Self-evolution analyzer — scans UMAF's own codebase and agent logs."""

from __future__ import annotations

import json
import os
from typing import Any

from agent import AgentResult, AgentRole
from tools import ToolRegistry


class SelfEvolutionAnalyzerRole(AgentRole):
    """Analyze UMAF's codebase structure and execution logs to identify improvement opportunities.

    Scans the project directory for code structure, agent logs for failure patterns,
    and writes an analysis report identifying concrete areas for improvement.
    """

    agent_name = "self_evolution_analyzer"
    max_steps = 20

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        return ToolRegistry.to_dicts(ToolRegistry.self_evolution_analyzer_tools())

    def build_task(self, backend: str, working_dir: str = "",
                   project_dir: str = ".", **context: Any) -> str:
        return f"""You are a self-evolution analyzer for the Universal Multi-Agent Framework (UMAF). Your job is to analyze UMAF's own codebase and execution logs to identify concrete, actionable improvements.

## Project Context
UMAF is a LangChain + DeepSeek multi-agent framework with 6 pipelines, 2 LLM backends, and 23 agent roles. It lives at `{project_dir}`.

## Analysis Steps

### 1. Scan Codebase Structure
Run `find {project_dir} -name "*.py" -not -path "*/.git/*" -not -path "*/__pycache__/*" -not -path "*/node_modules/*" | head -60` to understand the file layout. Read key files to understand:
- Pipeline architecture (pipeline/*.py)
- Agent role definitions (*/head_agent.py, */worker_agent.py, etc.)
- Tool registry and configuration
- Test coverage

### 2. Analyze Agent Logs
Check `agent_log/` directories in pipeline output folders for execution patterns. Read a few representative log files to understand:
- Which agents succeed vs fail most often
- Common failure modes (timeout, parse error, missing output)
- Average iteration counts
- Tool call patterns

### 3. Identify Improvement Opportunities
Based on your analysis, categorize findings into:
- **Prompt Quality**: vague, ambiguous, or missing instructions in build_task() methods
- **Parameter Tuning**: timeouts, max_steps, retry limits that are too low/high
- **Error Handling**: missing fallbacks, ungraceful failure modes
- **Code Quality**: duplication, dead code, missing type hints, naming inconsistencies
- **Test Gaps**: modules or behaviors with poor test coverage
- **Configuration**: tools_config.json improvements, missing tool assignments

### 4. Write Analysis Report
Write `analysis_report.json` to the working directory with this structure:
```json
{{
  "project_overview": {{
    "total_python_files": <int>,
    "pipelines": ["coder", "research", "coderpp", "topology", "skill", "feature"],
    "agent_roles": <int>
  }},
  "log_analysis": {{
    "logs_found": <bool>,
    "common_failure_modes": ["...", "..."],
    "avg_success_rate": <float>
  }},
  "improvement_opportunities": [
    {{
      "id": "SEO-001",
      "category": "prompt_quality | parameter_tuning | error_handling | code_quality | test_gaps | configuration",
      "title": "Short description",
      "description": "Detailed explanation of what's wrong and why it matters",
      "severity": "high | medium | low",
      "files_involved": ["path/to/file.py"],
      "suggested_fix": "Concrete suggestion for how to fix it"
    }}
  ],
  "summary": "Overall analysis summary in 2-3 sentences."
}}
```

Read every file you can. Be thorough. Write `analysis_report.json` then output TASK_COMPLETE."""

    def parse_result(self, result: AgentResult, working_dir: str,
                     project_dir: str = ".", **context: Any) -> dict[str, Any]:
        report_path = os.path.join(working_dir, "analysis_report.json")
        if os.path.exists(report_path):
            try:
                with open(report_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        # Fallback: extract JSON from agent messages
        from utils import extract_json_object
        for msg in reversed(result.messages):
            content = getattr(msg, "content", str(msg))
            if not isinstance(content, str):
                continue
            extracted = extract_json_object(content)
            if extracted:
                try:
                    parsed = json.loads(extracted)
                    if "improvement_opportunities" in parsed:
                        return parsed
                except json.JSONDecodeError:
                    continue

        return self._fallback_analyze(project_dir, working_dir)

    @staticmethod
    def _fallback_analyze(project_dir: str, working_dir: str) -> dict[str, Any]:
        """Deterministic fallback: scan the project directory for basic metrics."""
        py_files = []
        for root, dirs, files in os.walk(project_dir):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "node_modules", ".venv")]
            for f in files:
                if f.endswith(".py"):
                    py_files.append(os.path.relpath(os.path.join(root, f), project_dir))

        pipelines = []
        for f in py_files:
            if f.startswith("pipeline/") and f != "pipeline/__init__.py" and f != "pipeline/base.py":
                name = os.path.splitext(os.path.basename(f))[0]
                pipelines.append(name)

        return {
            "project_overview": {
                "total_python_files": len(py_files),
                "pipelines": pipelines,
                "agent_roles": 23,
            },
            "log_analysis": {
                "logs_found": False,
                "common_failure_modes": [],
                "avg_success_rate": 0.0,
            },
            "improvement_opportunities": [
                {
                    "id": "SEO-001",
                    "category": "test_gaps",
                    "title": "Expand test coverage for pipeline behavioral tests",
                    "description": "Several pipeline tests only verify structural properties (agent_name, max_steps) rather than actual behavior (parse_result, routing logic, file handling).",
                    "severity": "medium",
                    "files_involved": ["test/test_coder.py", "test/test_research.py", "test/test_coderpp.py"],
                    "suggested_fix": "Add behavioral tests that mock AgentRole.execute() and verify graph node state transitions, parse_result logic, and routing behavior.",
                },
            ],
            "summary": (
                f"Analyzed UMAF codebase at {project_dir}: {len(py_files)} Python files across "
                f"{len(pipelines)} pipelines. Primary improvement opportunity: expand test coverage "
                f"for behavioral verification of pipeline functionality."
            ),
        }