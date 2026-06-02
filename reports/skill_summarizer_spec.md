# Skill Summarizer Pipeline — Requirement Specification

## Overview

Build a **SkillPipeline** that scans a software project directory and produces a structured skill inventory: programming languages, frameworks, libraries, design patterns, domain knowledge, and tooling, with proficiency levels (detected/used/extensively-used) plus a human-readable markdown report.

## Optimized Topology (from Topology Optimizer)

**Pattern**: fan_out_fan_in (Domain-Parallel Detection, 36/50)
**Agents**: 7 total — 1 scanner, 4 domain-specific detectors, 1 aggregator, 1 report writer

```
[project_scanner]
   |
   ├── [python_detector]
   ├── [js_detector]
   ├── [infra_detector]
   └── [config_docs_detector]
   |
[aggregator]
   |
[report_writer] → skills.json + skills_report.md
```

## Pipeline State

```python
class SkillState(TypedDict):
    input_spec: str          # Project directory path or description
    working_dir: str
    backend: str
    project_scan: dict[str, Any]  # Scanner output: file tree, technology signals
    detector_outputs: list[dict[str, Any]]  # Results from domain detectors
    skill_inventory: dict[str, Any]  # Aggregated skill inventory
    status: str
```

## Agent Roles

### 1. SkillScannerRole
**File**: `skill/__init__.py` (all roles in one file for simplicity, or split across files)
**agent_name**: `skill_scanner`
**max_steps**: 6

Scans the project directory structure:
- Run `find` / `ls` to catalog files by extension
- Identify key config files (package.json, requirements.txt, Dockerfile, etc.)
- Detect project type signals (monorepo vs single project, language mix)
- Output: `project_scan.json` with file tree summary, identified languages, key config files list

**Tools**: `Read`, `Bash` (for claude_cli); `read_file`, `run_command` (for deepseek)

### 2. Domain Detector Agents (4 parallel)

All share the same base pattern. Each reads relevant files and outputs a domain-specific JSON skills report.

#### 2a. PythonDetectorRole
**agent_name**: `python_detector`, **max_steps**: 8
Detects: Python version, pip packages, frameworks (Django, Flask, FastAPI), testing (pytest, unittest), linting (ruff, black), type checking (mypy), data science (numpy, pandas, torch), patterns used

#### 2b. JSDetectorRole
**agent_name**: `js_detector`, **max_steps**: 8
Detects: Node.js version, npm/yarn/pnpm packages, frameworks (React, Vue, Next.js), testing (jest, vitest), build tools (webpack, vite, esbuild), TypeScript usage

#### 2c. InfraDetectorRole
**agent_name**: `infra_detector`, **max_steps**: 6
Detects: Docker, Kubernetes, CI/CD (GitHub Actions, GitLab CI), cloud providers, deployment patterns, IaC (Terraform, Pulumi)

#### 2d. ConfigDocsDetectorRole
**agent_name**: `config_docs_detector`, **max_steps**: 6
Detects: Config formats (YAML, TOML, JSON), documentation tooling, API specs (OpenAPI, GraphQL), project management tools

Each detector outputs: `{domain}_skills.json` with `{"skills": [{"name": "...", "category": "...", "proficiency": "detected|used|extensively-used", "evidence": "found in file X, used across N files"}]}`

### 3. SkillAggregatorRole
**agent_name**: `skill_aggregator`, **max_steps**: 8
- Reads all 4 domain skill reports
- Deduplicates entries (same library detected by multiple detectors)
- Resolves proficiency levels (extensively-used > used > detected)
- Categorizes skills: languages, frameworks, libraries, tools, patterns, domains
- Output: `skill_inventory.json` with unified, deduplicated skill catalog

### 4. SkillReportWriterRole
**agent_name**: `skill_report_writer`, **max_steps**: 6
- Reads the aggregated skill inventory
- Produces `skills.json` (structured, machine-readable)
- Produces `skills_report.md` (human-readable, organized by category with proficiency badges)
- Tools: Write only

## Pipeline Class: SkillPipeline

**File**: `pipeline.py` (append after TopologyPipeline)
**Class**: `SkillPipeline(BasePipeline)`
**name**: `"skill"`
**default_output_dir**: `"skill_output"`

### Flow
```
scanner → detectors (parallel via _run_parallel_agents) → aggregator → writer → END
```

Detectors run in parallel (no dependencies between them). All depend on scanner completing first. Aggregator depends on all 4 detectors. Writer depends on aggregator.

Each node reads relevant state, calls the corresponding AgentRole.execute(), and updates state.

## ToolRegistry Extensions

Add class methods to `ToolRegistry` in `tools.py`:
- `skill_scanner_tools()` — `Read`, `Bash`
- `skill_detector_tools()` — `Read`, `Bash` (shared by all 4 detectors)
- `skill_aggregator_tools()` — `Read`, `Write`
- `skill_writer_tools()` — `Write` only

## Integration

Register in `main.py`:
```python
PIPELINES = {
    ...
    "skill": SkillPipeline,
}
```

## Tests

Create `test_skill.py` with:
1. All agent roles instantiate correctly
2. ToolRegistry methods exist
3. SkillState has all keys
4. SkillPipeline instantiates
5. Fallback scanner works (run find/ls on a temp dir)
6. Fallback aggregator deduplicates correctly
7. Fallback report writer produces valid markdown
8. E2E fallback chain without LLM calls

## Key Design Constraints

- Follow UMAF patterns: `AgentRole` ABC, `ToolRegistry`, `BasePipeline` with LangGraph
- Python >= 3.11 syntax
- Backend-aware task prompts
- No duplicated tool definitions
- Checkpoint support via CheckpointManager
- All agent logs written to `agent_log/` subdirectory
