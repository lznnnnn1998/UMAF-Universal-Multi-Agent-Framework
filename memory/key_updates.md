---
name: key-updates
description: "Key takeaways — verified metrics across versions, critical bug summaries, architecture decisions that matter most"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9c942f95-5fb5-4276-8e53-c16059ca5e31
---

## Verified Metrics by Version

| Metric | v1.0 | v1.1 | v1.2 | v1.3 | v1.3.1 | v1.4 | v1.4.1 | v1.5 | v1.6 |
|--------|------|------|------|------|--------|------|--------|------|------|
| Worker completion | 3/7 (43%) | 5/5 (100%) | 4/6 (67%) | — | 4/4 (100%) | 7/7 (100%) | — | — | — |
| Top score | 25/50 | 46/50 | 43/50 | — | 47/50 | 48/50 | — | — | — |
| Pipeline time | 35min | 9min | 35min | — | 12min | 7.4min | — | — | — |
| LaTeX output | 3.6KB | 40KB | 41KB | — | generated | 60KB | — | — | — |
| Skills detected | — | — | — | — | — | — | — | 33 (11 categories) | 21 (claude_cli) |
| Unit/smoke tests | 0 | 0 | 0 | 8/8 | 8/8 | 8/8 | 15/15 | 42/42 | 97/97 | 99/99 | **379/379** |
| Python | 3.9 | 3.9 | 3.9 | 3.11 | 3.11 | 3.11 | 3.11 | 3.11 | 3.11 | 3.11 | 3.11 |
| Pipelines | 2 | 2 | 2 | 2 | 2 | 3 | 3 | 5 | 6 | 6 | **7** |

## Critical Bugs Fixed (Across All Versions)

1. **cwd sandboxing** (v1.1): `claude -p` wrote to project root. Fix: `cwd=working_dir`.
2. **Retry with untranslated task** (v1.1): Workers failed tools on retry. Fix: use `translated_task`.
3. **Sequential workers** (v1.1): 35min worst-case. Fix: ThreadPoolExecutor.
4. **Nested `claude -p`** (v1.2): Workers spawned recursive invocations. Fix: backend-aware tasks.
5. **Empty `--allowedTools`** (v1.2): No flag → all tools → permission denied. Fix: always pass.
6. **LaTeX backslash** (v1.3): `"\\textbackslash "` produced tab. Fix: raw string.
7. **Tool-before-TASK_COMPLETE** (v1.3.1): `write_file` calls never executed. Fix: reorder agent loop.
8. **Double-parse bug** (v1.5): `AgentRole.execute()` already calls `parse_result()`. Fix: don't call again on return value.

## Architecture Decisions That Matter

- **Backend-aware task generation** (v1.2): Most impactful decision. `claude_cli` agents get different tasks because the agent IS the runtime. No nesting.
- **Router always moves forward**: Partial results accepted at every stage. Without this, any single worker failure deadlocks.
- **Fallbacks at every stage**: Decompose → keywords, research → best-effort, scoring → auto-rank 25/50, LaTeX → template, topology → heuristic, skills → default. Pipeline never crashes.
- **Explicit `working_dir`**: No global state. Every tool, subprocess, and file operation is sandboxed.
- **OOP 5-layer hierarchy** (v1.4): `AgentRole` ABC + `ToolRegistry` centralization eliminates duplicated tool definitions across 16+ roles.
- **Meta-programming with CoderPP** (v1.5): Topology Optimizer + Skill Summarizer both built by CoderPP from `.md` specs. Validated: the topology it designed (fan-out/fan-in with domain-specific detectors) was the right architecture.

## New in v1.5

### Topology Optimizer
- 4-node linear graph: analyzer → designer → evaluator → writer
- Designs optimal agent topology for any task description
- 4 topology patterns: sequential, fan_out_fan_in, debate_consensus, hierarchical
- Verified: successfully designed Skill Summarizer topology

### Skill Summarizer  
- 4-node fan-out/fan-in graph: scanner → 4 parallel detectors → aggregator → writer
- Domain-parallel detection: Python, JS, Infra, ConfigDocs
- Detected 33 skills across 11 categories on UMAF repo
- Skills: langchain (extensively-used), DeepSeek API (extensively-used), LangGraph (used), ThreadPoolExecutor (used), DuckDuckGo (used), pytest (detected), urllib (extensively-used), argparse (used), dataclasses (extensively-used), subprocess (extensively-used), Claude CLI (extensively-used), etc.

## New in v1.6.1

### Dependency Injection Fixes (3 Pipelines)

**CoderPipeline** — Reviewer was blind to coder output:
- Added `coder_files: list[str]` to `MultiAgentState`
- `_coder_node` scans working directory for produced files after coder runs
- `ReviewerRole.build_task()` now accepts `coder_files` and renders "Files Produced by Coder" section

**SkillPipeline** — Upstream data in state never reached downstream agents:
- Detectors now receive `project_scan` via `execute()` with inline prompt summary
- Aggregator now receives `detector_outputs` via `execute()` with domain summary table
- Writer now receives `skill_inventory` via `execute()` with skill list preview

**CoderPPPipeline** — `_workers_node` bypassed dependency injection entirely:
- Was calling `_run_parallel_agents()` directly instead of `_run_workers_with_deps()`
- Fix: added `completed` dict + `_dependency_outputs` injection + dual-key registration directly in `_workers_node`
- Verified: 3-worker test with transitive deps, all reviewers passed, 131/131 tests

### Related
[[version_diffs]], [[architecture_progress]]

## New in v1.6

### Feature Pipeline
- 5-node graph: scanner → planner → coder ↔ reviewer (max 5 cycles) → writer
- First pipeline supporting BOTH `files_to_create` AND `files_to_modify` paths
- REVIEW_PASSED/REVIEW_FAILED token scanning pattern (reused from CoderPipeline)
- 5 new AgentRoles in `feature/`; 5 new ToolRegistry classmethods in `tools/feature_tools.py`

### Modular Package Structure
- `pipeline.py` (2,334 lines) → `pipeline/` (7 modules): base, coder, research, coderpp, topology, skill, feature
- `tools.py` (518 lines) + `tools_integration.py` (583 lines) → `tools/` (3 modules): registry, functions, feature_tools
- `feature_pipeline.py` (979 lines) → `feature/` (5 role files)
- Test files moved to `test/` directory
- `utils.py` — shared helpers (extract_json_object, safe_read)
- Backward compatible: all imports via `__init__.py` re-exports

### Verified
- All 6 pipelines pass end-to-end with claude_cli backend
- All 6 pipelines pass end-to-end with deepseek backend
- 97 tests pass (42 legacy + 55 feature)

## New in v1.7

### tools_config.json + Codebase Cleanup
- **tools_config.json**: Single source of truth for per-role tool assignments. Auto-loaded by `main.py`. `--tools-config` overrides. All `ToolRegistry.*_tools()` defaults changed from hardcoded lists to `[]` — `set_tool_config()` must be called first. `__global__` fallback support.
- **Code deduplication**: Removed ~200 lines — 5 `_extract_json_object` copies consolidated into `utils.py`, 2 `_extract_json_array` copies moved, 4 `sys.path.insert` hacks removed from `skill/`, `_PROFICIENCY_SCORES` centralized.
- **Dead code removal**: `run_agent()`, `_checkpoint_path()`, `_load_config()`, unused imports.
- **Backend-agnostic tool defaults**: Removed backend-differentiated tool lists, removed "do NOT search the web" prompt restrictions.
- **Verified**: 99/99 tests pass.

## New in v1.8

### Self-Evolution Pipeline
- 7-node graph: analyzer → planner → coder ↔ reviewer (max 3 iterations) → writer → END
- 5 new AgentRoles in `self_evolution/`: `SelfEvolutionAnalyzerRole`, `SelfEvolutionPlannerRole`, `SelfEvolutionCoderRole`, `SelfEvolutionReviewerRole`, `SelfEvolutionWriterRole`
- Analyzer scans UMAF codebase + agent logs for improvement opportunities
- Planner creates implementation plan from analysis findings
- Coder implements changes, detects via git diff or mtime
- Reviewer verifies with test suite, REVIEW_PASSED/REVIEW_FAILED token scanning
- Writer produces evolution_report.md
- 5 new ToolRegistry classmethods in `registry.py` (removed redundant `tools/self_evolution_tools.py`)
- `tools_config.json`: Added self_evolution section

### Test Enhancement (175 new behavioral tests)
- test_coder.py: 14→27 (graph node behavior, verdict detection, full loop simulation)
- test_research.py: 21→62 (parse_result for all 3 roles, JSON parsing, flow routing, resume state)
- test_coderpp.py: 26→58 (parse_result, worker file scanning, review.md override, resume)
- test_self_evolution.py: 49 new tests (5 roles, pipeline, graph nodes)
- New test files: test_pipeline.py, test_coder.py, test_coderpp.py, test_research.py, test_feature.py, conftest.py

### Verified
- 379/379 tests pass
- 7 pipelines, 32 concrete AgentRole subclasses, 8 tools, 23 ToolRegistry role methods
- Removed redundant files: `_run_*.py` temporary test runners, `_test_hang.py`, `review_verdict.txt`, `tools/self_evolution_tools.py`

### Related
[[version_diffs]], [[architecture_progress]], [[oop_refactoring]]
