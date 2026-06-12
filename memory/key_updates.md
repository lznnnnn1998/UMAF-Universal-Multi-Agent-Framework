---
name: key-updates
description: "Key takeaways — verified metrics across versions, critical bug summaries, architecture decisions that matter most"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9c942f95-5fb5-4276-8e53-c16059ca5e31
---

## Verified Metrics by Version

| Metric | v1.0 | v1.1 | v1.2 | v1.3 | v1.3.1 | v1.4 | v1.4.1 | v1.5 | v1.6 | v1.7 | v1.8 | v1.9 | **v2.0** |
|--------|------|------|------|------|--------|------|--------|------|------|------|------|------|------|
| Worker completion | 3/7 | 5/5 | 4/6 | — | 4/4 | 7/7 | — | — | — | — | — | — | — |
| Top score | 25/50 | 46/50 | 43/50 | — | 47/50 | 48/50 | — | — | — | — | — | — | — |
| Pipeline time | 35min | 9min | 35min | — | 12min | 7.4min | — | — | — | — | — | — |
| LaTeX output | 3.6KB | 40KB | 41KB | — | gen'd | 60KB | — | — | — | — | — | — |
| Skills detected | — | — | — | — | — | — | — | 33 | 21 | — | — | — |
| Unit/smoke tests | 0 | 0 | 0 | 8/8 | 8/8 | 8/8 | 15/15 | 42/42 | 97/97 | 99/99 | 379/379 | 403/403 | **480/480** |
| Test time | — | — | — | — | — | — | — | — | — | — | ~7s | 3.72s | **7.91s** |
| Python | 3.9 | 3.9 | 3.9 | 3.11 | 3.11 | 3.11 | 3.11 | 3.11 | 3.11 | 3.11 | 3.11 | 3.11 | 3.11 |
| Pipelines | 2 | 2 | 2 | 2 | 2 | 3 | 3 | 5 | 6 | 6 | 7 | 8 | 8 |
| AgentRoles | — | — | — | — | — | 14 | 14 | 18 | 23 | 23 | 32 | 39 | 39 |
| Tools | 5 | 5 | 5 | 6 | 7 | 7 | 7 | 7 | 7 | 8 | 8 | 8 | 8 |

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

## New in v2.0

### Feature Pipeline v2 — Multi-Coder Parallelism
- Planner decomposes features into `sub_tasks` with dependency graph
- Coders execute in topological levels: level[i] receives and verifies level[i-1] outputs
- `_run_parallel_agents()` within each level; `_feature_coder_worker()` entry point
- Cross-coder integration review: 5 dimensions (dependency consumption, import resolution, interface matching, data flow, integration tests)
- DEPENDENCY_VERIFIED / DEPENDENCY_ISSUE: token scanning
- Fallback to single-coder when no sub_tasks
- **Verified**: 6 coders, 4 topological levels, all deps verified, REVIEW_PASSED in 1 iteration

### Skill Pipeline v2 — Evidence-Based Assessment
- `_assess_proficiency()`: depth (signal specificity) × consistency (cross-file distribution) × integration (co-occurrence) − negative penalty
- `evidence_refs` on every detected skill with specific file paths
- 19 domains (up from 9): Computer Vision, RL, Networking, OS, Embedded, etc.
- 25+ modern tools: uv, ruff, biome, Playwright, Vitest, TailwindCSS, etc.
- `_infer_category()` replaces hardcoded `_SKILL_CATEGORY_MAP`
- `skill_graph` with cross-referenced skills and confidence boost
- Artifact-type-aware report structure with Skill Gap Analysis
- Scanner: file complexity scoring, generated file detection, 4000-char samples, `key_files`

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

## New in v1.9

### Plan Pipeline (8th pipeline)
- 6-node fan-out/fan-in: scanner → decomposer → 4 parallel analyzers → writer
- 7 AgentRoles in `plan/`: `PlanScannerRole`, `PlanDecomposerRole`, `PlanDependencyAnalyzerRole`, `PlanRiskAssessorRole`, `PlanResourceEstimatorRole`, `PlanCrossCuttingAnalyzerRole`, `PlanWriterRole`
- Transforms natural language task descriptions into structured implementation plans
- Guard clauses for resume/testability

### AgentRole Built-in Retry
- `_MAX_RETRIES = 3` in `AgentRole` base class with auto version-bump loop
- Every agent gets retry resilience without pipeline-level code
- Each attempt produces separate checkpoint + log file; prior context loaded via `load_previous()`

### TopologyPipeline Retry Loop
- Designer↔evaluator feedback loop: if best_score < 35/50, routes back with dimensional feedback
- `_MAX_RETRIES = 3`, `_SCORE_THRESHOLD = 35`
- New `iteration` and `evaluation_feedback` fields in `TopologyState`

### FeaturePipeline Version-Aware Retry
- Upgraded from `iteration` counter to version-bump pattern matching CoderPP
- `_MAX_VERSIONS = 5`, `version` field in `FeatureState`
- `project_dir` passthrough to coder/reviewer nodes

### Test Optimization
- test_feature_v2.py deleted (55 duplicate tests)
- `-n auto --timeout=30` in pyproject.toml — parallel by default
- Fixed 8 missing mocks: tests no longer call real LLMs
- 403/403 tests pass in 3.72s (down from 458/458 in ~7s)

### Architecture Decisions
- **Built-in agent retry** (v1.9): `AgentRole._MAX_RETRIES=3` — every agent gets version-bump retry with checkpoint context reuse for free. Pipelines no longer need their own retry logic for basic agent failures.
- **Topology designer↔evaluator retry** (v1.9): Quality feedback loop. Evaluator identifies low dimensions, designer improves, re-evaluates. Threshold 35/50, max 3 retries.
- **Feature version-aware retry** (v1.9): Matches CoderPP's pattern. Each failed review bumps version → new checkpoint preserves prior context. `_MAX_VERSIONS=5`.
- **Plan Pipeline** (v1.9): 8th pipeline. 6-node fan-out/fan-in with 4 parallel analyzers. Guard clauses for resume/testability.

### Related
[[version_diffs]], [[architecture_progress]], [[oop_refactoring]]
