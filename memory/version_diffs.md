---
name: version-diffs
description: "Complete changelog: v1.0→v2.0 (Feature Pipeline v2 multi-coder parallelism, Skill Pipeline v2 evidence-based assessment, 8 pipelines, 39 roles, 480 tests)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9c942f95-5fb5-4276-8e53-c16059ca5e31
---

## v2.0 (June 2026) — Feature Pipeline v2 + Skill Pipeline v2

**Why:** Transform single-agent pipelines into multi-agent parallel systems with dependency-aware execution; replace simplistic count-based skill detection with evidence-based, multi-dimensional assessment that works across any artifact type.

### Feature Pipeline v2 — Multi-Coder Parallelism
- Planner now produces `sub_tasks` with dependency graph (like Research head_agent)
- `_coders_node` replaces single `_coder_node` — coders execute in topological levels via `BasePipeline._topological_levels()`
- Within each level coders run in parallel via `_run_parallel_agents()` using `_feature_coder_worker()` entry point
- Dependency injection: coders at level[i] receive and verify outputs from level[i-1] via `_dependency_outputs` → `completed` dict
- Cross-coder integration review (5 dimensions: dependency consumption, import resolution, interface matching, data flow, integration tests)
- Dependency verification tokens: DEPENDENCY_VERIFIED / DEPENDENCY_ISSUE:
- 3 new FeatureState fields: `sub_tasks`, `coder_outputs`, `dependency_graph`
- `_MAX_CODER_RETRIES=3` with version-bump loop; fallback to single-coder when no sub_tasks
- `_build_dependency_graph()` for display and analysis
- **Verified**: 6 coders, 4 topological levels, all deps verified, REVIEW_PASSED in 1 iteration

### Skill Pipeline v2 — Evidence-Based Detection
- **Qualitative proficiency**: `_assess_proficiency()` replaces count-based model with multi-dimensional scoring (depth × consistency × integration − penalties)
- **evidence_refs**: every detected skill now has specific file paths with signal descriptions
- **19 domains** (up from 9): added Computer Vision, RL, Networking, OS, Embedded, DevOps, Data Engineering, Frontend, Mobile, Blockchain
- **Multi-word phrase matching** reduces false positives; negative signals exclude false matches
- **25+ modern tools**: uv, ruff, biome, pnpm, bun, Svelte, SolidJS, Playwright, Vitest, TailwindCSS, shadcn/ui, etc.
- **Version detection** from config files for tool proficiency
- **Project vs ecosystem tool distinction**
- **Scanner v2**: file complexity scoring via `_STRUCTURAL_KEYWORDS`, generated/minified file filtering, 4000-char samples, `key_files` list
- **Aggregator v2**: `_infer_category()` replaces hardcoded `_SKILL_CATEGORY_MAP`, evidence merging across detectors, `cross_referenced` flag with confidence boost, `skill_graph` generation
- **Writer v2**: `_ARTIFACT_EXPECTED_AREAS` for Skill Gap Analysis, `_ARTIFACT_CATEGORY_ORDER` for type-specific section ordering, `_ARTIFACT_TOOL_CATEGORY_ORDER` for tool table grouping

### Test Expansion
- 480 tests (up from 403): test_skill.py +2255, test_feature.py +905 (multi-coder execution, topological levels, dependency injection, cross-coder verification)
- 7.91s runtime (parallel, default)

### Verified
- Feature Pipeline v2: 6 coders, 4 topological levels, all dependency verification passed, REVIEW_PASSED in 1 iteration
- Skill Pipeline v2: enhanced detection with 19 domains, 25+ tools, evidence_refs, skill_graph — end-to-end verified
- All 480 tests pass

## v1.9 (June 2026) — Plan Pipeline + Retry Loops + Test Optimization

**Why:** Enable structured implementation planning from natural language descriptions; add built-in agent retry so every pipeline gets version-bump resilience for free; add topology designer↔evaluator feedback loop; upgrade Feature pipeline to match CoderPP retry capabilities; optimize test suite with default parallel execution.

### Plan Pipeline (8th pipeline)
- 6-node fan-out/fan-in graph: scanner → decomposer → 4 parallel analyzers → writer → END
- 7 AgentRoles in `plan/`: `PlanScannerRole`, `PlanDecomposerRole`, `PlanDependencyAnalyzerRole`, `PlanRiskAssessorRole`, `PlanResourceEstimatorRole`, `PlanCrossCuttingAnalyzerRole`, `PlanWriterRole`
- Transforms natural language task descriptions into structured implementation plans with dependency graphs, risk matrices, resource estimates, and cross-cutting concern maps
- Guard clauses: scanner skips when `file_manifest` exists; decomposer skips when `task_tree.tree` exists — enables pipeline resume and testable pre-populated state
- 4 parallel analyzers (1 for claude_cli, 4 for deepseek); writer produces `plan_spec.json` + `plan_report.md`

### AgentRole Built-in Retry
- `_MAX_RETRIES = 3` in `AgentRole` base class with auto version-bump loop in `execute()`
- Every agent gets retry resilience without pipeline-level code — each attempt produces separate checkpoint + log file
- Prior context loaded via `CheckpointManager.load_previous()` on each retry
- `ClaudeCLILLM._invoke_stream()`: simplified — if files were written before timeout, treat as success rather than false-negative failure

### TopologyPipeline Retry Loop
- Designer↔evaluator feedback loop: if best score < 35/50 and retries remain, routes back to designer with dimensional feedback
- `_MAX_RETRIES = 3`, `_SCORE_THRESHOLD = 35`
- New `iteration` and `evaluation_feedback` fields in `TopologyState`
- Designer accepts `evaluation_feedback` parameter for targeted improvements
- Evaluator outputs `evaluation_feedback` listing low-scoring dimensions

### FeaturePipeline Version-Aware Retry
- Coder↔reviewer loop upgraded from simple `iteration` counter to version-bump pattern matching CoderPP
- `_MAX_VERSIONS = 5`, `version` field in `FeatureState` (14 fields total)
- `project_dir` passthrough to coder and reviewer nodes (was missing — agents couldn't find project files)
- Router uses `version > _MAX_VERSIONS` instead of `iteration >= 5`

### ToolRegistry & Config
- 7 new classmethods for Plan pipeline (scanner, decomposer, 4 analyzers, writer) — 35 total
- `tools_config.json`: Added `plan` section with per-role tool assignments
- `main.py`: 8 modes — coder, research, coderpp, topology, skill, feature, self_evolution, plan
- `pipeline/__init__.py`: exports `PlanPipeline` + `PlanState`

### Test Optimization
- **Removed duplicate suite**: test_feature_v2.py deleted (533 lines, 55 tests — identical coverage to test_feature.py)
- **Default parallelism**: `-n auto --timeout=30` in `pyproject.toml` addopts; `pythonpath = ["."]`
- **Fixed missing mocks**: SelfEvolutionWriterRole (1 slow test, was 5.48s), CoderRole (2 tests), decompose functions (5 tests)
- **New tests**: test_plan.py (integration tests with guard-aware mocking); test_topology.py (retry loop tests); test_feature.py (3 versioning tests)
- **Dependencies**: `pytest-xdist>=3.0`, `pytest-timeout>=2.0` added to requirements.txt

### Verified
- 403/403 tests pass in 3.72s (parallel, default) — down from 458/458 in ~7s
- 8 pipelines, 39 AgentRoles, 35 ToolRegistry methods, 8 tools

### Related
[[architecture_progress]], [[key_updates]], [[oop_refactoring]]

---

## v1.8 (June 2026) — Self-Evolution Pipeline + Test Enhancement

**Why:** UMAF should be able to analyze and improve itself. Tests only verified syntax/structure, not actual behavior.

### Self-Evolution Pipeline
- 7-node graph: analyzer → planner → coder ↔ reviewer (max 3 iterations) → writer → END
- 5 AgentRoles in `self_evolution/`: `SelfEvolutionAnalyzerRole`, `SelfEvolutionPlannerRole`, `SelfEvolutionCoderRole`, `SelfEvolutionReviewerRole`, `SelfEvolutionWriterRole`
- `SelfEvolutionPipeline(BasePipeline)` in `pipeline/self_evolution.py`
- `SelfEvolutionState` TypedDict (12 fields)
- 5 new ToolRegistry classmethods (`self_evolution_analyzer_tools()` through `self_evolution_writer_tools()`) in `registry.py`
- Removed redundant `tools/self_evolution_tools.py` (methods were duplicated in registry.py)
- `tools_config.json`: Added `self_evolution` section with per-role tool assignments
- `main.py`: 7 modes — coder, research, coderpp, topology, skill, feature, self_evolution
- Safety: Operates in current git branch; all changes revertible

### Test Enhancement (175 new behavioral tests)
- **test_coder.py** (14→27): Graph node behavior (file scanning, verdict detection, reverse-scan, router), build_task truncation, full loop simulation
- **test_research.py** (21→62): `parse_result` for all 3 roles, `_extract_json_array` bracket counting, flow routing, resume state reconstruction
- **test_coderpp.py** (26→58): `parse_result` for decomposer/worker/reviewer/observer, worker file scanning, review.md authoritative override, flow routing, resume with ENVIRONMENT.md
- **test_self_evolution.py**: 49 new tests (5 roles, pipeline, graph nodes)
- New test files: test_pipeline.py, test_coder.py, test_coderpp.py, test_research.py, test_feature.py, conftest.py
- All tests now verify functionality and expected behavior, not just syntax/structure

### Cleanup
- Removed redundant files: `_run_all_tests.py`, `_run_fast_tests.py`, `_run_tests.py`, `_run_timed_tests.py`, `_test_hang.py`, `review_verdict.txt`
- Removed `tools/self_evolution_tools.py` (duplicate of registry.py methods)
- Cleaned up `tools/__init__.py` (removed self_evolution_tools import)

### Verified
- 379/379 tests pass
- 7 pipelines, 32 concrete AgentRole subclasses (+ 3 abstract = 35 total)
- 8 tools, 23 ToolRegistry role methods

---

## v1.7 (June 2026) — tools_config.json + Codebase Cleanup

**Why:** Tool assignments were hardcoded across 18+ role methods. ~200 lines of duplicated utility code. Dead imports and unused functions.

### tools_config.json
- Single source of truth for per-role tool assignments
- Auto-loaded by `main.py` at startup
- `--tools-config` flag overrides with custom file
- All `ToolRegistry.*_tools()` defaults changed from hardcoded lists to `[]` — `set_tool_config()` must be called first
- Metadata keys (`__about__`, `_description`, etc.) stripped on load
- `__global__` key provides fallback for unlisted roles

### Code Deduplication
- 5 `_extract_json_object` copies consolidated into `utils.py`
- 2 `_extract_json_array` copies moved to `utils.py`
- 4 `sys.path.insert` hacks removed from `skill/`
- `_PROFICIENCY_SCORES` centralized (was 5 inline copies)
- Added `extract_json_array()` to `utils.py`

### Dead Code Removal
- `run_agent()`, `BaseAgent._checkpoint_path()`, `_checkpoint_path()` from `agent.py`
- `_load_config()`, `_claude_env`, `get_claude_env()` from `claude_config.py`
- Unused imports from `coderpp/head_agent.py`, `research/head_agent.py`, `pipeline/__init__.py`

### Backend-Agnostic Tool Defaults
- Removed backend-differentiated tool lists in `research_decomposer_tools()`
- Removed "do NOT search the web" prompt restrictions
- Tools are now purely config-driven

### Verified
- 99/99 tests pass

---

## v1.6.1 (June 2026) — Dependency Injection Fixes Across 3 Pipelines

**Why:** Audit revealed 3 pipelines had dependency gaps where upstream agent outputs never reached dependent downstream agents. Coder reviewer was blind to coder output; Skill downstream agents relied on disk-based file discovery despite having data in LangGraph state; CoderPP workers_node completely bypassed `_run_workers_with_deps()`.

### CoderPipeline Fix
- Added `coder_files: list[str]` to `MultiAgentState` TypedDict
- `_coder_node` now scans working directory for produced files after coder runs
- `ReviewerRole.build_task()` now accepts `coder_files: list[str] = []` and renders "Files Produced by Coder" section

### SkillPipeline Fix
- `_detectors_node` embeds `project_scan` in each detector's item dict; `_run_detector` passes it to `role.execute(project_scan=...)`
- `_aggregator_node` passes `detector_outputs` from state to `role.execute(detector_outputs=...)`
- `_writer_node` passes `skill_inventory` from state to `role.execute(skill_inventory=...)`
- All 4 detector `build_task()` methods now accept `project_scan` and render inline summary via `_format_scan_summary()`
- Aggregator `build_task()` accepts `detector_outputs` with inline domain status/skills table
- Writer `build_task()` accepts `skill_inventory` with inline skill list preview

### CoderPP Fix (Critical)
- `_workers_node` had its own topological level iteration but called `_run_parallel_agents()` directly — never injected `_dependency_outputs`
- Added `completed` dict + dependency resolution before each topological level + dual-key registration after each level
- Verified: 3-worker test (string_utils → validator → cli), dependency injection confirmed in checkpoints, 131/131 tests passing

### Verified
- All 97 tests pass after all changes
- CoderPP end-to-end: dependency injection confirmed in worker checkpoints

---
## v1.6 (June 2026) — Feature Pipeline + Modular Package Structure

**Why:** Flat files were growing unwieldy (pipeline.py: 2,334 lines, tools.py+tools_integration.py: 1,101 lines). Needed a pipeline for adding/editing code in existing projects (not just generating new modules).

### Feature Pipeline
- 5-node graph: scanner → planner → coder ↔ reviewer (max 5 cycles) → writer
- 5 AgentRoles in `feature/`: `FeatureScannerRole`, `FeaturePlannerRole`, `FeatureCoderRole`, `FeatureReviewerRole`, `FeatureReportWriterRole`
- First pipeline supporting both `files_to_create` AND `files_to_modify`
- REVIEW_PASSED/REVIEW_FAILED token scanning pattern (reused from CoderPipeline)
- `FeatureState` TypedDict (12 fields), `FeaturePipeline(BasePipeline)` in pipeline/feature.py
- 5 new ToolRegistry classmethods in `tools/feature_tools.py`, auto-applied at import time

### Modular Package Structure
- **`pipeline.py` → `pipeline/`** (7 modules): base (550 lines), coder (168), research (464), coderpp (708), topology (179), skill (278), feature (210). All 5 concrete pipelines + base. Backward compatible via `__init__.py` re-exports.
- **`tools.py` + `tools_integration.py` → `tools/`** (3 modules): registry.py (ToolSpec + ToolRegistry), functions.py (7 implementations), feature_tools.py (5 feature role methods). `__init__.py` auto-applies feature patches at import time.
- **`feature_pipeline.py` → `feature/`** (5 role files + `__init__.py`): functional modules contain only agent role definitions.
- **Test files → `test/`** directory: test_smoke.py, test_topology.py, test_skill.py, test_feature_v2.py (55 tests)
- **`utils.py`**: shared helpers — `extract_json_object()`, `safe_read()`
- **`main.py`**: 6 modes (added --mode feature)

### Verified
- All 6 pipelines pass end-to-end with claude_cli backend:
  - Coder: fibonacci function, 2 iterations, review passed
  - Topology: 3-agent RAG pipeline, 37/50
  - Skill: 21 skills detected (Python + Infra)
  - Feature: palindrome.py + tests, 1 iteration, review passed
  - Research: 7/7 workers, scores 40-46/50, LaTeX generated
  - CoderPP: 2/2 workers, 2/2 reviewed, project assembled
- All 6 pipelines pass with deepseek backend
- 97 tests pass (42 legacy + 55 feature)

---
## v1.5 (June 2026) — Topology Optimizer + Skill Summarizer Pipelines

**Why:** Extend UMAF from 3 pipelines to 5 by building two new pipelines via meta-programming (CoderPP generates both).

### New pipelines

**Topology Optimizer** (`--mode topology`): Given a task description, determines the optimal multi-agent topology.
- 4-node linear graph: analyzer → designer → evaluator → writer
- 4 AgentRole subclasses in `topology/`: `TopologyAnalyzerRole`, `TopologyDesignerRole`, `TopologyEvaluatorRole`, `TopologyWriterRole`
- Analyzer assesses 6 complexity factors; Designer proposes 2-4 candidate topologies using 4 patterns (sequential, fan_out_fan_in, debate_consensus, hierarchical)
- Evaluator scores on 5 dimensions (latency, reliability, cost_efficiency, simplicity, scalability) each 1-10
- Writer produces `topology_spec.json` + `topology_report.md`
- `TopologyState` TypedDict, `TopologyPipeline(BasePipeline)` in pipeline.py

**Skill Summarizer** (`--mode skill`): Scans a project directory, extracts structured skill inventory.
- 4-node fan-out/fan-in graph: scanner → 4 parallel detectors → aggregator → writer
- 7 AgentRole subclasses in `skill/`: `SkillScannerRole`, `PythonDetectorRole`, `JSDetectorRole`, `InfraDetectorRole`, `ConfigDocsDetectorRole`, `SkillAggregatorRole`, `SkillReportWriterRole`
- Domain-parallel detection: Python, JavaScript, Infra, ConfigDocs each handled by specialized detector
- Aggregator deduplicates and categorizes skills; Writer produces `skills.json` + `skills_report.md`
- `SkillState` TypedDict, `SkillPipeline(BasePipeline)` in pipeline.py

### ToolRegistry additions
- 4 new classmethods: `topology_analyzer_tools()`, `topology_designer_tools()`, `topology_evaluator_tools()`, `topology_writer_tools()`
- 5 entry points in main.py: coder, research, coderpp, topology, skill

### Meta-programming approach
- CoderPP generated both pipelines from `.md` spec files (extended `_decompose()` to read `.md` in addition to `.tex`)
- Topology Optimizer validated by designing the Skill Summarizer topology (fan-out/fan-in with domain-specific detectors — an excellent design)
- Skill Summarizer verified on this repo: 33 skills across 11 categories

### Code
- `topology/`: 5 files, ~1200 lines (analyzer, designer, evaluator, writer, __init__)
- `skill/`: 5 files, ~2400 lines (scanner, detectors, aggregator, writer, __init__)
- `pipeline.py`: 2108 lines (+457 in v1.5)
- `tools.py`: +4 ToolRegistry methods

### Verified
- Topology Optimizer end-to-end with claude_cli: `topology_spec.json` (20KB), `topology_report.md` (16KB)
- Skill Summarizer end-to-end with claude_cli: `skills.json` (11KB), `skills_report.md` (10KB), 33 skills detected
- 42 smoke tests pass (15 core + 14 topology + 13 skill)
- Fallback chain verified: all agent roles have deterministic fallback methods

---

## v1.4.1 (June 2026) — 8 Bug Fixes

**Why:** Smoke tests revealed edge cases in agent loop, checkpointing, and error handling.

### Changes
| # | Area | Fix |
|---|------|-----|
| 1 | Agent loop | Tool calls executed BEFORE TASK_COMPLETE check (responses with both write+tools now work) |
| 2 | Force wrap-up | Stronger: final steps forbid all tools except write_file |
| 3 | Post-loop | Exhaustion message explicitly requires writing file immediately |
| 4 | CheckpointManager | Fixed version bump context injection |
| 5 | Error spiral | 3→2 consecutive errors threshold tightened |
| 6 | Unknown tools | Warning dedup per agent session |
| 7 | Mid-loop | Write reminder at ~2/3 of max steps if write_file not yet called |
| 8 | Smoke tests | 15 smoke tests added for agent/pipeline core |

### Verified
15/15 smoke tests pass; agent loop handles edge cases robustly.

---

## v1.4 (June 2026) — OOP Refactoring + Pipeline Robustness

**Why:** Procedural code with duplicated tool definitions across 8+ files; research pipeline needed dependency management.

### OOP Architecture
- 5-layer class hierarchy: Data types → Infrastructure → Agent core → Concrete roles → Pipeline classes
- `AgentRole` ABC with template method: `tools_for_backend()`, `build_task()`, `parse_result()`, `execute()`
- `ToolRegistry` centralization: no duplicated tool definitions (was 8+ copies)
- 3 dead `graph.py` files removed (replaced by `pipeline.py`)
- `AgentResult` dataclass, `ToolSpec` dataclass, `LLMProvider` ABC

### Pipeline Robustness
- **Stop-on-failure**: `_run_workers_with_deps` breaks out of topological level loop on failure, blocks downstream
- **Version-bump retry**: Failed workers retry with `version+1` → `CheckpointManager.load_previous(version)` restores messages
- **Honest `parse_result`**: `ResearchWorkerRole.parse_result()` checks `os.path.isfile()` before reporting success
- **Worker retry state machine**: `worker_retry` status, max 3 retries, max 4 versions
- **Timeout**: Worker timeout 300s → 600s

### CoderPP Pipeline
- Multi-file code generation: organizer → workers → reviewer
- 4 new roles: `CoderPPDecomposerRole`, `CoderPPWorkerRole`, `CoderPPReviewerRole`, `OrganizerRole`

### Verified
7/7 workers (100%); scores 48, 47, 45, 44, 43, 39, 38/50; 60KB LaTeX; 443s pipeline time.

---

## v1.3.1 (May 2026) — Worker Output Fix & arxiv.org Access

**Why:** Workers produced TASK_COMPLETE with write_file calls that were never executed.

### Changes
- Reordered agent loop: execute tool calls BEFORE checking TASK_COMPLETE
- Mid-loop write reminder at ~2/3 of max steps
- Stronger force wrap-up: forbid all tools except write_file
- `download_file` tool: framework-level urllib download → local file
- Pre-fetch layer: arxiv.org content pre-downloaded for claude_cli workers
- Default working dir: `tempfile.mkdtemp()` → `research_output/` in repo

### Verified
4/4 workers produce files (up from 2/4); scores 47, 46, 43, 41/50.

---

## v1.3 (May 2026) — Python 3.11 & Code Quality

**Why:** Python 3.9 patterns deprecated; `_latex_escape()` had latent bug.

### Changes
- Python >= 3.11: `Optional[X]` → `X | None`; `.python-version` set to 3.11
- Bug fix: `_latex_escape()` backslash → tab (raw string fix)
- Dynamic decomposition: sub-topic count 2-8 based on complexity (was fixed 5-7)
- New tool: `web_fetch` (urllib-based, bypasses Claude Code permissions)
- Dead code removed: `_TOOL_NAME_TRANSLATION`, `_build_system_prompt`
- Simplifications: retry path, head_agent prompt, research router

### Verified
8 unit tests pass; end-to-end coder pipeline verified.

---

## v1.2 (May 2026) — Backend-Aware Agents

**Why:** Workers used nested `claude -p` calls causing recursive invocations and timeouts.

### Changes
- Backend-aware worker tasks: claude_cli workers use native tools; deepseek workers use `call_claude`
- Head agent Read-only for claude_cli; scoped permissions; conversation logger
- Security: `claude_env_sample.json` → `.example.json` template
- Timeout 120s→300s, parallelism 4→2, `--allowedTools` always passed

### Verified
4/6 workers (21-26KB each); top score 43/50; 41KB LaTeX.

---

## v1.1 (May 2026) — 12 Bug Fixes

### Critical
| # | Issue | Fix |
|---|-------|-----|
| 1 | Retry used untranslated task → workers fail tools | `task` → `translated_task` |
| +1 | `claude -p` wrote to project root | `cwd=working_dir` in subprocess.run |

### High
| # | Issue | Fix |
|---|-------|-----|
| 2 | `"error"` substring caused false retries | Check `"error:"`, `"[stderr]"`, `"timed out"` |
| 3 | Sequential workers → 35min worst-case | ThreadPoolExecutor, max 4 concurrent |
| 4 | Daemon threads orphaned processes on timeout | ThreadPoolExecutor with subprocess.run timeout |

### Medium
| # | Issue | Fix |
|---|-------|-----|
| 5 | Tool name translation missed patterns | `\b` word-boundary regex |
| 6 | Stale `review_passed=True` skipped reviewer | Coder resets `review_passed=False` |
| 7 | Fallback decomposition always 5 generic titles | Keyword extraction, pad to ≥5 |

### Low
| # | Issue | Fix |
|---|-------|-----|
| 8 | "Only 0 step(s) remaining" | Special-case: "This is your LAST step" |
| 9 | Regex couldn't parse nested JSON args | Brace-counting extraction |
| 10 | Greedy regex matched across arrays | Non-greedy `[\s\S]*?` |
| 11 | Only `&`/`%` escaped in LaTeX | All 10 special chars |

### Before/After (v1.0→v1.1)
| Metric | Before | After |
|--------|--------|-------|
| Worker completion | 3/7 (43%) | 5/5 (100%) |
| Top score | 25/50 (auto-rank) | 46/50 (real) |
| Pipeline time | 35min (sequential) | 9min (parallel) |
| Research files | 0 | 40KB + JSON |

### Related
[[architecture_progress]], [[key_updates]], [[oop_refactoring]]
