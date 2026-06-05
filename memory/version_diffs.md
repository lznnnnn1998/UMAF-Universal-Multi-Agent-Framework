---
name: version-diffs
description: "Complete changelog: v1.0→v1.1 (12 bug fixes), v1.2 (backend-aware), v1.3 (Python 3.11), v1.4 (OOP+pipeline), v1.4.1 (8 bug fixes), v1.5 (Topology+Skill), v1.6 (Feature+modular)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9c942f95-5fb5-4276-8e53-c16059ca5e31
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
