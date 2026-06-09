# Universal Multi-Agent Framework (UMAF) v1.8

LangChain + DeepSeek multi-agent framework with seven pipelines and two backends. OOP architecture with 5-layer class hierarchy.

## Architecture

```
main.py → pipeline/       → agent.py → llm.py        (all pipelines)
               │                │          ├── ChatOpenAI (deepseek-chat)
               ▼                ▼          └── ClaudeCLILLM (subprocess)
        BasePipeline      AgentRole ABC
        ├── CoderPipeline       ├── CoderRole
        ├── ResearchPipeline    ├── ReviewerRole
        ├── CoderPPPipeline     ├── ResearchWorkerRole
        ├── TopologyPipeline    ├── ResearchDecomposerRole
        ├── SkillPipeline       ├── ResearchReviewerRole
        ├── FeaturePipeline     ├── WriterRole
        └── SelfEvolutionPipeline ├── SelfEvolutionAnalyzerRole
                                ├── SelfEvolutionPlannerRole
                                ├── SelfEvolutionCoderRole
                                ├── SelfEvolutionReviewerRole
                                ├── SelfEvolutionWriterRole
                                ├── FeatureScannerRole
                                ├── FeaturePlannerRole
                                ├── FeatureCoderRole
                                ├── FeatureReviewerRole
                                ├── FeatureReportWriterRole
                                ├── TopologyAnalyzerRole
                                ├── TopologyDesignerRole
                                ├── TopologyEvaluatorRole
                                ├── TopologyWriterRole
                                ├── SkillScannerRole
                                ├── DomainExpertiseDetectorRole
                                ├── TechnicalCraftDetectorRole
                                ├── MethodologyDetectorRole
                                ├── RigorDetectorRole
                                ├── SkillAggregatorRole
                                └── SkillReportWriterRole
                                (32 roles total)
```

## Modules

### `llm.py` — Two backends
- **DeepSeek** (default): `ChatOpenAI` with `deepseek-chat`, temp=0.3, max_tokens=4096. Reads `DEEPSEEK_API_KEY` from `.env`.
- **Claude CLI**: `ClaudeCLILLM` shells out to `claude -p` subprocess (300s timeout). Injects env from `claude_env_sample.json`. Accepts `cwd` and `allowed_tools`.

Factory: `get_llm(backend)`.

### `tools/` — Eight tools + ToolRegistry
`read_file`, `write_file`, `write_lines` (preferred for code — avoids multi-line string escaping), `run_command` (30s timeout), `call_claude` (120s, env-injected), `web_search` (DuckDuckGo Lite, no API key), `web_fetch` (urllib, 20s timeout), `download_file` (urllib, 30s timeout, saves to local file). Modular package: `registry.py` (ToolSpec + ToolRegistry with 23 role-specific methods, all defaults empty — tools come from config), `functions.py` (8 implementations + TOOL_MAP), `feature_tools.py` (5 feature pipeline role methods, auto-applied at import). `__init__.py` re-exports — no duplicated tool definitions.

### `tools_config.json` — Canonical tool assignments (v1.7)
Single source of truth for per-role tool assignments. Auto-loaded by `main.py` at startup. Maps pipeline → role → tool list. Also defines per-tool timeout overrides. `--tools-config` flag overrides with a custom file. All `ToolRegistry.*_tools()` methods return `[]` by default — `set_tool_config()` must be called first. Metadata keys (`__about__`, `_description`, etc.) are stripped on load. `__global__` key provides fallback for unlisted roles.

### `agent.py` — Agent core
- **`BaseAgent`**: Autonomous agent loop with circuit breakers (force wrap-up, error spiral detection, unknown tool warnings).
- **`AgentRole`** (ABC): Template method — `tools_for_backend()`, `build_task()`, `parse_result()`, `execute()`. Subclass for new agent types. **Important**: `execute()` internally calls `parse_result()` and returns parsed dict — do NOT call `parse_result()` again on the return value.
- **`CheckpointManager`**: Saves/loads agent state. `load_previous(version)` restores messages, resets iterations for context-reusing retries.
- **Conversation logger**: `_save_agent_log()` writes to `agent_log/<name>_<timestamp>.json`.
- **Pre-fetch layer**: For `claude_cli` workers, arxiv.org content is pre-downloaded at the framework level (via `download_file`) before the agent runs, avoiding Claude Code's cc-switch domain verification.

### `pipeline/` — Seven pipelines
**`BasePipeline`**: Output dir management, double-check confirmation, `_topological_levels()`, `_run_workers_with_deps()`, `_run_parallel_agents()`.

**`CoderPipeline`**: Coder (all 6 tools) → Reviewer (no write_file). Max 5 cycles. Coder resets `review_passed=False` each run.

**`ResearchPipeline`**:
```
head (decompose) → workers (dependency-ordered) → reviewer (score) → writer (LaTeX) → END
```
- **head_agent.py**: Backend-aware decomposition. Dynamically scales sub-topic count 2-8 based on topic complexity. Falls back to `_fallback_decompose()`.
- **worker_agent.py**: Backend-aware tasks + pre-fetch layer. Dependency-ordered via `_topological_levels()`. Stop-on-failure blocks downstream tasks. Version-bump retry with context reuse via `CheckpointManager.load_previous()`. `parse_result()` checks `os.path.isfile()` before reporting success. Worker timeout 600s.
- **reviewer_agent.py**: 5-dimension scoring (depth, accuracy, relevance, clarity, originality, each 1-10). Writes `scoring_report.json`.
- **writer.py**: LaTeX generation. Falls back to `_fallback_latex()` template. `_latex_escape()` handles all 10 special chars.
- **Flow dict**: `decomposed → workers`, `worker_retry → workers` (version+1, failed only), `researched → reviewer`, `researched_partial → reviewer`, `reviewed → writer`, `written → END`.
- **Constants**: `HEAD_TIMEOUT=120`, `WORKER_TIMEOUT=600`, `RESEARCH_MAX_VERSIONS=4`, `RESEARCH_MAX_WORKER_RETRIES=3`.

**`CoderPPPipeline`**: Multi-file code generation with organizer → workers → reviewer. `_decompose()` reads `.tex` and `.md` spec files.

**`TopologyPipeline`** (v1.5):
```
analyzer → designer → evaluator → writer → END
```
- **analyzer.py**: `TopologyAnalyzerRole` — assesses task across 6 complexity factors (data_dependencies, parallelism_opportunities, tool_requirements, error_domains, latency_sensitivity, scale)
- **designer.py**: `TopologyDesignerRole` — proposes 2-4 candidate topologies using 4 patterns (sequential, fan_out_fan_in, debate_consensus, hierarchical)
- **evaluator.py**: `TopologyEvaluatorRole` — scores topologies on 5 dimensions (latency, reliability, cost_efficiency, simplicity, scalability, each 1-10), sorts by total_score descending
- **writer.py**: `TopologyWriterRole` — writes `topology_spec.json` and `topology_report.md`

**`SkillPipeline`** (v1.5, v2 detectors):
```
scanner → 4 parallel detectors → aggregator → writer → END
```
- **scanner.py**: `SkillScannerRole` — classifies artifact type, deep-reads content, produces `artifact_analysis.json`
- **detectors.py**: 4 artifact-agnostic detectors — `DomainExpertiseDetectorRole`, `TechnicalCraftDetectorRole`, `MethodologyDetectorRole`, `RigorDetectorRole` — each reads `artifact_analysis.json`, infers skills from evidence
- **aggregator.py**: `SkillAggregatorRole` — reads domain report files, deduplicates, categorizes skills by universal dimensions
- **writer.py**: `SkillReportWriterRole` — produces `skills.json` (project, skills_by_category, all_skills) and `skills_report.md`

**`FeaturePipeline`** (v1.6):
```
scanner → planner → coder ↔ reviewer (max 5 cycles) → writer → END
```
- **scanner.py**: `FeatureScannerRole` — scans project directory, produces `project_context.json` with language, conventions, file manifest
- **planner.py**: `FeaturePlannerRole` — creates `implementation_plan.json` with both `files_to_create` AND `files_to_modify`
- **coder.py**: `FeatureCoderRole` — reads existing files, creates new files, modifies existing files, writes and runs tests
- **reviewer.py**: `FeatureReviewerRole` — validates implementation via REVIEW_PASSED/REVIEW_FAILED token scanning (same pattern as CoderPipeline)
- **writer.py**: `FeatureReportWriterRole` — produces `feature_report.md`

**`SelfEvolutionPipeline`** (v1.8):
```
analyzer → planner → coder ↔ reviewer (max 3 iterations) → writer → END
```
- **analyzer.py**: `SelfEvolutionAnalyzerRole` — scans UMAF codebase and agent logs, produces `analysis_report.json` with improvement opportunities
- **planner.py**: `SelfEvolutionPlannerRole` — creates `implementation_plan.json` from analysis findings
- **coder.py**: `SelfEvolutionCoderRole` — implements improvements, detects changed files via git diff or mtime
- **reviewer.py**: `SelfEvolutionReviewerRole` — verifies changes by running tests, scans for REVIEW_PASSED/REVIEW_FAILED tokens
- **writer.py**: `SelfEvolutionWriterRole` — produces `evolution_report.md`
- **Safety**: operates in current git branch; all changes revertible with `git checkout -- .`

### `main.py` — Entry point
```
python3 main.py [--mode coder|research|coderpp|topology|skill|feature|self_evolution] [--backend deepseek|claude_cli] [--working-dir PATH] [--tools-config PATH] [--target PATH] [--clean] [--yes] "requirement"
```
- `--tools-config` defaults to `tools_config.json` in repo root
- `--target` specifies directory/file to analyze (skill/feature/topology/self_evolution modes)

### `claude_config.py` — Env setup
Loads `claude_env_sample.json` (12 env vars). Falls back to `.example.json`. `merge_claude_env()` merges with `os.environ`.

### Directories
- `pipeline/`: base, coder, research, coderpp, topology, skill, feature, self_evolution (7 pipeline classes + BasePipeline)
- `tools/`: registry, functions, feature_tools (ToolSpec + 7 tool implementations + 23 role methods)
- `self_evolution/`: analyzer, planner, coder, reviewer, writer (5 agent roles)
- `feature/`: scanner, planner, coder, reviewer, writer (5 agent roles)
- `research/`: head_agent, worker_agent, reviewer_agent, writer (4 agent roles)
- `coderpp/`: head_agent, worker_agent, reviewer_agent, organizer (4 agent roles)
- `topology/`: analyzer, designer, evaluator, writer (4 agent roles)
- `skill/`: scanner, detectors, aggregator, writer (4 agent roles)
- `test/`: 10 test files (379 tests) — conftest, test_smoke, test_pipeline, test_coder, test_research, test_coderpp, test_topology, test_skill, test_feature, test_self_evolution
- `utils.py`: shared helpers (extract_json_object, extract_json_array, safe_read, _PROFICIENCY_SCORES)

## Setup

```bash
pip install -r requirements.txt
# Set DEEPSEEK_API_KEY in .env
# For claude_cli: cp claude_env_sample.example.json claude_env_sample.json (edit API key)
# Scope permissions in .claude/settings.local.json
```

## Key Design Decisions

- OOP class hierarchy: Data types → Infrastructure → Agent core → Concrete roles → Pipeline classes
- `AgentRole` ABC + `ToolRegistry` centralization (no duplicated tool definitions)
- Tool metadata + TOOL_MAP separation; explicit `working_dir` (no global state)
- Tool name translation for Claude CLI (Python names → native names via regex)
- Tool assignment driven by `tools_config.json` (v1.7): no hardcoded defaults in code; all 23 role methods return `[]` — `set_tool_config()` is the single source of truth
- Backend-aware task generation (v1.2): no nested `claude -p` for claude_cli workers
- **Python >= 3.11**: `X | None` syntax, no deprecated `Optional[X]` or `Union[X, Y]`
- Fallbacks at every stage; DuckDuckGo Lite (no API key); all agents logged for debugging
- Dependency-aware execution (v1.4): stop-on-failure blocks downstream workers, version-bump retries reuse context via checkpoints
- Router always moves forward: `researched_partial` accepted at reviewer stage
- **Double-parse anti-pattern**: `AgentRole.execute()` already calls `parse_result()` internally — pipeline nodes must use the return value directly, never call `parse_result()` again

## Known Limitations

- `claude -p` may write to slightly different filenames than requested (pipeline verifies existence)
- Worker timeouts: complex tasks may exceed 600s (increase `ClaudeCLILLM.timeout`)
- DeepSeek JSON tool-call format less reliable than native tool calling
- DuckDuckGo scraping is regex-based and fragile to layout changes
- Subprocess needs `.claude/` settings scoped to its working directory
- Claude Code's cc-switch blocks arxiv.org domain verification → workaround: `download_file` pre-fetches content at framework level before agents run
- CoderPP workers can get stuck on TaskOutput framework calls when modifying pipeline.py

## Version History

### v1.8 (June 2026) — Self-Evolution Pipeline + Test Enhancement
- **Self-Evolution Pipeline**: 7-node graph (analyzer → planner → coder ↔ reviewer → writer). UMAF analyzes its own codebase and agent logs, identifies improvement opportunities, implements changes, verifies with tests, and documents results. 5 AgentRoles in `self_evolution/`. MAX_ITERATIONS=3 for coder↔reviewer loop.
- **ToolRegistry**: 5 new classmethods (`self_evolution_analyzer_tools()` through `self_evolution_writer_tools()`) in `registry.py`. Removed redundant `tools/self_evolution_tools.py` (methods were duplicated).
- **tools_config.json**: Added `self_evolution` section with per-role tool assignments (analyzer, planner, coder, reviewer, writer).
- **main.py**: 7 modes — coder, research, coderpp, topology, skill, feature, self_evolution.
- **Test enhancement**: 175 new behavioral tests across test_coder.py (14→27), test_research.py (21→62), test_coderpp.py (26→58). Tests now verify graph node behavior, parse_result logic, flow routing, fallback methods, and resume state reconstruction — not just structure.
- **New tests**: test_self_evolution.py (49 tests), test_pipeline.py, test_coder.py, test_coderpp.py, test_research.py, test_feature.py, conftest.py.
- **Removed redundant files**: `_run_*.py` temporary test runners, `_test_hang.py`, `review_verdict.txt`.
- **Verified**: 379/379 tests pass.

### v1.7 (June 2026) — tools_config.json + Codebase Cleanup
- **tools_config.json**: Single source of truth for per-role tool assignments. Auto-loaded by `main.py`. `--tools-config` overrides. All `ToolRegistry.*_tools()` defaults changed from hardcoded lists to `[]` — `set_tool_config()` must be called first. Metadata keys stripped on load. `__global__` fallback support.
- **Code deduplication**: Removed ~200 lines — 5 `_extract_json_object` copies consolidated into `utils.py`, 2 `_extract_json_array` copies moved, 4 `sys.path.insert` hacks removed from `skill/`, `_PROFICIENCY_SCORES` centralized (was 5 inline copies). Added `extract_json_array()` to `utils.py`.
- **Dead code removal**: `run_agent()`, `BaseAgent._checkpoint_path()`, `_checkpoint_path()` from `agent.py`; `_load_config()`, `_claude_env`, `get_claude_env()` from `claude_config.py`; unused imports from `coderpp/head_agent.py`, `research/head_agent.py`, `pipeline/__init__.py`.
- **Backend-agnostic tool defaults**: Removed backend-differentiated tool lists in `research_decomposer_tools()`, removed "do NOT search the web" prompt restrictions. Tools are now purely config-driven.
- **Verified**: 99/99 tests pass.

### v1.6.1 (June 2026) — Dependency Injection Fixes Across 3 Pipelines
- **CoderPipeline**: Reviewer was blind to coder output — now receives `coder_files` (files produced by coder, scanned from working directory) via `execute()` and displayed in reviewer prompt as "Files Produced by Coder" section. Added `coder_files: list[str]` to `MultiAgentState`.
- **SkillPipeline**: Upstream data in LangGraph state never reached downstream agents — detectors, aggregator, and writer all relied on discovering files from disk. Fixed by passing `project_scan`, `detector_outputs`, and `skill_inventory` via `execute()` kwargs, with inline summaries embedded in prompts so agents know what was computed upstream.
- **CoderPPPipeline**: `_workers_node` had its own topological level loop but called `_run_parallel_agents()` directly, completely bypassing `_run_workers_with_deps()` and never injecting `_dependency_outputs`. Fixed by adding `completed` dict + dependency resolution + dual-key registration (by `sub_task_id` and `module_name`) directly in `_workers_node`. Verified: 3-worker test with transitive dependencies, all reviewers passed, 131/131 tests passing in assembled project.
- **Verified**: All 97 tests pass. CoderPP dependency injection confirmed in worker checkpoints.

### v1.6 (June 2026) — Feature Pipeline + Modular Package Structure
- **Feature Pipeline**: 5-node graph (scanner → planner → coder ↔ reviewer → writer) for adding/editing code in existing projects. 5 AgentRoles in `feature/`. Supports both `files_to_create` AND `files_to_modify`.
- **Modular packages**: `pipeline/` (7 modules, split from 2,334-line monolith), `tools/` (3 modules: registry + functions + feature_tools), `test/` (4 test files). `feature/` contains ONLY agent role definitions.
- **Shared utilities**: `utils.py` — `extract_json_object()` and `safe_read()` used across feature roles.
- **ToolRegistry**: 5 new feature role classmethods via `feature_tools.py`, auto-applied at import time.
- **main.py**: 6 modes — coder, research, coderpp, topology, skill, feature.
- **Verified**: All 6 pipelines pass end-to-end with claude_cli backend. 99 tests pass.

### v1.5 (June 2026) — Topology Optimizer + Skill Summarizer
- **Topology Optimizer**: 4-node linear graph (analyzer → designer → evaluator → writer). Determines optimal agent topology for any task description. 4 AgentRoles in `topology/`.
- **Skill Summarizer**: 4-node fan-out/fan-in graph (scanner → 4 parallel detectors → aggregator → writer). Domain-parallel detection (Python, JS, Infra, ConfigDocs). 7 AgentRoles in `skill/`.
- **Meta-programming**: Both pipelines generated by CoderPP from `.md` spec files. `_decompose()` extended to handle `.md` files.
- **ToolRegistry**: 4 new classmethods for topology agents.
- **main.py**: 5 modes — coder, research, coderpp, topology, skill.
- **Verified**: Topology Optimizer produced valid spec (20KB JSON, 16KB report); Skill Summarizer detected 33 skills across 11 categories. 42 smoke tests pass.

### v1.4.1 (June 2026) — 8 Bug Fixes
- Agent loop: tool execution before TASK_COMPLETE check; stronger force wrap-up; post-loop forced write; mid-loop write reminder
- CheckpointManager: fixed version bump context injection; error spiral threshold 3→2
- 15 smoke tests added for agent/pipeline core

### v1.4 (June 2026) — Pipeline Robustness & OOP Refactoring
- **OOP**: 5-layer class hierarchy, `AgentRole` ABC, `ToolRegistry` centralization, 3 dead `graph.py` files removed
- **Stop-on-failure**: `_run_workers_with_deps` breaks out on failure, blocks downstream dependents
- **Version-bump retry**: Failed workers retry with context reuse via `CheckpointManager.load_previous()`
- **Honest `parse_result`**: `ResearchWorkerRole.parse_result()` checks `os.path.isfile()`
- **Worker retry state machine**: max 3 retries, max 4 versions
- **Timeout**: Worker timeout 300s → 600s. **CoderPP pipeline** added.
- **Verified**: 7/7 workers (100%); scores 48-38/50; 60KB LaTeX; 443s pipeline time.

### v1.3.1 (May 2026) — Worker Output Fix & arxiv.org Access
- **Bug fix**: Reordered agent loop to execute tool calls BEFORE checking TASK_COMPLETE
- **`download_file` tool**: Framework-level urllib download → local file
- **Pre-fetch layer**: arxiv.org content pre-downloaded for claude_cli workers
- **Verified**: 4/4 workers produce files (up from 2/4); scores 47, 46, 43, 41/50.

### v1.3 (May 2026) — Code Quality & Modernization
- **Python >= 3.11**: `Optional[X]` → `X | None`; `.python-version` set to 3.11
- **Bug fix**: `_latex_escape()` backslash producing tab (raw string fix)
- **Dynamic decomposition**: sub-topic count 2-8 based on complexity
- **New tool**: `web_fetch` (urllib, bypasses Claude Code permissions)
- All 8 unit tests pass; end-to-end coder pipeline verified

### v1.2 (May 2026) — Backend-Aware Agents
- Backend-aware worker tasks (no nested `claude -p`), head agent Read-only for claude_cli
- Conversation logger, scoped permissions, security cleanup (`.example.json` template)
- Timeout 120s→300s, parallelism 4→2, `--allowedTools` always passed
- Verified: 4/6 workers (21-26KB each), top score 43/50, 41KB LaTeX
