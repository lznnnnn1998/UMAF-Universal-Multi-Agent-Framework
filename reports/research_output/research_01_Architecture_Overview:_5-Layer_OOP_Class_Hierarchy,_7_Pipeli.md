# Architecture Overview: 5-Layer OOP Class Hierarchy, 7 Pipelines, and 32 AgentRoles

## Overview

The Universal Multi-Agent Framework (UMAF) is a LangChain + DeepSeek multi-agent system built on a strict 5-layer OOP class hierarchy. The architecture is designed around three architectural backbones — **BaseAgent** (autonomous agent loop with dual backends), **AgentRole** (abstract template method defining agent lifecycle), and **BasePipeline** (LangGraph-based pipeline orchestrator) — that together form a composable, backend-agnostic framework for coordinating LLM-powered agents through complex, multi-stage workflows.

The architectural motivation is to separate concerns vertically (data → infrastructure → agent logic → specialized behavior → orchestration) and horizontally (seven independent pipelines for distinct use cases). Each layer builds on the previous: data types define the shape of information; infrastructure provides LLM backends and tools; the agent core implements the autonomous reasoning loop; concrete roles specialize behavior through the Template Method pattern; and pipeline classes compose roles into graph-structured workflows using LangGraph's `StateGraph`. This layering enables independent evolution of each concern — tools can be reconfigured via `tools_config.json` without touching agent code, new pipelines can be added by implementing three ABC methods, and backends can be swapped transparently.

UMAF's seven pipelines span code generation (Coder, CoderPP), research synthesis (Research), skill analysis (Skill), topology optimization (Topology), feature implementation (Feature), and meta-cognition (SelfEvolution). Together they comprise 32 concrete `AgentRole` subclasses, each implementing `tools_for_backend()`, `build_task()`, and optionally `parse_result()` — a consistent interface that makes the framework extensible without modifying the core engine. The system supports two LLM backends: DeepSeek via `ChatOpenAI` (JSON tool-call parsing with circuit breakers) and Claude CLI via subprocess streaming (native tool calling with incremental checkpointing), with tool assignments driven by a single `tools_config.json` file serving as the canonical source of truth.

## Key Methods & Approaches

### 1. The 5-Layer OOP Class Hierarchy

UMAF's architecture is organized into five strictly ordered layers, each building capabilities on top of the previous:

**Layer 1 — Data Types**: The foundation consists of three complementary type systems:
- **`ToolSpec`** (`tools/registry.py:10`): A frozen dataclass with `name`, `description`, and `parameters` fields that defines the schema for every tool available to agents. Eight canonical `ToolSpec` instances exist: `READ_FILE`, `WRITE_FILE`, `WRITE_LINES`, `RUN_COMMAND`, `CALL_CLAUDE`, `WEB_SEARCH`, `WEB_FETCH`, `DOWNLOAD_FILE`.
- **`AgentResult`** (`agent.py:240`): A structured result dataclass holding `messages` (list), `iterations` (int), and `success` (bool) — the standard return type from any agent run.
- **TypedDict State Types**: Each pipeline defines its own `TypedDict` for LangGraph state management: `MultiAgentState` (Coder, 6 fields), `ResearchState` (Research, 9 fields), `CoderPPState` (CoderPP, 11 fields), `TopologyState` (Topology, 7 fields), `SkillState` (Skill, 8 fields), `FeatureState` (Feature, 13 fields), `SelfEvolutionState` (SelfEvolution, 13 fields). These enforce type safety across the graph while remaining plain dicts for LangGraph compatibility.
- **`CheckpointManager`** (`agent.py:25`): Handles versioned checkpoint persistence (`{safe_name}_v{version:02d}_checkpoint.json`), log merging, and cross-version context injection for retry scenarios.

**Layer 2 — Infrastructure**: Provides the execution substrate:
- **`LLMProvider`** (ABC, `llm.py:18`): Abstract interface with a single `invoke(messages) -> AIMessage` method. Two implementations exist:
  - `DeepSeekProvider` (`llm.py:28`): Wraps `ChatOpenAI` with model `deepseek-chat`, temperature 0.3, max_tokens 8192. Reads `DEEPSEEK_API_KEY` from `.env`.
  - `ClaudeCLILLM` (`llm.py:67`): Shells out to `claude -p` subprocess via `subprocess.Popen` with `--output-format stream-json`, 600s timeout, and `bypassPermissions` mode. Supports both `invoke()` (text output) and `stream_invoke()` (streaming JSON events). Injects environment from `claude_env_sample.json`.
- **`ToolRegistry`** (`tools/registry.py:17`): Centralized registry of 8 `ToolSpec` instances + 28 classmethods returning role-specific tool lists (all defaulting to `[]`, overridden by `tools_config.json`). `TOOL_MAP` maps tool names to callable implementations from `tools/functions.py`. The `set_tool_config()` method applies JSON-based overrides using case-insensitive role matching against pipeline-aware sections.
- **Tool Implementations** (`tools/functions.py`): Eight functions — `read_file`, `write_file`, `write_lines`, `run_command` (30s timeout), `call_claude` (600s configured timeout), `web_search` (DuckDuckGo Lite), `web_fetch` (urllib, 30s timeout), `download_file` (urllib, 30s timeout). All accept `working_dir` as a keyword argument for path resolution.

**Layer 3 — Agent Core**: The autonomous execution engine:
- **`BaseAgent`** (`agent.py:247`): A ~830-line class implementing the full autonomous agent loop. Key mechanisms:
  - **Dual-backend dispatch**: `run()` delegates to `_run_deepseek()` or `_run_claude_cli()`.
  - **DeepSeek loop** (`_run_deepseek`): Iterates `max_steps` times. Each step: invoke LLM → parse JSON tool call (4 strategies: markdown fences, standard order, reversed order, JSON repair) → execute tool → check `TASK_COMPLETE` → apply interventions.
  - **Claude CLI loop** (`_run_claude_cli`): Launches `claude -p` as a subprocess, parses `stream-json` events (`assistant`, `user`, `result`), records messages incrementally, and checkpoints after each event. Hard timeout via `threading.Timer`. Supports one automatic retry on error/timeout.
  - **Circuit breakers**: Force wrap-up at `max_steps - 3` threshold, error spiral detection at 3+ consecutive persistent errors (timeout, not found, permission denied), unknown tool warnings, write reminders at `max_steps - 4`.
  - **Tool name translation** (`_CLAUDE_NATIVE_TOOL_SPECS`): Maps Python tool names to Claude CLI native names (e.g., `write_file` → `Write`, `run_command` → `Bash`).
  - **JSON repair** (`_repair_json`): Fixes trailing commas, raw newlines/tabs in strings via state-machine-based `_escape_raw_whitespace_in_strings()`.

- **`AgentRole`** (ABC, `agent.py:1088`): Template Method pattern defining the agent lifecycle:
  - `tools_for_backend(backend) -> list[dict]`: Abstract — returns tool specs for the role.
  - `build_task(backend, **context) -> str`: Abstract — builds the full task prompt.
  - `parse_result(result, working_dir, **context) -> Any`: Optional — parses raw output into structured data.
  - `execute(working_dir, backend, resume_from, version, **context) -> Any`: Concrete — orchestrates: instantiate `BaseAgent` → `agent.run()` → wrap in `AgentResult` → `self.parse_result()`. This is the single entry point for pipeline nodes. **Critical invariant**: `execute()` already calls `parse_result()` internally, so pipeline nodes must use the return value directly — never call `parse_result()` again (the "double-parse anti-pattern").

- **`BaseDecomposerRole`** (`agent.py:1155`): Abstract intermediate class shared by `ResearchDecomposerRole` and `CoderPPDecomposerRole`. Implements the Template Method pattern for decomposition: `_role_prompt()`, `_sizing_guide()`, `_sub_unit_requirements()`, `_json_template()`, `_fallback_decompose()`, `_backend_instructions()`. The concrete `build_task()` assembles prompts from template methods; `parse_result()` implements a three-tier extraction strategy: inline JSON array from agent messages → disk files (decomposition.json) → programmatic fallback decomposition.

**Layer 4 — Concrete Roles**: 32 `AgentRole` subclasses, each implementing the three abstract methods with specific behavior (detailed in Section 3 below).

**Layer 5 — Pipeline Classes**: Seven `BasePipeline` subclasses, each implementing `_build_graph()`, `_build_initial_state()`, `_decompose()`, and `_print_results()` to compose roles into LangGraph workflows with status-based routing.

### 2. The Seven Pipelines: Graph Topologies and Flow Routing

All seven pipelines use LangGraph's `StateGraph` with conditional edges driven by a `status` field in the state dict. `BasePipeline._status_router()` (`base.py:504`) builds a router function from a flow map (`{status: next_node}`) and a set of terminal error statuses.

#### 2.1 CoderPipeline (v1.0)

**Topology**: Cyclic 2-node graph with max 5 iterations.
```
coder → reviewer ↔ coder (loop) → END
         ↓ (REVIEW_PASSED)
        END
```

**Nodes**: 2 (coder, reviewer). **Cycles**: coder↔reviewer via conditional edges.

**Flow routing** (`coder.py:155-160`): The `_router` function checks `review_passed` and `iteration >= 5`. If passed → END. If not passed and under 5 iterations → route to the other agent (reviewer→coder or coder→reviewer). **Entry point**: coder.

**Key state fields**: `review_passed: bool`, `iteration: int`, `coder_files: list[str]`. The reviewer receives `coder_files` via `execute()` kwargs (v1.6.1 fix: reviewer was previously blind to coder output).

**Tool assignment**: Coder gets 6 tools (read_file, write_file, run_command, call_claude, web_search, web_fetch); Reviewer gets 5 (no write_file).

#### 2.2 ResearchPipeline (v1.2-v1.4)

**Topology**: 4-node linear graph with a self-loop on the workers node for retry.
```
head → workers (self-loop for retry) → reviewer → writer → END
```

**Nodes**: 4 (head, workers, reviewer, writer). **Self-loop**: workers → workers on `worker_retry` status.

**Flow routing** (`research.py:434`):
```
decomposed → workers, worker_retry → workers, researched → reviewer,
researched_partial → reviewer, reviewed → writer, written → END
```

**Key mechanisms**:
- **Head agent** decomposes research topic into 2-8 sub-topics based on complexity. Timeout: 300s.
- **Workers** execute with dependency ordering via `_topological_levels()`. Stop-on-failure blocks downstream. Version-bump retry (max 4 versions, 5 worker retries) with context reuse via `CheckpointManager.load_previous()`. Worker timeout: 900s.
- **Reviewer** scores each work on 5 dimensions (depth, accuracy, relevance, clarity, originality, each 1-10) + justification. Writes `scoring_report.json`.
- **Writer** generates LaTeX report with `_latex_escape()` handling 10 special characters. Falls back to `_fallback_latex()` template.

**Resilience**: `researched_partial` status (partial worker success) is still routed to reviewer — the router always moves forward. Duplicate detection via MD5 fingerprinting of worker summaries.

#### 2.3 CoderPPPipeline (v1.4-v1.6.1)

**Topology**: 5-node graph with worker self-loop and observer insertion point.
```
head → workers (self-loop for retry) → observer → reviewer ↔ workers (retry loop) → organizer → END
```

**Nodes**: 5 (head, workers, observer, reviewer, organizer). **Self-loop**: workers → workers on retry; reviewer → workers on `reviewed_retry`.

**Flow routing** (`coderpp.py:648-658`):
```
decomposed → workers, worker_all_success → observer, worker_retry → workers,
worker_skip_observer → reviewer, observed → reviewer,
reviewed_all_passed → organizer, reviewed_max_versions → organizer,
reviewed_retry → workers, assembled → END
```

**Key mechanisms**:
- **Head agent** decomposes code generation into modules with dependency declarations. Reads `.tex` and `.md` spec files. Timeout: 500s.
- **Observer** (`coderpp/organizer.py:observe_workers`): Head agent re-invoked to spy on worker progress, producing observations that inform the reviewer.
- **Workers** execute with topological dependency ordering. The `_workers_node` directly manages `completed` dict + dual-key registration (by `sub_task_id` and `module_name`) for dependency injection — a fix from v1.6.1 that corrected the bypass of `_run_workers_with_deps()`. File validation checks for empty/skeletal files (< 100 bytes).
- **Reviewer** distinguishes between worker failures (need worker retry) and reviewer failures (worker code is fine, just re-check). Max 5 versions, 5 worker retries. Timeout: 1200s.
- **Organizer** assembles only passed modules into the final project directory.

**Post-reviewer retry logic**: After reviewer runs, failed modules are classified as "worker needs retry" (no `.py` files beyond `__init__.py` and `test_` files) or "reviewer needs re-check" (code exists but reviewer flagged issues). This prevents unnecessary code regeneration.

#### 2.4 TopologyPipeline (v1.5)

**Topology**: 4-node strict linear graph with no loops.
```
analyzer → designer → evaluator → writer → END
```

**Nodes**: 4 (analyzer, designer, evaluator, writer). **No cycles**.

**Flow routing** (`topology.py:141-146`):
```
initialized → analyzer, analyzed → designer, designed → evaluator, evaluated → writer, written → END
```
Terminal errors: `error_analysis_failed`, `error_design_failed`, `error_evaluation_failed`, `error_writer_failed`.

**Key mechanisms**:
- **Analyzer** assesses task complexity across 6 factors: data_dependencies, parallelism_opportunities, tool_requirements, error_domains, latency_sensitivity, scale. Produces an `overall_complexity` rating.
- **Designer** proposes 2-4 candidate topologies using 4 design patterns: sequential, fan_out_fan_in, debate_consensus, hierarchical. Each topology specifies agent roles and their connections.
- **Evaluator** scores topologies on 5 dimensions (latency, reliability, cost_efficiency, simplicity, scalability, each 1-10). Sorts by `total_score` descending.
- **Writer** produces `topology_spec.json` (recommended topology with agents list) and `topology_report.md`.

#### 2.5 SkillPipeline (v1.5, v2 detectors)

**Topology**: 4-node fan-out/fan-in graph with parallel detector execution.
```
scanner → [4 parallel detectors] → aggregator → writer → END
```

**Nodes**: 4 graph nodes (scanner, detectors, aggregator, writer). The "detectors" node internally runs 4 agent roles in parallel via `_run_parallel_agents()`. **No cycles**.

**Flow routing** (`skill.py:260-264`):
```
initialized → scanner, scanned → detectors, detected → aggregator,
detected_partial → aggregator, aggregated → writer, written → END
```

**Key mechanisms**:
- **Scanner** classifies artifact type (language, framework, complexity), deep-reads content, and produces `artifact_analysis.json` with metadata and surface scan.
- **4 Parallel Detectors** (fan-out at detector node): Each detector is artifact-agnostic — they read the same `artifact_analysis.json` but extract different skill dimensions:
  - `DomainExpertiseDetectorRole`: Business/domain knowledge
  - `TechnicalCraftDetectorRole`: Programming patterns, code quality
  - `MethodologyDetectorRole`: Process, tooling, dev practices
  - `RigorDetectorRole`: Testing, documentation, error handling depth
  Parallelism is limited to 1 for `claude_cli` backend (prevents OOM from 4 simultaneous heavy subprocesses).
- **Aggregator** deduplicates skills across detectors, categorizes by universal dimensions, and produces `skill_inventory.json`.
- **Writer** generates `skills.json` (project, skills_by_category, all_skills) and `skills_report.md`.

**Dependency injection fix (v1.6.1)**: Upstream data (`project_scan`, `detector_outputs`, `skill_inventory`) was previously never reaching downstream agents. Fixed by passing these as `execute()` kwargs with inline summaries embedded in prompts.

#### 2.6 FeaturePipeline (v1.6)

**Topology**: 5-node graph with coder↔reviewer cycle (max 5 iterations).
```
scanner → planner → coder ↔ reviewer (max 5 cycles) → writer → END
```

**Nodes**: 5 (scanner, planner, coder, reviewer, writer). **Cycle**: coder↔reviewer.

**Flow routing** (`feature.py:180-221`): Linear flow for scanner/planner via `_status_router`; custom routers for coder (`_coder_router`: coded → reviewer) and reviewer (`_reviewer_router`: passed → writer, not passed and <5 iterations → coder, else → writer).

**Key mechanisms**:
- **Scanner** surveys the project directory, identifies language, conventions, and builds a `project_context.json` with file manifest and code patterns.
- **Planner** generates `implementation_plan.json` with both `files_to_create` and `files_to_modify` — supporting modifications to existing codebases, not just greenfield generation.
- **Coder** reads existing files, creates new ones, modifies existing ones, and writes/runs tests. Receives `review_issues` from the reviewer for iterative fixes.
- **Reviewer** validates implementation via the same REVIEW_PASSED/REVIEW_FAILED token scanning pattern used by CoderPipeline.
- **Writer** generates `feature_report.md`.

#### 2.7 SelfEvolutionPipeline (v1.8)

**Topology**: 5-node graph with coder↔reviewer cycle (max 3 iterations).
```
analyzer → planner → coder ↔ reviewer (max 3 iterations) → writer → END
```

**Nodes**: 5 (analyzer, planner, coder, reviewer, writer). **Cycle**: coder↔reviewer with `plan_revision` status routing back to coder.

**Flow routing** (`self_evolution.py:197-205`):
```
analyzed → planner, planned → coder, implemented → reviewer,
verified → writer, plan_revision → coder, completed → END
```

**Key mechanisms**:
- **Analyzer** scans UMAF's own codebase and agent logs (`agent_log/`), identifies improvement opportunities, and produces `analysis_report.json`.
- **Planner** generates `implementation_plan.json` from analysis findings with specific file targets and improvement descriptions.
- **Coder** implements improvements, detecting changed files via `git diff` or mtime comparison. Operates on the actual source tree (not a copy).
- **Reviewer** verifies changes by running the test suite (`pytest`) and scanning for REVIEW_PASSED/REVIEW_FAILED tokens. Records `test_results`.
- **Writer** documents the evolution in `evolution_report.md`.

**Safety**: Operates in the current git branch; all changes revertible with `git checkout -- .`. `MAX_ITERATIONS=3` (lower than CoderPipeline's 5, balancing thoroughness with the risk of destructive self-modification).

### 3. Complete Taxonomy of 32 AgentRoles

All 32 roles inherit from `AgentRole` (ABC) and override `tools_for_backend()`, `build_task()`. Some override `parse_result()` for structured output extraction. Two head agents additionally inherit from `BaseDecomposerRole`, which itself inherits from `AgentRole`.

#### 3.1 CoderPipeline (2 roles)

| # | Role Class | File | Responsibility | Key Methods |
|---|-----------|------|----------------|-------------|
| 1 | `CoderRole` | `pipeline/coder.py:27` | Generates code to fulfill requirements. 15-step budget. | `build_task()` enforces design guidelines: testable functions with parameters, CLI argument support, test coverage. |
| 2 | `ReviewerRole` | `pipeline/coder.py:49` | Reviews generated code for bugs and correctness. 10-step budget. | `build_task()` receives `coder_files` (v1.6.1 fix) for targeted review. Outputs REVIEW_PASSED/REVIEW_FAILED. |

#### 3.2 ResearchPipeline (4 roles)

| # | Role Class | File | Responsibility | Key Methods |
|---|-----------|------|----------------|-------------|
| 3 | `ResearchDecomposerRole` | `research/head_agent.py` | Decomposes research topics into 2-8 sub-topics. Inherits from `BaseDecomposerRole`. | `_sizing_guide()` scales count by complexity. `_fallback_decompose()` uses keyword-based heuristic splitting. Backend-aware: claude_cli version uses Read tool to fetch .tex files. |
| 4 | `ResearchWorkerRole` | `research/worker_agent.py` | Researches individual sub-topics, producing `.md` report files. 600s timeout. | `parse_result()` verifies `os.path.isfile()` before reporting success (v1.4 honest parse fix). Receives pre-fetched arxiv.org content for claude_cli backend. |
| 5 | `ResearchReviewerRole` | `research/reviewer_agent.py` | Scores research outputs on 5 dimensions (depth, accuracy, relevance, clarity, originality, each 1-10). | `parse_result()` extracts scored works with justifications. Writes `scoring_report.json`. |
| 6 | `WriterRole` | `research/writer.py` | Generates LaTeX research proposal. | `_latex_escape()` handles all 10 LaTeX special characters. Falls back to `_fallback_latex()` template with proper section structure. |

#### 3.3 CoderPPPipeline (5 roles)

| # | Role Class | File | Responsibility | Key Methods |
|---|-----------|------|----------------|-------------|
| 7 | `CoderPPDecomposerRole` | `coderpp/head_agent.py` | Decomposes code generation into modules with dependency declarations. Inherits from `BaseDecomposerRole`. | `_json_template()` specifies `id`, `module_name`, `description`, `files_to_create`, `dependencies`. `_disk_fallback_paths()` includes `modules/` directory scan. |
| 8 | `CoderPPWorkerRole` | `coderpp/worker_agent.py` | Implements individual code modules. Receives environment spec and dependency outputs. | `build_task()` includes `_dependency_outputs` summary so workers know upstream API contracts. |
| 9 | `CoderPPReviewerRole` | `coderpp/reviewer_agent.py` | Reviews generated modules for correctness and integration compatibility. | Distinguishes between code-missing (worker retry needed) vs logic-bug (just re-check). |
| 10 | `OrganizerRole` | `coderpp/organizer.py` | Assembles passed modules into final project structure. | Creates `project/` directory with `__init__.py` files, wires inter-module imports, writes setup files. |
| 11 | `ObserverRole` | `coderpp/organizer.py` | Observes worker progress mid-pipeline. Re-uses head agent with observation-specific prompt. | `observe_workers()` function (standalone, not a class) inspects module files and reports status to inform reviewer. |

#### 3.4 TopologyPipeline (4 roles)

| # | Role Class | File | Responsibility | Key Methods |
|---|-----------|------|----------------|-------------|
| 12 | `TopologyAnalyzerRole` | `topology/analyzer.py` | Assesses task complexity across 6 factors. | `parse_result()` extracts `complexity_factors` dict with `overall_complexity` rating and per-factor scores. |
| 13 | `TopologyDesignerRole` | `topology/designer.py` | Proposes 2-4 candidate topologies using 4 design patterns. | `build_task()` specifies 4 patterns (sequential, fan_out_fan_in, debate_consensus, hierarchical). Each topology includes agent list with role types and connections. |
| 14 | `TopologyEvaluatorRole` | `topology/evaluator.py` | Scores topologies on 5 dimensions and ranks by total_score. | `parse_result()` returns sorted list with `total_score`, per-dimension scores, and justifications. |
| 15 | `TopologyWriterRole` | `topology/writer.py` | Produces `topology_spec.json` and `topology_report.md`. | `parse_result()` returns `{spec, spec_path, report_path}`. Selects highest-scoring topology as recommendation. |

#### 3.5 SkillPipeline (7 roles)

| # | Role Class | File | Responsibility | Key Methods |
|---|-----------|------|----------------|-------------|
| 16 | `SkillScannerRole` | `skill/scanner.py` | Classifies artifact type, deep-reads content, produces `artifact_analysis.json`. | `_fallback_deep_scanner()` provides deterministic fallback using file extension heuristics. |
| 17 | `DomainExpertiseDetectorRole` | `skill/detectors.py` | Detects domain expertise from artifact evidence (e.g., finance, ML, networking). | `_fallback_detect()` uses keyword matching when LLM fails. |
| 18 | `TechnicalCraftDetectorRole` | `skill/detectors.py` | Detects technical skills: patterns, code quality, architecture decisions. | Analyzes coding patterns, abstraction usage, error handling. |
| 19 | `MethodologyDetectorRole` | `skill/detectors.py` | Detects methodology: process, tooling, CI/CD, testing practices. | Identifies testing frameworks, build tools, deployment patterns. |
| 20 | `RigorDetectorRole` | `skill/detectors.py` | Detects rigor: documentation quality, test coverage depth, edge case handling. | Evaluates comment density, test thoroughness, validation logic. |
| 21 | `SkillAggregatorRole` | `skill/aggregator.py` | Deduplicates skills across 4 detectors, categorizes by universal dimensions. | `_fallback_aggregator()` provides rule-based merging when LLM aggregation fails. |
| 22 | `SkillReportWriterRole` | `skill/writer.py` | Produces `skills.json` and `skills_report.md`. | `_fallback_skills_json()` and `_fallback_report_md()` provide template-based generation. |

#### 3.6 FeaturePipeline (5 roles)

| # | Role Class | File | Responsibility | Key Methods |
|---|-----------|------|----------------|-------------|
| 23 | `FeatureScannerRole` | `feature/scanner.py` | Scans project directory, identifies language, conventions, file manifest. | `_fallback_scanner()` uses `os.walk()` with extension-based heuristics when LLM scan fails. |
| 24 | `FeaturePlannerRole` | `feature/planner.py` | Creates `implementation_plan.json` with `files_to_create` AND `files_to_modify`. | Supports both greenfield and brownfield development. Plan includes per-file implementation notes. |
| 25 | `FeatureCoderRole` | `feature/coder.py` | Reads existing files, creates new files, modifies existing files, writes and runs tests. | Receives `review_issues` for iterative fixes; detects changed files for reviewer consumption. |
| 26 | `FeatureReviewerRole` | `feature/reviewer.py` | Validates implementation via REVIEW_PASSED/REVIEW_FAILED token scanning. | Same pattern as CoderPipeline reviewer — scans AIMessage content for verdict tokens. |
| 27 | `FeatureReportWriterRole` | `feature/writer.py` | Produces `feature_report.md` with implementation summary, changed files list, and review status. | Generates structured markdown report from pipeline state. |

#### 3.7 SelfEvolutionPipeline (5 roles)

| # | Role Class | File | Responsibility | Key Methods |
|---|-----------|------|----------------|-------------|
| 28 | `SelfEvolutionAnalyzerRole` | `self_evolution/analyzer.py` | Scans UMAF codebase and agent logs, identifies improvement opportunities. | `parse_result()` extracts `improvement_opportunities` list with per-opportunity priority, affected files, and rationale. |
| 29 | `SelfEvolutionPlannerRole` | `self_evolution/planner.py` | Creates `implementation_plan.json` from analysis findings. | `parse_result()` extracts `improvements` list with file targets, change descriptions, and expected impact. |
| 30 | `SelfEvolutionCoderRole` | `self_evolution/coder.py` | Implements improvements to UMAF source code. Detects changed files via git diff or mtime. | Operates on the actual source tree; receives `review_issues` for iterative fixes. Detects changed files for reviewer verification. |
| 31 | `SelfEvolutionReviewerRole` | `self_evolution/reviewer.py` | Verifies changes by running tests (`pytest`), scans for REVIEW_PASSED/REVIEW_FAILED. | `parse_result()` extracts `review_passed`, `review_issues`, and `test_results` from agent output. |
| 32 | `SelfEvolutionWriterRole` | `self_evolution/writer.py` | Documents the evolution in `evolution_report.md`. | Generates structured report listing changed files, review results, test outcomes, and recommendations. |

### 4. The Architectural Backbone: BaseAgent, AgentRole, and BasePipeline

The three classes form a layered backbone where each addresses a distinct concern:

**BaseAgent** (`agent.py:247`) handles the *execution mechanism* — how an individual agent interacts with an LLM to complete a task. It encapsulates the tool-call loop, circuit breakers, checkpoint persistence, and backend dispatch. Key design choices:
- **Two backends, one interface**: `run(task) -> dict` works identically for DeepSeek and Claude CLI.
- **Intervention system**: Four `_maybe_*` methods inject prompts at specific thresholds — nudge to write (`max_steps - 4`), force wrap-up (`max_steps - 3`), error spiral detection (3 consecutive errors), unknown tool warnings.
- **Post-loop forced write**: If the agent exhausts steps without writing output, `_force_final_write()` makes up to 2 additional LLM calls to salvage the task.
- **Checkpoint granularity**: DeepSeek checkpoints after every tool execution; Claude CLI checkpoints after every stream-json event — enabling sub-step recovery.

**AgentRole** (`agent.py:1088`) handles the *behavior specialization* — what a particular role knows and how it interprets results. It uses the Template Method pattern:
- `tools_for_backend()`: Declares tool dependencies (resolved through `ToolRegistry` and overridden by `tools_config.json`).
- `build_task()`: Constructs the prompt with role-specific instructions, design guidelines, and backend-aware formatting.
- `parse_result()`: Transforms raw `AgentResult` into structured data (lists, dicts, scores) suitable for pipeline consumption.
- `execute()`: Orchestrates the full lifecycle — this is the single integration point for pipeline nodes.

**BasePipeline** (`pipeline/base.py:14`) handles the *orchestration* — how multiple agents are composed into workflows. It provides:
- **Output directory management**: `manage_output_dir()` handles `--clean` and `--resume` flag semantics.
- **Double-check mechanism**: `confirm_decomposition()` shows the user proposed sub-tasks and allows interactive editing before execution.
- **Dependency-aware execution**: `_topological_levels()` groups tasks by dependency order with cycle detection and automatic edge removal.
- **Parallel execution**: `_run_parallel_agents()` uses `ThreadPoolExecutor` for concurrent agent execution with timeout handling and retry support.
- **Status-based routing**: `_status_router()` builds LangGraph conditional edge functions from declarative flow maps.
- **Dependency injection**: `_run_workers_with_deps()` injects upstream outputs into downstream tasks via `_dependency_outputs`, enabling data flow between topological levels.

**Dependency flow**: `main.py` → `Pipeline.run()` → `BasePipeline` lifecycle (manage dir → confirm decomposition → build graph → invoke) → Pipeline-specific `_build_graph()` composes `AgentRole.execute()` calls within LangGraph node functions → `AgentRole.execute()` instantiates `BaseAgent` → `BaseAgent.run()` drives the LLM interaction loop.

### 5. Tool Assignment Architecture

The `tools_config.json` file (`tools_config.json`) serves as the single source of truth for per-role tool assignments (v1.7 design). Key properties:
- **Structure**: Top-level keys are pipeline names (`research`, `coder`, `coderpp`, `topology`, `skill`, `feature`, `self_evolution`). Each pipeline maps role names to lists of tool names.
- **Fallback**: `__global__` key provides fallback assignments for roles not explicitly listed in their pipeline section.
- **Timeout overrides**: `__timeouts__` section overrides per-tool default timeouts (e.g., `call_claude: 600`).
- **Case-insensitive matching**: `ToolRegistry._apply_override()` matches role names case-insensitively against method suffixes (e.g., `"worker"` matches both `research_worker_tools()` and `coderpp_worker_tools()`).
- **Metadata stripping**: Keys starting with `_` (except `__global__`) are stripped during loading, allowing inline documentation without affecting runtime behavior.
- **All defaults are empty**: All 28 `ToolRegistry.*_tools()` classmethods return `[]` by default — `set_tool_config()` must be called first, ensuring no accidental tool leakage.

### 6. Resilience Patterns

UMAF employs multiple layers of resilience:

1. **Circuit breakers in BaseAgent**: Force wrap-up, error spiral detection, write reminders prevent infinite loops and wasted API calls.
2. **Checkpoint-based retry**: Workers that fail can resume from checkpoints with full message history, preserving prior reasoning while getting a fresh step budget. `CheckpointManager.load_previous(n)` loads the highest version below `n`.
3. **Stop-on-failure in dependency graphs**: If a level fails to produce outputs, downstream levels that depend on those outputs are deferred — preventing cascading failures.
4. **Fallback at every level**: Decomposition falls back to keyword-based splitting; parsing falls back to disk files then programmatic defaults; LaTeX falls back to template; skill detection falls back to deterministic heuristics.
5. **Dual-backend transparency**: The same pipeline can run on DeepSeek or Claude CLI without code changes — backend differences are handled by `AgentRole.build_task(backend)` and `BaseAgent.run()` dispatch.
6. **Router always moves forward**: `researched_partial` (incomplete results) is still accepted at the reviewer stage rather than aborting the pipeline.
7. **Merge on completion**: All agent checkpoints are merged into `*_merged.json` files at pipeline completion, providing a consolidated audit trail.

## Important Papers & References

While UMAF is a self-contained engineering artifact rather than a research publication, its design draws on several established patterns and frameworks:

- **LangGraph (LangChain, 2024)** — Provides the `StateGraph` abstraction and conditional edge routing used by all seven UMAF pipelines. UMAF extends this with status-based routing and TypedDict state management.
- **Template Method Pattern (Gamma et al., "Design Patterns", 1994)** — The `AgentRole` ABC and `BaseDecomposerRole` directly implement this pattern, with `execute()` as the template method and `tools_for_backend()`/`build_task()`/`parse_result()` as the primitive operations.
- **Chain-of-Thought Prompting (Wei et al., NeurIPS 2022)** — UMAF's `build_task()` method for every role constructs detailed, structured prompts that guide the LLM through multi-step reasoning, effectively implementing chain-of-thought at the prompt engineering level.
- **Self-Refine (Madaan et al., NeurIPS 2023)** — The coder↔reviewer loop in CoderPipeline, FeaturePipeline, and SelfEvolutionPipeline implements iterative self-refinement where an LLM critiques and improves its own output.
- **AutoGPT / BabyAGI (2023)** — The autonomous agent loop in `BaseAgent._run_deepseek()` with tool-call parsing and task completion detection follows the pattern established by early autonomous agent frameworks.
- **Map-Reduce for LLMs (LangChain documentation)** — The ResearchPipeline decompose→parallel workers→aggregate pattern implements a map-reduce architecture for research synthesis.
- **Mixture-of-Agents (Wang et al., 2024)** — The multi-agent topology with specialized roles (analyzer, designer, evaluator) in TopologyPipeline mirrors the mixture-of-agents approach where different LLM instances handle different aspects of a problem.
- **Anthropic Claude Code SDK (2025-2026)** — The `ClaudeCLILLM` backend integrates with Claude Code's subprocess interface, using `stream-json` output format for incremental event processing.

## Open Questions & Future Directions

1. **Dynamic topology optimization**: The TopologyPipeline currently *recommends* optimal topologies but does not *execute* them. A future version could close this loop by having the pipeline dynamically construct and execute the recommended topology, enabling UMAF to self-optimize its own agent graph for each task.

2. **Cross-pipeline composition**: Currently each pipeline operates independently. There is no mechanism to chain pipelines (e.g., Research → CoderPP to implement research proposals, or Skill → Feature to add features matching detected skill gaps). Pipeline composition would require a meta-orchestrator.

3. **Streaming and incremental output**: DeepSeek backend uses polling (invoke → check response → parse tool call), while Claude CLI backend uses stream-json events. Neither backend supports true streaming output to the user during agent execution. Implementing LangChain's `astream_events` or streaming tool result display would improve UX.

4. **Tool safety and sandboxing**: `run_command` and `call_claude` are powerful but potentially dangerous. Current safety relies on Claude CLI's permission model (bypassPermissions mode for subprocess agents). A dedicated sandbox layer (Docker container per agent, restricted filesystem access) would be appropriate for production deployments.

5. **Cost optimization**: The framework currently has no cost tracking or budget enforcement. With large research decompositions (8 workers × 600s each × multiple versions), costs can scale rapidly. Implementing token counting, cost estimation per pipeline run, and budget-based early termination would be valuable.

6. **Parallelism scaling limits**: The current implementation uses `ThreadPoolExecutor` with configurable `max_workers`. For 8+ parallel workers, Python's GIL becomes a bottleneck for CPU-bound tool operations (though I/O-bound LLM calls scale well). An `asyncio`-based or process-pool executor could improve throughput.

7. **Memory and context persistence across sessions**: `CheckpointManager` operates within a single pipeline run. There is no mechanism to persist learning across separate invocations (e.g., the SelfEvolution pipeline's findings feeding into future runs). A persistent memory store (vector DB of past agent logs) would enable cumulative improvement.

8. **Multi-model orchestration**: Currently DeepSeek powers all agents uniformly. The TopologyPipeline's design suggests different agents could benefit from different models (cheaper models for simple tasks, more capable models for complex reasoning). Implementing per-agent model selection would be a natural extension.

9. **Observability and tracing**: Agent logs are written as JSON files but there is no structured tracing (OpenTelemetry, LangSmith integration). For debugging complex multi-agent interactions, a trace viewer showing message flow, tool calls, and checkpoint transitions would significantly improve debuggability.

10. **Evaluation framework**: The reviewer agents provide a basic scoring mechanism, but there is no automated regression testing of pipeline output quality. A benchmark suite with known input-output pairs for each pipeline would enable continuous quality monitoring.

## Relevance to Main Topic

This architectural analysis is foundational for understanding UMAF as a whole. The 5-layer OOP hierarchy provides the structural vocabulary for discussing any component of the system. The seven pipelines represent the complete operational capability of the framework, and understanding their individual topologies and flow routing is essential for extending, debugging, or optimizing UMAF. The 32 AgentRole taxonomy serves as a reference for role-based access control, tool permission scoping, and capability analysis.

The architectural backbone — BaseAgent, AgentRole, and BasePipeline — demonstrates a clean separation of execution mechanism, behavior specialization, and workflow orchestration that is generalizable beyond UMAF. This pattern of layered abstraction with Template Method specialization could inform the design of other multi-agent frameworks seeking to balance flexibility (add new roles without modifying the engine) with safety (circuit breakers, checkpointing, fallback chains) and configurability (externalized tool assignments).
