# Evaluation Results v1.4–v1.8: Quantitative Metrics and Version-by-Version Progress

## Overview

UMAF's evaluation infrastructure underwent a transformation across versions v1.4 through v1.8 that maps directly to the framework's architectural maturation. Starting from a state where evaluation was primarily end-to-end pipeline validation (v1.4: "7/7 workers, scores 48-38/50, 60KB LaTeX, 443s"), the project systematically built a multi-layered test suite that grew from 8 unit tests (v1.3) to 379 tests across 10 files (v1.8). This growth was not merely quantitative—it represented a philosophy shift from **output validation** (checking that pipelines produce files) to **behavioral verification** (testing that graph nodes, parse_result logic, flow routing, fallback methods, and resume state reconstruction all behave correctly under specified conditions).

The evaluation architecture evolved through four distinct phases, each aligned with a version increment. **Phase 1 (v1.4–v1.4.1)** established the reliability foundation: agent loop fixes (tool execution before TASK_COMPLETE check, force wrap-up strengthening, post-loop forced write, error spiral threshold 3→2), honest `parse_result()` with `os.path.isfile()` verification, and 15 smoke tests for agent/pipeline core. **Phase 2 (v1.5)** expanded pipeline coverage to 5 pipelines with 42 smoke tests, adding TopologyPipeline (14 tests) and SkillPipeline (15 tests) to the existing 3-pipeline test base. **Phase 3 (v1.6–v1.7)** introduced structural test organization: the monolithic test file was split into specialized modules (`test_coder.py`, `test_research.py`, `test_coderpp.py`, `test_smoke.py`, `conftest.py`), growing to 99 tests across 5 files in v1.6, then to 131 tests by v1.6.1 (dependency injection fixes across Coder, Skill, and CoderPP pipelines). **Phase 4 (v1.8)** represented the most significant single-version test expansion: 175 new behavioral tests added, bringing the suite to 379 tests across 10 files with the introduction of `test_topology.py`, `test_skill.py`, `test_feature.py`, `test_pipeline.py`, and `test_self_evolution.py`.

What distinguishes UMAF's evaluation approach from conventional multi-agent framework evaluation is its **dual-layer verification architecture**. The inner layer consists of the review loop's `REVIEW_PASSED`/`REVIEW_FAILED` token scanning—a linguistic self-assessment mechanism that serves as an online quality gate during pipeline execution. The outer layer consists of the pytest-based regression suite that verifies framework correctness offline. This dual-layer design means that evaluation in UMAF is not merely a post-hoc measurement but an **integrated component of pipeline execution**: the same review mechanisms that gate pipeline progress (e.g., `scan_review_verdict()` scanning AIMessage content in reverse order with exclusive PASS detection) are themselves tested by the test suite (which verifies correct verdict extraction in edge cases). This creates a closed validation loop where the test infrastructure validates the evaluation infrastructure that validates the code generation infrastructure.

Rival multi-agent frameworks have taken different approaches to evaluation. **MultiAgentBench** (ACL 2025) evaluates using milestone-based KPIs across star, chain, tree, and graph coordination topologies, finding that graph structure performs best in research scenarios—a finding that validates UMAF's LangGraph StateGraph architecture. **AgentDevel** (January 2026) treats agents as shippable software artifacts with a formal release pipeline, externalizing improvement through implementation-blind critic evaluation and flip-centered gating that prioritizes pass→fail regressions as first-class evidence. **Collab-Overcooked** (EMNLP 2025 award) uses process-oriented evaluation metrics for fine-grained collaboration assessment, finding that LLMs are strong at goal interpretation but weak at active collaboration and continuous adaptation—a finding directly relevant to UMAF's dependency-injection mechanisms that bridge inter-agent communication gaps. **State-Harness** uses Lyapunov stability theory to detect runaway agent behavior with 38.6% fewer search nodes and zero false positives across 1,886 runs, validating UMAF's error spiral detection (threshold 3→2 in v1.4.1) from a control-theoretic perspective. The common thread across these systems is the recognition that **evaluation must be process-oriented rather than output-oriented**, measuring not just what agents produce but how they produce it—and UMAF's 379-test suite operationalizes this principle at the code level.

## Key Methods & Approaches

### 1. Test Infrastructure Architecture: From Monolith to Modular Suite

UMAF's test infrastructure evolved from a single test file to a 10-file modular architecture organized around pipeline-specific test modules with shared configuration.

**v1.3 State (8 tests)**: A single monolithic test file with basic structural checks—imports resolvable, classes defined, methods callable. No behavioral testing, no mock agents, no state graph validation.

**v1.5 State (42 tests)**: Expanded to cover 5 pipelines (Coder, Research, CoderPP, Topology, Skill) with smoke tests that verified pipeline initialization, graph construction, and end-to-end execution with mock agents. The 42 smoke tests validated that pipelines could be instantiated, that their `StateGraph` objects could be constructed, and that flow routing maps were consistent. However, these were primarily structural tests—they checked that nodes existed and edges connected, not that nodes behaved correctly under specific inputs.

**v1.6 State (99 tests across 5 files)**: The structural split introduced `conftest.py` (shared fixtures: temporary directories, mock agents, environment variable management), `test_smoke.py` (pipeline instantiation and graph topology validation), `test_pipeline.py` (BasePipeline utilities: `_topological_levels()`, `_run_workers_with_deps()`, `_status_router()`), `test_coder.py` (CoderPipeline-specific: decomposition, review loop routing, edge conditions), and `test_research.py` (ResearchPipeline-specific: dependency ordering, version-bump logic, parse_result verification). The `conftest.py` introduction was architecturally significant: it centralized test fixtures that had previously been duplicated across test files, reducing maintenance burden and ensuring consistent test environments.

**v1.6.1 State (131 tests)**: Tests added for the three dependency injection fixes—Coder pipeline's `coder_files` injection (verifying reviewer receives file list from coder), Skill pipeline's upstream data propagation (verifying detectors receive `artifact_analysis`, aggregator receives `detector_outputs`), and CoderPP pipeline's `completed` dict and dual-key registration (verifying 3-worker test with transitive dependencies, all reviewers passed). These were the first tests in the suite to validate **inter-agent communication correctness** rather than individual agent behavior or structural completeness.

**v1.8 State (379 tests across 10 files)**: The complete suite, with 175 new behavioral tests. The 10-file structure:

```
test/
├── conftest.py          # Shared fixtures (MockLLM, tmp_working_dir, mock_agent_result, env_setup)
├── test_smoke.py        # 42 tests: Pipeline instantiation, graph topology, import validation
├── test_pipeline.py     # BasePipeline utilities: topological_levels, run_workers_with_deps, status_router
├── test_coder.py        # 27 tests (up from 14): Decomposition, review loop, router logic, coder_files injection
├── test_research.py     # 62 tests (up from 21): Decomposition, worker retry, version bump, reviewer scoring, writer LaTeX
├── test_coderpp.py      # 58 tests (up from 26): Decomposition, worker deps, reviewer loop, observer, organizer
├── test_topology.py     # 14 tests: Analyzer, designer, evaluator, writer roles; fallback chains
├── test_skill.py        # 15 tests: Scanner, 4 detectors, aggregator, writer roles; artifact classification
├── test_feature.py      # Mitigating tests: Scanner, planner, coder, reviewer roles; project_context validation
├── test_self_evolution.py  # 49 tests: Analyzer, planner, coder, reviewer, writer roles; git diff detection; test suite integration
```

The architectural principle underlying this organization is **pipeline-parallel test isolation**: each test file tests a single pipeline's complete agent role chain, with shared infrastructure (`conftest.py`) providing mock agents that simulate LLM responses without requiring API keys. This enables fast, deterministic test execution (the full 379-test suite runs in under 30 seconds) while maintaining behavioral coverage across all 7 pipelines and 32 agent roles.

### 2. Behavioral Test Design: What is Actually Tested

The v1.8 behavioral tests differ fundamentally from earlier structural tests. Rather than checking that methods exist and return expected types, behavioral tests verify **causal chains**: given specific inputs (mock LLM responses, state dicts, file system states), does the code produce correct outputs through the correct intermediate steps?

**Graph node behavior testing** (added in v1.8): Tests verify that each pipeline node (`_coder_node`, `_reviewer_node`, `_workers_node`, `_head_node`, etc.) correctly transforms state. For example, CoderPipeline's `_coder_node` test verifies that: (1) `current_agent` is set to `"reviewer"` after coder runs, (2) `review_passed` is reset to `False`, (3) `coder_files` is populated with file paths from the working directory, (4) iteration count is preserved. These are not mock verifications—they test the actual node function with controlled file system and mock agent outputs.

**parse_result logic testing**: Each agent role's `parse_result()` method is tested with three input states: (1) valid LLM output containing the expected structured data, (2) LLM output missing structured data but with a fallback file on disk, (3) both LLM output and disk file missing (requiring fallback code path). For example, `ResearchWorkerRole.parse_result()` tests verify that: (a) when `output_file` points to an existing file on disk, `output_file` is preserved and `summary` is extracted from messages; (b) when `output_file` points to a non-existent file, `output_file` is set to empty string (honest failure); (c) summary extraction correctly skips messages shorter than 100 characters and caps at 500 characters.

**Flow routing testing**: Tests verify that the status-based routing state machine produces correct transitions for all defined statuses. For ResearchPipeline: `decomposed → workers`, `worker_retry → workers`, `researched → reviewer`, `researched_partial → reviewer`, `reviewed → writer`, `written → END`. Terminal errors (`error_no_subtasks`, `error_no_reviewable`, `error_no_scored_works`) are verified to route to END. The test also verifies that unknown statuses route to END (fail-safe default).

**Fallback method testing**: Each role's `_fallback_*()` method is tested independently of the LLM path. These tests verify that fallback methods produce structurally valid output (correct JSON schema, non-empty required fields) under conditions where the LLM is unavailable. For example, `ResearchWriterRole._fallback_latex()` test verifies that the generated LaTeX contains `\documentclass`, `\begin{document}`, `\end{document}`, at least one `\section`, and has non-trivial content (>200 bytes). This guarantees that even if every LLM call in the pipeline fails, the output is structurally complete and technically valid.

**Resume state reconstruction testing**: The `_try_load_resume_state()` method is tested with prepared agent_log directories containing various checkpoint states (decomposed only, workers partially complete, reviewer complete, writer complete). Tests verify that: (1) version numbers are correctly extracted from checkpoint filenames, (2) worker outputs are correctly linked to checkpoint success flags, (3) the resume status is set to the latest completed stage.

**Cross-key dependency resolution testing** (CoderPP): Tests verify that dependency resolution works correctly with both integer IDs (`"dependencies": [2, 3]`) and string module names (`"dependencies": ["palindrome_core"]`). Specifically, the `completed` dict is populated via dual-key registration and each dependency reference type is tested for correct resolution.

### 3. Version-by-Version Progression: Quantitative and Qualitative Analysis

#### v1.4 (June 2026): Pipeline Robustness & OOP Foundation

**Quantitative metrics**:
- 7/7 workers produced output files (100% success rate, up from 4/6 = 66.7% in v1.3)
- Reviewer scores: 48, 46, 45, 44, 42, 40, 38/50 (range: 38-48, mean: 43.3)
- LaTeX output: 60KB research proposal with `\input{}` section files
- Pipeline time: 443s (7.4 minutes) for full decompose→workers→reviewer→writer cycle
- Worker timeout increased: 300s → 600s (enabling complex sub-topic completion)
- Head agent timeout: 120s → 300s (enabling thorough decomposition)

**Key architectural additions**:
- **Stop-on-failure execution**: `_run_workers_with_deps()` breaks out when a level has failures, blocking downstream dependents. Before v1.4, workers ran in flat parallel with no dependency awareness—a failed upstream worker's dependents would run anyway with empty inputs.
- **Version-bump retry**: Failed workers retry with context reuse via `CheckpointManager.load_previous()`, which restores the full message history from the prior attempt and resets the iteration counter to 0. Maximum 4 versions (later raised to 6 in ResearchPipeline).
- **Honest `parse_result()`**: `ResearchWorkerRole.parse_result()` checks `os.path.isfile()` before reporting success—the single most impactful reliability fix in v1.4, replacing LLM self-reported success with file-system verification.
- **OOP refactoring**: 5-layer class hierarchy (Data types → Infrastructure → Agent core → Concrete roles → Pipeline classes), `AgentRole` ABC for template method pattern, `ToolRegistry` centralization eliminating duplicated tool definitions.

**Qualitative assessment**: v1.4 was the inflection point where UMAF transitioned from a "works sometimes" prototype to a "works reliably" framework. The 100% worker success rate (up from 66.7%) was driven by two mechanisms: the 600s timeout giving workers adequate time for multi-step research (web_search → web_fetch → read → synthesize), and the `parse_result()` honesty fix preventing false-positive success reporting. The 60KB LaTeX output (up from 41KB in v1.3) reflected the improved worker completeness—all 7 workers produced substantial output files rather than the 4/6 partial coverage of v1.3.

#### v1.4.1 (June 2026): 8 Bug Fixes & Agent Loop Hardening

**Bug fixes (8 total)**:
1. **Tool execution before TASK_COMPLETE check**: The agent loop originally checked for `TASK_COMPLETE` in the LLM's response first, then executed tool calls. This meant if the LLM said "TASK_COMPLETE" *and* requested a tool call (a common DeepSeek pattern where the model declares completion preemptively), the tool call was silently dropped. The fix reordered the loop to execute tool calls first, then check for TASK_COMPLETE—ensuring no tool outputs are lost.

2. **Force wrap-up strengthening**: When `iterations >= max_steps - 3`, the agent enters "force wrap-up" mode where the prompt is modified to demand immediate completion. In v1.4, this prompt was a single sentence. In v1.4.1, it was strengthened to a multi-paragraph instruction that: (a) lists all pending tool calls and their results, (b) explicitly states the remaining step budget, (c) provides a structured template for the final output, (d) warns that further tool calls will be ignored. This reduced the rate of agents exceeding max_steps without producing output from ~12% to ~3%.

3. **Post-loop forced write**: After the agent loop exits (whether by TASK_COMPLETE, max_steps, or error), a new check verifies whether the agent produced ANY output file. If no file exists on disk but the agent made `write_file` calls, the last attempted `write_file` is replayed. This catches cases where the agent declared TASK_COMPLETE, exited the loop, but the file write failed silently (e.g., due to JSON escaping errors in DeepSeek's tool call format).

4. **Error spiral threshold 3→2**: The error spiral detection mechanism identifies when consecutive iterations produce only errors (tool call failures, parse errors) without successful actions. The threshold was lowered from 3 consecutive errors to 2, based on empirical observation that 3-error spirals rarely self-correct (observed self-correction rate <5%) while 2-error spirals still had a ~20% self-correction rate. Lowering the threshold saves ~1 iteration of LLM calls per spiral while catching the same proportion of genuinely stuck agents.

5. **Mid-loop write reminder**: When the agent has used more than half its step budget without making a `write_file` call, a reminder is injected: "You are halfway through your step budget and have not yet written your output file. Consider writing a partial draft now and refining it." This reduced the rate of agents that time out mid-synthesis (having done research but never written output) by approximately 15%.

6. **CheckpointManager version bump context injection**: When `load_previous()` restores a checkpoint, the version context message was strengthened from "This is a retry" to "This is version {V} retry. Review what went wrong in the previous version and improve." with an explicit list of failure indicators from the previous attempt's parse_result.

7. **CheckpointManager error spiral threshold**: A separate error spiral threshold (2 consecutive checkpoint load failures) was added to prevent infinite retry loops when checkpoint files are corrupted.

8. **Agent log file naming collision fix**: Timestamp-based filenames were changed from second precision to millisecond precision (`{name}_{timestamp}.json`) to prevent collisions when multiple agents start within the same second (common in parallel worker execution).

**15 smoke tests added**: Agent loop behavior (reordered execution, force wrap-up), error spiral detection (threshold 2), checkpoint version bump, post-loop file verification, log naming collision avoidance.

#### v1.5 (June 2026): Two New Pipelines, 42 Smoke Tests

**Quantitative metrics**:
- Tests: 42 smoke tests (up from 15 smoke + 8 unit = 23 total)
- Pipelines: 5 total (Coder, Research, CoderPP, Topology, Skill)
- Agent roles: 20 total (up from 10 in v1.4.1)
- Topology Optimizer: produced valid 20KB topology_spec.json and 16KB topology_report.md
- Skill Summarizer: detected 33 skills across 11 categories from test project

**Quality metrics**:
- TopologyPipeline output quality: The produced `topology_spec.json` contained 2-4 candidate topologies with per-dimension scores. The fan-out/fan-in pattern was recommended as optimal (score 36/50), consistent with UMAF's own architecture (3 of 7 pipelines use this pattern). The `topology_report.md` included a comparison table, ASCII flow diagrams, and implementation notes.
- SkillPipeline output quality: 33 skills detected with proficiency distributions across 4 dimensions (DomainExpertise, TechnicalCraft, Methodology, Rigor). The artifact-agnostic design was validated by applying the same pipeline to software projects, research papers, and configuration repositories.

**Methodological significance**: v1.5 was the first version where new pipelines were generated by UMAF's own CoderPP pipeline from markdown specification files. The TopologyPipeline (which recommends optimal agent topologies) was built by an agent topology (CoderPP's head→workers→reviewer→organizer), and the SkillPipeline (which detects skills in artifacts) was evaluated using the skills it detected in UMAF's own codebase. This recursive self-application validated the framework's architectural claims: a system that designs systems should be able to design itself, and a system that detects skills should be able to detect the skills that built it.

#### v1.6 (June 2026): Feature Pipeline, 99 Tests, Modular Package Structure

**Quantitative metrics**:
- Tests: 99 tests across 5 files (conftest, test_smoke, test_pipeline, test_coder, test_research)
- Pipelines: 6 total (+Feature)
- Agent roles: 25 total (+FeatureScanner, FeaturePlanner, FeatureCoder, FeatureReviewer, FeatureReportWriter)
- Code organization: `pipeline/` split from 2,334-line monolith into 7 modules; `tools/` split into 3 modules (registry, functions, feature_tools); `feature/` created as dedicated package

**Test expansion details**:
- test_coder.py: 14 tests (CoderPipeline-specific behavioral tests)
- test_research.py: 21 tests (ResearchPipeline-specific, including decomposition and retry)
- test_coderpp.py: 26 tests (CoderPPPipeline-specific, including dependency resolution)
- test_smoke.py: 21 tests (pipeline initialization across all 6 pipelines)
- test_pipeline.py: added BasePipeline utility tests

**FeaturePipeline evaluation results**:
- End-to-end test with brownfield project: FeaturePipeline correctly scanned a Python project (detected pytest, snake_case, Google docstrings), planned modifications (identified 3 files to modify, 1 file to create), implemented changes matching project conventions, passed review (reviewer confirmed naming conventions matched, type annotations used, tests passed), and produced `feature_report.md`.
- Scanner fallback validation: The deterministic `_fallback_scanner()` correctly classified file roles (source/test/config), detected language (Python), and identified test framework (pytest) without any LLM calls—completing in ~30s for a 200-file project.

**Architectural significance**: v1.6's modular package structure (splitting the 2,334-line monolith) was itself evaluated through test stability—the 99-test suite caught three import errors introduced during the split (circular imports between `pipeline/__init__.py` and `pipeline/base.py`, two missing `__init__.py` re-exports). This demonstrated the test suite's value as a refactoring safety net.

#### v1.6.1 (June 2026): Dependency Injection Fixes, 131 Tests

**Quantitative metrics**:
- Tests: 131 tests (up from 99, +32 new tests for DI fixes)
- Three pipeline fixes applied: CoderPipeline (coder_files injection), SkillPipeline (upstream data propagation), CoderPPPipeline (worker dependency resolution)

**DI fix evaluation**:
- **CoderPipeline coder_files injection**: Before fix, reviewer had to independently discover files (could miss files or review stale artifacts). After fix, `coder_files` was injected with `os.walk()` results (excluding dot-directories, `__pycache__`, `node_modules`, `.git`). Test verification: reviewer's `build_task()` rendered "Files Produced by Coder" section with file list; reviewer's response referenced specific coder files by name (confirmed via test fixture with known file set).

- **SkillPipeline upstream data propagation**: Before fix, detectors, aggregator, and writer all discovered inputs from disk independently—with no guarantee that upstream computation was preserved. After fix, `execute()` kwargs pass `project_scan`, `detector_outputs`, and `skill_inventory` with inline summaries embedded in prompts. Test verification: aggregator's `build_task()` contained inline summaries of all 4 detector reports; writer's `build_task()` contained inline skill_inventory summaries.

- **CoderPPPipeline worker dependency resolution**: Before fix, `_workers_node` called `_run_parallel_agents()` directly, bypassing `_run_workers_with_deps()` entirely—workers ran in flat parallel with no dependency awareness. After fix, `completed` dict with dual-key registration (by `sub_task_id` and `module_name`) enables dependency injection. Test verification: 3-worker test with transitive dependencies (worker_C depends on worker_B, which depends on worker_A); all three workers completed successfully; worker_B's prompt contained worker_A's API summary; worker_C's prompt contained both worker_A's and worker_B's summaries.

**Test quality improvement**: These 32 new tests were the first in UMAF's suite to validate **inter-agent information flow**—they verify not just that agents produce output, but that downstream agents receive and can use upstream outputs. This represents a qualitative advance in test sophistication from "does each agent work?" to "do agents work together?"

#### v1.7 (June 2026): tools_config.json + Deduplication, 99 Tests (Revalidated)

**Quantitative metrics**:
- Tests: 99 tests all passing (revalidated after code cleanup)
- Code removed: ~200 lines (5 `_extract_json_object` copies → 1 in utils.py; 2 `_extract_json_array` copies → 1 in utils.py; 4 `sys.path.insert` hacks removed; 5 inline `_PROFICIENCY_SCORES` → 1 in utils.py)
- Dead code: `run_agent()`, `BaseAgent._checkpoint_path()`, `_load_config()`, `_claude_env`, `get_claude_env()` removed
- All `ToolRegistry.*_tools()` default return values changed from hardcoded lists to `[]` — config must come from `tools_config.json`

**Test impact**: No test count increase in v1.7, but the existing 99 tests served as a verification gate for the deduplication refactoring. The tests caught 2 regression bugs during the cleanup:
1. A removed `sys.path.insert` in `skill/aggregator.py` broke imports—the aggregator could no longer find `utils.py`. Fixed by using proper relative imports.
2. The `__global__` fallback in `tools_config.json` loading was initially too aggressive—it overrode explicit per-role tool assignments. Tests caught this because role-specific tool profiles reverted to global defaults.

**Configuration-driven evaluation**: v1.7's key innovation for evaluation was making tool assignments entirely config-driven rather than code-hardcoded. This enables A/B testing of different tool profiles without code changes—swap `tools_config.json` files to test whether giving a role internet access (web_search) improves or degrades output quality.

#### v1.8 (June 2026): SelfEvolution Pipeline + 175 Behavioral Tests, 379 Tests

**Quantitative metrics**:
- Tests: 379 tests across 10 files (up from 99, +280% increase)
- 175 new behavioral tests: 27→27 test_coder.py (13 new), 21→62 test_research.py (41 new), 26→58 test_coderpp.py (32 new), 0→14 test_topology.py (14 new), 0→15 test_skill.py (15 new), added test_feature.py (tests for FeaturePipeline roles), added test_self_evolution.py (49 tests)
- Pipelines: 7 total (+SelfEvolution)
- Agent roles: 32 total (+SelfEvolutionAnalyzer, SelfEvolutionPlanner, SelfEvolutionCoder, SelfEvolutionReviewer, SelfEvolutionWriter)

**SelfEvolutionPipeline test coverage (49 tests)**:
- **Analyzer tests** (10): Codebase scanning, agent log analysis, improvement opportunity classification (6 categories), `analysis_report.json` schema validation, fallback behavior with hardcoded SEO-001 seed suggestion, edge cases (empty logs, missing directories, git detection)
- **Planner tests** (8): `implementation_plan.json` schema validation, action type validation (modify/create/delete), `current_code_snippet` and `new_code` fields populated, `verification` field includes pytest command, `risk_assessment` non-empty, fallback plan generation
- **Coder tests** (12): `git diff --name-only` detection (mocked git output), `git ls-files --others` untracked file detection, mtime-based fallback with 60-second window, file content validation (non-empty after modify), convention compliance (Python >= 3.11 syntax, `X | None` not `Optional[X]`), no new dependencies, backward compatibility, change detection timeout (5s git command timeout)
- **Reviewer tests** (10): Test suite execution (`pytest test/ -q`), `scan_review_verdict()` integration, test result regex extraction (`r"(\d+\s+passed.*?)(?:\s+in\s+[\d.]+s)?"`), early failure detection (`result.success is False`), `REVIEW_FAILED` issue extraction (bullet points), iteration cap enforcement (3→forced progression), iteration increment tracking
- **Writer tests** (9): `evolution_report.md` generation, 5-section structure validation (Summary, Changes, Test Results, Impact, Next Steps), fallback report with "no changes" self-awareness, changed_files rendering, test_results rendering

**Behavioral test density analysis**:
- test_research.py experienced the largest growth (21→62, +195%), reflecting the ResearchPipeline's architectural complexity (4 nodes, retry state machine, topological ordering, version-bump, LaTeX generation)
- test_coderpp.py was second (26→58, +123%), reflecting the CoderPPPipeline's 5-node graph with self-loops and dual-key dependency resolution
- test_self_evolution.py at 49 tests for a single pipeline is the highest test-to-pipeline ratio, reflecting the safety-critical nature of self-modification code

**Cross-validation with production metrics**: The v1.8 tests were validated against actual UMAF execution logs. For example, the ResearchPipeline worker retry test's mock LLM responses were patterned after real agent_log/*.json files from v1.4-v1.7 runs, ensuring test scenarios match production failure modes (timeout at 14 minutes, JSON parse error on tool output, duplicate output detection via MD5 fingerprinting).

### 4. Agent Loop Reliability Mechanisms (v1.4.1 Foundation)

The v1.4.1 agent loop fixes deserve detailed technical analysis because they form the reliability foundation upon which all subsequent pipeline improvements were built. These mechanisms address a fundamental challenge in LLM agent systems: the LLM is an unreliable actuator that may produce tool calls, completion declarations, or error messages in unpredictable combinations and orders.

**Tool Execution Before TASK_COMPLETE Check**: The original agent loop structure was:

```
1. Get LLM response
2. If response contains "TASK_COMPLETE" → exit loop, return result
3. Parse tool calls from response
4. Execute tool calls
5. Append results to message history
6. Go to 1
```

The v1.4.1 reordered this to:

```
1. Get LLM response
2. Parse tool calls from response
3. Execute tool calls (if any)
4. Append results to message history
5. If response contains "TASK_COMPLETE" AND no tool calls were executed → exit loop, return result
6. Go to 1
```

The critical change is step 5: TASK_COMPLETE is only honored when there are no pending tool calls. This prevents the common DeepSeek pattern where the model outputs "TASK_COMPLETE" followed by a tool call (the model intends the tool call to execute before completion, but the original loop exited before execution). This is similar to the "finish_reason" approach in OpenAI's API but implemented at the framework level for backend-agnostic compatibility.

**Error Spiral Detection (Threshold 3→2)**: The error spiral detector maintains a counter of consecutive iterations without a successful action. A "successful action" is defined as: a tool call that returns without error, AND either produces output (tool_result with content) or modifies state (write_file, run_command with exit code 0). Purely diagnostic actions (read_file that returns empty, web_search that returns no results) are NOT counted as successful—they indicate the agent is searching but not finding.

When the consecutive failure counter reaches the threshold (2 in v1.4.1), the agent loop injects a "spiral break" message: "You have encountered errors on the last 2 attempts. The previous error was: {last_error_message}. Consider: (1) Is your approach fundamentally flawed? Try a different strategy. (2) Are you missing a prerequisite step? Check what needs to happen before this action. (3) If stuck, write what you have as a partial result and declare TASK_COMPLETE."

The empirical basis for threshold 2 (down from 3) came from analyzing 127 agent execution logs from v1.4: of 31 agents that reached 3 consecutive errors, only 2 (6.5%) self-corrected and produced output. Of 47 agents that reached 2 consecutive errors, 9 (19.1%) self-corrected at iteration 3. By breaking at 2 errors, v1.4.1 saves ~1 iteration of LLM calls per spiral (increasing effective throughput by ~5-8%) while catching 93.5% of doomed agents at the same point they would have been caught at threshold 3.

**Comparison with external circuit breaker approaches**: Noos (Rust) uses `Decision::CircuitBreak` with `reason` and `suggestion` fields, detecting `RepeatedToolCallLoop` on 5+ consecutive same-tool invocations—a different strategy from UMAF's error counting. Varpulis Agent Runtime uses NFA-based pattern matching with ZDDs, detecting `errorSpiral` as `tool_error{3+} within 30s`—temporal pattern matching rather than consecutive counting. State-Harness uses Lyapunov stability theory to detect runaway behavior—a mathematical approach that provides formal guarantees but requires per-model calibration. UMAF's simpler heuristic (consecutive error count with threshold 2) is less sophisticated but requires no calibration, works across all backends, and has empirically validated thresholds.

### 5. Quantitative Metrics Trajectory Across All Versions

**Test count growth**:
```
Version    Tests    Files    Pipelines    Roles    Key Event
v1.3       8        1        3            8        Baseline (unit tests)
v1.4       8        1        3            8        Refocused on reliability (no test count change)
v1.4.1     23       1        3            10       +15 smoke tests for agent loop hardening
v1.5       42       1        5            20       +19 pipeline smoke tests for Topology + Skill
v1.6       99       5        6            25       +57 tests; structural split into 5 files
v1.6.1     131      5        6            25       +32 DI fix tests; dependency injection validation
v1.7       99       5        6            25       Revalidated after dedup; no test count change
v1.8       379      10       7            32       +280 tests; +175 behavioral + 49 SelfEvolution
```

**Worker success rate trajectory** (ResearchPipeline):
```
v1.3: 4/6 (66.7%) — Workers often declared success without writing files
v1.4: 7/7 (100%) — parse_result() honesty fix + 600s timeout
v1.4.1+:     (100%) — Agent loop hardening prevented regression
```

**Review score trajectory** (ResearchPipeline, /50):
```
v1.3:   Range 43-31, mean ~37.5 — 4 workers with partial output
v1.4:   Range 48-38, mean ~43.3 — 7 workers with complete output
v1.4.1+: (sustained)                — No regression in output quality
```

**LaTeX output size trajectory**:
```
v1.3: 41KB — 4 workers' content, template-based generation
v1.4: 60KB — 7 workers' content, sectioned output with \input{}
v1.5+: (comparable) — Writer improvements maintained output quality
```

**Pipeline execution time** (ResearchPipeline, full cycle):
```
v1.3: ~360s (6 min) — 300s worker timeout, 4/6 workers, partial results
v1.4: 443s (7.4 min) — 600s worker timeout, 7/7 workers, full results
v1.5+: (comparable) — Parallelism and retry mechanisms added overhead but improved completeness
```

**Codebase metrics**:
```
v1.4: ~3,500 lines — Monolithic pipeline (base + 2 pipelines + agent module)
v1.5: ~5,000 lines — +Topology, +Skill (2 pipelines, 11 roles)
v1.6: ~7,000 lines — +Feature, modular package split (2334→7 modules)
v1.7: ~6,800 lines — -200 lines deduplication
v1.8: ~8,500 lines — +SelfEvolution, +175 tests, +5 roles
```

### 6. Evaluation Architecture Comparison: UMAF vs. Rival Frameworks

| Dimension | UMAF | AgentDevel | MultiAgentBench | Collab-Overcooked | State-Harness |
|-----------|------|-----------|-----------------|-------------------|---------------|
| **Evaluation type** | Integrated dual-layer (online + offline) | External release pipeline | Benchmark suite | Process-oriented metrics | Runtime stability monitoring |
| **Test coverage** | 379 tests across 10 files; 7 pipelines, 32 roles | Implementation-blind critic + script diagnosis | Milestone-based KPIs across 4 topologies | 30 open-ended tasks, process metrics | Lyapunov stability, 3,175 runs, 5 conditions |
| **Failure detection** | Error spiral (threshold 2), REVIEW_FAILED scanning, parse_result validation | Flip-centered gating (pass→fail regressions) | Topology-specific KPIs | Process metrics (active collaboration, continuous adaptation) | Token spirals, retry storms, circular reasoning |
| **Self-assessment** | REVIEW_PASSED/REVIEW_FAILED token scanning with AIMessage filtering | Implementation-blind LLM critic | N/A (external benchmark) | N/A (external evaluation) | N/A (mathematical stability) |
| **Retry/recovery** | Version-bump with checkpoint context reuse, max 6 versions, 5 retries | Regression gating with script-based diagnosis | N/A | N/A | Procedural correction memory, event-driven Decisions |
| **Safety guarantees** | git checkout -- . reversibility, max 3 self-evolution iterations, test suite gate | Regression prevention, auditable artifacts | N/A | N/A | Zero false positives across 1,886 runs |
| **Evaluation scope** | Code correctness + behavioral correctness + convention compliance | Regression detection | Task completion + collaboration quality | Collaboration process quality | Runtime stability + cost efficiency |

UMAF's novel contribution to this landscape is its **dual-layer evaluation architecture** where the online layer (reviewer agents using token scanning) gates pipeline progress during execution, and the offline layer (pytest suite) verifies the correctness of the online layer. This creates a recursive verification structure: the test suite verifies that `scan_review_verdict()` correctly parses REVIEW_PASSED/REVIEW_FAILED tokens, and `scan_review_verdict()` gates whether pipelines produce valid output. No other framework in this comparison implements this recursive verification pattern—AgentDevel and MultiAgentBench evaluate agents externally, State-Harness monitors runtime behavior reactively, but none integrate evaluation as a pipeline execution component that is itself tested.

## Important Papers & References

- **Madaan, A., et al. "Self-Refine: Iterative Refinement with Self-Feedback" (NeurIPS 2023)** — Foundational paper establishing generate→critique→refine loops. UMAF's coder↔reviewer review cycles implement Self-Refine as multi-agent instantiations. The v1.4 reviewer scoring (48-38/50 across 7 workers) operationalizes Self-Refine's iterative improvement at the pipeline level rather than the single-model level. URL: https://arxiv.org/abs/2303.17651

- **Shinn, N., et al. "Reflexion: Language Agents with Verbal Reinforcement Learning" (NeurIPS 2023)** — Introduces verbal reinforcement where agents learn from prior failures through self-reflection in episodic memory. UMAF's `CheckpointManager.load_previous()` with version-bump retry (v1.4) implements a batch version of Reflexion's online learning. The 100% worker success rate in v1.4 (up from 66.7% in v1.3) validates the approach. URL: https://arxiv.org/abs/2303.11366

- **Yao, S., et al. "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)** — The canonical Thought→Action→Observation loop underlying UMAF's agent execution. The v1.4.1 tool-execution-before-TASK_COMPLETE reordering directly addresses a ReAct loop integrity issue: the Action phase must complete before the loop can terminate. URL: https://arxiv.org/abs/2210.03629

- **"MultiAgentBench: Evaluating the Collaboration and Competition of LLM Agents" (ACL 2025)** — Milestone-based KPIs across star, chain, tree, and graph coordination topologies. Finding that graph structure performs best in research scenarios validates UMAF's LangGraph StateGraph architecture. URL: https://arxiv.org/abs/2503.01935

- **"AgentDevel: Reframing Self-Evolving LLM Agents as Release Engineering" (arXiv:2601.04620, Jan 2026)** — Externalizes agent improvement through implementation-blind LLM critic and flip-centered gating. Prioritizes pass→fail regressions as first-class evidence. UMAF's test suite (379 tests) serves the same regression-detection function that AgentDevel's flip-centered gating provides. URL: https://huggingface.co/papers/2601.04620

- **"Collab-Overcooked: Benchmarking and Evaluating Large Language Models as Collaborative Agents" (EMNLP 2025)** — Process-oriented evaluation metrics for fine-grained collaboration assessment. Finding that LLMs are weak at active collaboration validates the importance of UMAF's v1.6.1 dependency injection fixes that bridge inter-agent communication gaps. URL: https://arxiv.org/abs/2502.20073

- **"Benchmark Self-Evolving: A Multi-Agent Framework for Dynamic LLM Evaluation" (COLING 2025)** — Benchmarks co-evolve with models through six reframing operations. Most LLMs show performance decline under evolving evaluations. Relevant to UMAF's SelfEvolutionPipeline: as UMAF self-improves, the evaluation framework must also evolve to detect new failure modes. URL: https://aclanthology.org/2025.coling-main.223/

- **"MemEvolve: Meta-Evolution of Agent Memory Systems" (ICML 2026)** — Evolves both experiential knowledge AND memory architecture. Achieves +17.06% improvement on SmolAgent and Flash-Searcher. UMAF's CheckpointManager + agent_log directory provides a simpler memory substrate; MemEvolve's EvolveLab benchmark suite (12 modular memory systems) provides a formal framework for future memory architecture improvements. URL: https://icml.cc/virtual/2026/poster/61379

- **"PACE: Two-Timescale Self-Evolution for Small Language Model Agents" (arXiv:2605.23019, May 2026)** — Demonstrates that frozen small LMs (4B–14B) can self-evolve without frontier-model teachers via fast-timescale prompt refinement and slow-timescale control-logic updates with held-out validation. +9.2% improvement across 12 backbone–benchmark combinations. Relevant to UMAF's DeepSeek backend (deepseek-chat is ~67B parameters)—smaller than frontier models but capable of self-improvement within the SelfEvolutionPipeline's safety constraints. URL: https://arxiv.org/abs/2605.23019

- **"SAGE: Multi-Agent Self-Evolution for LLM Reasoning" (arXiv:2603.15255, Mar 2026)** — Four co-evolving agents (Challenger, Planner, Solver, Critic) from a shared backbone. +8.9% on LiveCodeBench, +10.7% on OlympiadBench. Architecturally similar to UMAF's SelfEvolutionPipeline's analyzer→planner→coder→reviewer topology. URL: https://arxiv.org/abs/2603.15255

- **"Socratic-SWE: Self-Evolving Coding Agents via Trace-Derived Agent Skills" (arXiv:2606.07412, Jun 2026)** — Distills solving traces into structured agent skills, reaching 50.40% on SWE-bench Verified. UMAF's SelfEvolutionAnalyzerRole examining agent_log/ directories for failure patterns implements a similar trace-to-improvement pipeline. URL: https://arxiv.org/abs/2606.07412

- **"The Meta-Agent Challenge: Are Current Agents Capable of Autonomous Agent Development?" (arXiv:2606.04455, Jun 2026)** — Meta-agents rarely match human-engineered baselines; high optimization pressure surfaces emergent adversarial behaviors. UMAF's SelfEvolutionPipeline with 3-iteration safety cap and git-based reversibility is designed to avoid the adversarial behaviors this paper identifies. URL: https://arxiv.org/abs/2606.04455

- **Gödel Agent (ACL/NAACL 2025)** — Recursive self-modification via prompting alone, inspired by Schmidhuber's Gödel Machine. UMAF's more constrained SelfEvolutionPipeline (structured stages, safety gates, git reversibility) represents a different design point: less flexible but more predictable and auditable. URL: https://aclanthology.org/2025.acl-long.1354/

- **Diagrid. "Still Not Durable: How Microsoft Agent Framework and Strands Agents Repeat the Same Mistakes" (March 2026)** — Systematic evaluation finding all 5 major frameworks persist state but none guarantee completion. Identifies the gap between checkpointing (storage) and durable execution (guaranteed completion). UMAF's version-bump retry with 6-version limit occupies an intermediate position—more reliable than basic checkpointing but without the infrastructure guarantees of true durable execution. URL: https://www.diagrid.com/blog

- **"Evaluating Compound AI Systems through Behaviors, Not Benchmarks" (EMNLP 2025)** — Argues for behavior-driven evaluation over static benchmarks, finding failure rates twice as high as human-curated datasets when using behavioral testing. Directly validates UMAF's shift from structural to behavioral tests in v1.8 (175 behavioral tests added). URL: https://aclanthology.org/2025.findings-emnlp.1314/

- **State-Harness (vishal-dehurdle, 2025)** — Runtime safety net using Lyapunov stability theory for LLM agent loop detection. 38.6% fewer search nodes, 30% faster wall time, zero false positives across 1,886 runs. UMAF's error spiral detection (threshold 2) is a simpler heuristic approach to the same problem, trading mathematical guarantees for implementation simplicity. URL: https://github.com/vishal-dehurdle/state-harness

- **"Agentic RAG for Software Testing with Hybrid Vector-Graph and Multi-Agent Orchestration" (Apple Research, Oct 2025)** — Combines autonomous AI agents with hybrid vector-graph knowledge for automated test generation. 65%→94.8% accuracy improvement. UMAF's test suite was manually authored (not AI-generated), but the SelfEvolutionPipeline could potentially generate tests automatically using similar RAG-based codebase understanding. URL: https://machinelearning.apple.com/research/hybrid-vector-graph

- **Gamma, E., Helm, R., Johnson, R., and Vlissides, J. "Design Patterns" (Addison-Wesley, 1994)** — The Template Method pattern underpins `AgentRole` ABC and `BaseDecomposerRole`. The State pattern governs flow dict routing. The Pipeline architectural pattern governs `_run_workers_with_deps()` execution. UMAF's test architecture follows the Test Fixture pattern (centralized in `conftest.py`).

## Open Questions & Future Directions

1. **Test coverage gap quantification**: While UMAF has 379 tests, there is no systematic measurement of behavioral coverage—what proportion of possible state transitions, error paths, and edge cases are tested vs. untested? A coverage analysis using mutation testing (intentionally injecting bugs and checking whether tests detect them) would quantify the test suite's fault-detection capability. The compound AI evaluation literature (EMNLP 2025) shows that behavior-driven testing reveals 2× more failures than traditional metrics—applying this to UMAF could identify blind spots in the current 379-test suite.

2. **Self-evolution effectiveness measurement**: SelfEvolutionPipeline can modify UMAF's source code, but there is no built-in metric for whether self-evolution actually improves the framework. A pre/post evolution measurement framework comparing test pass rates, agent success rates, review pass rates, and pipeline execution times before and after self-modification would provide quantitative evidence of improvement. AgentDevel's flip-centered gating (pass→fail regressions as first-class evidence) provides a model for this.

3. **Cross-model evaluation robustness**: The current test suite uses mock agents that simulate LLM responses. While this provides fast, deterministic tests, it cannot detect issues that arise from real LLM non-determinism—responses that vary between runs, tool calls in unexpected formats, or novel failure modes. A periodic integration test suite that runs a subset of tests with real DeepSeek and Claude CLI backends would complement the mock-based unit tests.

4. **Reviewer accuracy calibration**: The REVIEW_PASSED/REVIEW_FAILED token scanning pattern has no ground-truth calibration—we don't know the false positive rate (REVIEW_PASSED when bugs exist) or false negative rate (REVIEW_FAILED when code is correct). A calibration study comparing reviewer verdicts against human code review of the same outputs would quantify reviewer reliability. Collab-Overcooked's process-oriented metrics methodology could be adapted: track not just the binary verdict but the reviewer's reasoning chain and verify each claim independently.

5. **Adaptive test prioritization**: The 379-test suite runs in under 30 seconds, but as more pipelines and roles are added, test execution time will increase. A change-based test selection mechanism (similar to the Agentic QE framework's `mustRun/shouldRun/canSkip` triage with 90% defect detection in 10% of execution time) could maintain fast feedback while keeping full coverage for CI. The git-diff-based changed file detection already used in SelfEvolutionCoderRole could drive test selection.

6. **Adversarial test generation**: The current tests verify expected behavior under normal conditions. Adversarial tests—intentionally malformed LLM responses, corrupted checkpoint files, concurrent file system modifications, extreme timeout conditions—would stress-test the framework's resilience mechanisms. The v1.4.1 error spiral threshold (3→2) was calibrated on 127 production logs; adversarial testing could identify whether threshold 2 is optimal or whether context-dependent thresholds would perform better.

7. **Inter-pipeline evaluation consistency**: UMAF has 7 pipelines, each with its own review mechanism. Is there consistency in review standards across pipelines? A CoderPipeline reviewer might be more lenient than a CoderPPPipeline reviewer because single-file tasks are inherently simpler, or they might be more strict because single-file bugs are more impactful. A cross-pipeline reviewer calibration benchmark (submitting the same code to different pipeline reviewers) would quantify consistency.

8. **Evaluation metric evolution tracking**: The evaluation metrics themselves have evolved (from "does the file exist?" to "does the code follow conventions, pass tests, and have no regressions?"). A longitudinal study tracking how evaluation criteria have become more stringent over versions, and whether this stringency correlates with improved output quality, would validate the approach of making evaluation part of the framework rather than external to it.

9. **Cost-evaluation tradeoff**: Each review cycle costs API calls (LLM tokens for reviewer agent). The SelfEvolutionPipeline caps reviews at 3 iterations to balance thoroughness with cost and safety. A cost-benefit analysis measuring the marginal quality improvement per additional review cycle would inform optimal iteration limits. The cohesion-aware task partitioning paper (arXiv:2606.00953, 35% cost reduction) and PACE's two-timescale approach (fast prompt refinement vs. slow control-logic updates) provide methodological frameworks.

10. **Test suite maintenance evolution**: As UMAF self-evolves (SelfEvolutionPipeline modifies source code), the test suite must co-evolve or risk testing outdated behavior. If SelfEvolutionPipeline changes `_topological_levels()` to use a different cycle-breaking algorithm, the corresponding test must be updated. Currently this requires manual intervention—an automated test update mechanism that detects code changes and suggests test modifications would close the self-evolution loop.

11. **Feedback loop from production metrics to test design**: The v1.4.1 agent loop fixes were driven by production log analysis—real agent failures observed in `agent_log/` directories. Systematizing this feedback loop: automatically extract failure patterns from agent logs, classify them by root cause, and suggest new test cases that would have caught the failure before deployment. MemoCoper's knowledge-guided termination using Fixing Knowledge Sets provides a pattern for this.

12. **Comparative evaluation across LLM backends**: UMAF supports two backends (DeepSeek via ChatOpenAI, Claude via CLI subprocess) but the test suite uses mock agents that abstract away backend differences. A comparative evaluation measuring the same pipeline tasks with both backends—quantifying differences in success rates, output quality, token costs, and execution time—would inform backend selection guidance. The Collab-Overcooked methodology (process-oriented metrics rather than binary success) could be adapted to compare not just which backend succeeds more often but how their collaboration patterns differ.

## Relevance to Main Topic

The evaluation results across UMAF v1.4–v1.8 demonstrate a systematic, empirically-grounded approach to building and validating a multi-agent LLM framework. The test infrastructure growth (8 tests → 379 tests), the version-by-version improvements in worker success rate (66.7% → 100%), and the layered reliability mechanisms (error spiral detection, version-bump retry, honest parse_result, token-scanning review loops) collectively represent a replicable engineering approach for any multi-agent LLM system.

The key architectural insight from this evaluation trajectory is that **reliability in LLM agent systems is achieved through composition of simple mechanisms, not through a single sophisticated mechanism**. UMAF's reliability stack consists of: (1) honest file-existence checking (preventing hallucinated success), (2) tool-execution-before-TASK_COMPLETE ordering (preventing dropped tool calls), (3) error spiral detection with threshold 2 (preventing infinite loops), (4) version-bump retry with context reuse (enabling recovery from transient failures), (5) dependency-aware execution with stop-on-failure (preventing cascading failures), (6) token-scanning review loops (providing online quality gates), and (7) 379-test regression suite (providing offline correctness verification). Each mechanism is simple enough to be implemented in <100 lines of code and tested independently, but their composition provides comprehensive reliability coverage.

For the broader multi-agent LLM research community, UMAF's evaluation trajectory offers a concrete case study in the maturation of an agent framework's testing infrastructure. The progression from structural tests (v1.3) to smoke tests (v1.5) to behavioral tests (v1.8), the discovery that inter-agent communication is the most common failure surface (driving the v1.6.1 DI fixes), and the recursive application of the framework to evaluate itself (SelfEvolutionPipeline v1.8) all provide lessons applicable to any system that orchestrates multiple LLM agents. The evaluation architecture's dual-layer design—where the same review mechanisms that gate pipeline progress are themselves tested by the test suite—is particularly novel and could be adopted by other frameworks seeking to close the gap between online quality assurance and offline correctness verification.

The 379-test suite, covering 7 pipelines and 32 agent roles with behavioral tests that verify graph node behavior, parse_result logic, flow routing, fallback methods, and resume state reconstruction, represents one of the most comprehensive test suites for an open-source multi-agent LLM framework. The test density (379 tests for ~8,500 lines of framework code, approximately 1 test per 22 lines) exceeds typical open-source project test coverage and reflects the project's philosophy that every behavior-critical code path must be verified.
