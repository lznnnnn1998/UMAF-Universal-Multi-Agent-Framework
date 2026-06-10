# FeaturePipeline and SelfEvolutionPipeline: Project-Aware Code Generation and Autonomous Self-Improvement

## Overview

UMAF's FeaturePipeline (v1.6) and SelfEvolutionPipeline (v1.8) represent two advanced, complementary capabilities that extend the framework beyond greenfield code generation into brownfield development and meta-cognitive self-improvement. The FeaturePipeline is a 5-node cyclic graph (`scanner → planner → coder ↔ reviewer (max 5 cycles) → writer`) designed for adding, modifying, or refactoring code within **existing projects**. Unlike CoderPipeline and CoderPPPipeline—which generate code from scratch in an empty working directory—FeaturePipeline grounds every agent action in a detailed `project_context.json` produced by a scanner role that surveys the target project's language, conventions, test patterns, and file manifest. This project-awareness enables three critical capabilities absent from greenfield pipelines: (1) the planner can specify both `files_to_create` AND `files_to_modify`, supporting surgical edits to existing codebases; (2) the coder reads existing files before modifying them, matching all conventions from the project context; and (3) the reviewer validates not just correctness but convention compliance, ensuring modifications blend with the surrounding codebase.

The SelfEvolutionPipeline is a 5-node cyclic graph (`analyzer → planner → coder ↔ reviewer (max 3 iterations) → writer`) that inverts the framework's gaze: UMAF analyzes and improves **itself**. The analyzer scans UMAF's own codebase and agent execution logs (`agent_log/` directories) to identify improvement opportunities across six categories (prompt quality, parameter tuning, error handling, code quality, test gaps, configuration). The planner creates a concrete implementation plan targeting specific source files. The coder directly modifies UMAF's source code in-place—detecting changed files via `git diff --name-only` for tracked changes, `git ls-files --others --exclude-standard` for new files, and mtime-based fallback scanning for recently-modified `.py` files. The reviewer runs `pytest test/` to verify no regressions, using the same `REVIEW_PASSED`/`REVIEW_FAILED` token scanning pattern shared across CoderPipeline, CoderPPPipeline, and FeaturePipeline. Critically, the pipeline operates on the **current git branch** so all changes are revertible with `git checkout -- .`—a lightweight safety guarantee that avoids complex sandboxing while still enabling true self-modification.

The architectural motivation for these pipelines addresses two fundamental gaps in LLM agent frameworks. First, most code generation systems (including UMAF's CoderPipeline and CoderPPPipeline) treat every task as greenfield development, but real-world software engineering is overwhelmingly brownfield—modifying, extending, and refactoring existing codebases that carry years of accumulated conventions, patterns, and constraints. FeaturePipeline's scanner→planner→coder→reviewer topology instantiates the **context-grounded development** pattern, where every downstream agent makes decisions informed by empirically observed project realities rather than generic best practices. Second, while frameworks like GBase and Autogenesis explore recursive self-improvement through dedicated sandboxes and protocol layers, SelfEvolutionPipeline demonstrates that a lightweight, git-revertible approach can achieve meaningful self-improvement without infrastructure overhead—the analyzer identifies real issues from execution traces, the coder applies fixes, and the reviewer's test suite run provides a fast regression gate. Together, these pipelines complete UMAF's code generation capability spectrum: CoderPipeline (single-file), CoderPPPipeline (multi-file greenfield), FeaturePipeline (brownfield modification), and SelfEvolutionPipeline (self-modification).

## Key Methods & Approaches

### 1. FeaturePipeline: Scanner → Planner → Coder ↔ Reviewer → Writer

#### 1.1 Graph Topology and Flow Routing

FeaturePipeline's `StateGraph` (`pipeline/feature.py:36-226`) uses a `FeatureState` TypedDict with 13 fields: `input_spec`, `working_dir`, `backend`, `project_dir`, `status`, `iteration`, `project_context`, `implementation_plan`, `changed_files`, `review_passed`, `review_issues`, and `feature_report`. Unlike CoderPipeline's simple 2-node graph with a single `_router` function, FeaturePipeline uses three separate routers:

**Linear status router** (`feature.py:180-187`): Handles the scanner→planner→writer linear segments via `BasePipeline._status_router()` with flow map `{"initialized": "scanner", "scanned": "planner", "planned": "coder", "written": END}` and terminal errors `error_scanner_failed` and `error_planner_failed`.

**Coder router** (`feature.py:189-192`): A dedicated function that routes `coded → reviewer` or `END` for any other status. This ensures the coder always flows to the reviewer after implementation.

**Reviewer router** (`feature.py:194-200`): Implements the core review loop logic—`review_passed → writer`, `iteration < 5 → coder` (retry), `iteration >= 5 → writer` (force progression after max cycles). This is architecturally identical to CoderPipeline's `_router` but uses FeatureState's 13-field structure.

The entry point is always `scanner`—the pipeline starts with project analysis before any planning or coding. This is a critical design difference from CoderPipeline (entry point: coder) and SelfEvolutionPipeline (entry point: analyzer), reflecting the brownfield requirement that implementation must be informed by existing project context.

#### 1.2 FeatureScannerRole: Project Context Generation

The scanner (`feature/scanner.py:16-251`) is the gateway to project-aware development. Its `build_task()` method (`scanner.py:29-95`) provides a 5-step structured prompt:

1. **Structure discovery**: Use `find` or `ls` to map the project directory tree
2. **Configuration analysis**: Read `pyproject.toml`, `setup.cfg`, `package.json`, `requirements.txt` to determine language, version, framework, and dependencies
3. **Convention extraction**: Read 3-5 representative source files to identify naming conventions (snake_case vs camelCase), import grouping style (stdlib-first, absolute vs relative), type annotation usage level, docstring format (Google, NumPy, Sphinx, none), and error handling patterns (try/except, return None, raise)
4. **Test pattern detection**: Read 1-2 test files to determine test framework (pytest, unittest, jest), file naming convention (`test_*.py` vs `*_test.py`), fixture patterns (conftest.py, setup/teardown), and mock library
5. **Output generation**: Write `project_context.json` with the full schema including language, conventions, test_patterns, tech_stack, and file_manifest

The output schema defined in the prompt (`scanner.py:54-85`) is notably comprehensive—it specifies 8 top-level fields with nested objects for conventions (5 sub-fields), test patterns (4 sub-fields), tech stack (2 sub-fields), and file manifest (a list of `{path, role}` objects). This structured schema is what makes FeaturePipeline's project-awareness concrete: the planner, coder, and reviewer all read and reference this JSON file.

**Parse result** (`scanner.py:97-131`) follows the standard 3-tier extraction strategy: (1) scan agent messages for JSON objects containing `file_manifest` or `conventions` keys, (2) read `project_context.json` from disk if message extraction fails, (3) invoke `_fallback_scanner()` as a deterministic last resort.

**Deterministic fallback** (`scanner.py:134-251`): The `_fallback_scanner()` method is a 117-line static method that performs project analysis without any LLM calls. Its design is notable for its depth:
- Runs `find` with exclusion patterns for `.git`, `__pycache__`, `node_modules`, `.venv`, `venv`, `dist`, `build`, `.tox`—capped at 2000 files
- Classifies files into roles: test (in test directories like `tests`, `test`, `spec`, `specs`, `__tests__`), config (extensions `.json`, `.yaml`, `.yml`, `.toml`, `.cfg`, `.ini`, `.env` or names like `requirements.txt`, `package.json`), source (extensions `.py`, `.js`, `.ts`, etc.), or other
- Detects language by counting `.py` vs `.js`/`.ts` files
- Samples source files for convention detection: checks for `from __future__ import annotations` (type annotation level), `"""` (docstring format), `try:` (error handling)
- Detects test framework by reading test files for `pytest` references
- Produces a complete `project_context.json`-compatible dict with `_fallback: True` marker

This fallback is particularly important because project scanning via LLM can be slow and expensive for large codebases—a 2000-file project would require many sequential `read_file` calls. The deterministic fallback completes in ~30 seconds regardless of project size.

#### 1.3 FeaturePlannerRole: Dual Create/Modify Planning

The planner (`feature/planner.py:15-131`) introduces the key innovation of FeaturePipeline: planning for both new files AND modifications to existing files. Its `build_task()` (`planner.py:28-82`) reads `project_context.json` (displaying up to 8000 characters), then produces `implementation_plan.json` with three component lists:

**`files_to_create`**: Each entry specifies `path`, `description`, `interfaces` (list of functions/classes to implement), and `dependencies` (other files needed). This is similar to CoderPP's `files_to_create` but with added interface specifications.

**`files_to_modify`**: Each entry specifies `path` (the existing file to edit), `section` (where to edit—e.g., "after imports", "in class Foo", "at end of file"), `change` (what to change—e.g., "add import for X", "add call to Y"), and `description` (why this change is needed). The `section` field is the critical difference from CoderPP: it provides surgical targeting within existing files rather than requiring full file rewrites.

**`test_files`**: Each entry specifies `path` and `covers` (list of source files tested), ensuring test coverage is planned alongside implementation.

The planner's instructions enforce validation rules: no circular dependencies, all imports must resolve, and existing files must be READ before planning modifications. The `claude_cli` variant adds explicit instruction to "Read existing files before planning modifications."

**Parse result** (`planner.py:84-117`) uses the same 3-tier strategy: messages → disk file → `_fallback_plan()`. The `_fallback_plan()` (`planner.py:120-131`) produces an empty plan with `_fallback: True` and a note that "no AI planning was performed"—a graceful degradation that prevents pipeline crashes.

#### 1.4 FeatureCoderRole: Context-Grounded Implementation

The coder (`feature/coder.py:14-131`) is the most complex FeaturePipeline role (25-step budget, the highest of all feature roles). Its `build_task()` (`coder.py:28-95`) is structured as a 5-step ordered procedure:

1. **Read plan and context**: Understand both the plan and the existing project conventions
2. **Modify existing files FIRST**: For each `files_to_modify` entry, read the file, make the surgical change, and write the COMPLETE modified file—not just the changed portion. The order (modify before create) is deliberate: new files often depend on modifications to existing infrastructure (new imports, new routes, new configuration)
3. **Create new files**: Read dependency files first, then write complete files matching ALL project conventions
4. **Write tests**: Match test patterns from `project_context` exactly—same framework, same naming, same fixture patterns
5. **Run tests and fix**: Iterate until all tests pass

The coder receives `review_issues` when in a retry iteration, which are rendered as a "Previous Review Issues — FIX THESE" section with bullet points. This is architecturally identical to how CoderPipeline and FeaturePipeline's coder roles receive reviewer feedback.

**Critical rules** enforced in the prompt: "NEVER skip reading an existing file before modifying it," "Write the COMPLETE file when modifying—not just the changed part," "Match ALL conventions from project_context exactly," and "Write EVERY file listed in the plan."

**Parse result** (`coder.py:97-131`) is simpler than other parse_result methods: it reads the implementation plan from disk, then checks `os.path.isfile()` and `os.path.getsize() > 0` for each planned file (create, modify, and test). This yields a `changed_files` list validated against the actual plan—a more structured approach than the general `os.walk()` scan used in CoderPipeline's `_coder_node` for `coder_files` injection. Importantly, it only reports files that were planned, not every file in the working directory.

#### 1.5 FeatureReviewerRole: 4-Dimensional Validation

The reviewer (`feature/reviewer.py:13-101`) evaluates implementation across four dimensions:

1. **Completeness**: Have ALL `files_to_create` been written? Have ALL `files_to_modify` been correctly edited? Are ALL `test_files` present?
2. **Correctness**: Do imports resolve? Are there circular dependencies? Do tests pass when run? Are all interfaces actually implemented?
3. **Convention Compliance**: Does naming match `project_context`? Are type hints used consistently? Are docstrings present in the correct format? Does import style match the project?
4. **Test Quality**: Do tests cover happy path, edge cases, and error handling? Are they properly structured with fixtures? Do they assert meaningful behavior?

The build_task receives `project_context` (truncated at 4000 chars), `implementation_plan` (truncated at 6000 chars), and the `changed_files` list from the coder. It is explicitly instructed to "Read every changed file" and "Run the tests to verify they pass."

**Parse result** (`reviewer.py:75-100`) uses `utils.scan_review_verdict()` for token scanning—the same shared function used by CoderPipeline and SelfEvolutionPipeline. When `REVIEW_FAILED` is detected, it extracts issue descriptions from bullet points (lines starting with `- ` or `* `) and lines containing "issue" or "fail" (minimum 10 chars to filter noise). Issues are capped at 20 to prevent state bloat. The extracted issues are stored in `review_issues` and passed back to the coder in the next iteration—completing the review loop.

**Tool profile**: The reviewer gets only `read_file` and `run_command` (2 tools)—the most restricted profile of any FeaturePipeline role. This enforces the critique-only pattern: the reviewer cannot modify code, only read it and run tests. This is identical to CoderPipeline's reviewer tool restriction (no `write_file`), reflecting the same design philosophy that reviewers should evaluate, not fix.

#### 1.6 FeatureReportWriterRole: Terminal Documentation

The writer (`feature/writer.py:14-94`) is a minimal role (5-step budget) that produces `feature_report.md` with a structured 7-section format: Summary, Files Created, Files Modified, Conventions Followed, Test Results, Review Results, and Known Limitations. It reads `implementation_plan.json` (truncated at 5000 chars) and receives `changed_files`, `review_passed`, and `review_issues` via context. The parse method checks for `feature_report.md` on disk and falls back to a deterministic `_fallback_report()` that generates a basic markdown template.

### 2. SelfEvolutionPipeline: Analyzer → Planner → Coder ↔ Reviewer → Writer

#### 2.1 Graph Topology and Flow Routing

SelfEvolutionPipeline's `StateGraph` (`pipeline/self_evolution.py:48-235`) uses a `SelfEvolutionState` TypedDict with 13 fields: `input_spec`, `working_dir`, `backend`, `project_dir`, `status`, `iteration`, `analysis_report`, `implementation_plan`, `changed_files`, `review_passed`, `review_issues`, `test_results`, and `evolution_report`. Unlike FeaturePipeline's multiple routers, SelfEvolutionPipeline uses a single `_status_router` with flow map:

```
analyzed → planner
planned → coder
implemented → reviewer
verified → writer
plan_revision → coder
completed → END
```

The `plan_revision` status is the key self-loop mechanism. In `_reviewer_node` (`self_evolution.py:145-178`), the reviewer sets `status` to `verified` (when tests pass) or `plan_revision` (when tests fail). The `plan_revision` status routes back to the coder for another implementation attempt. However, there's a critical safety gate at line 161-169: when `iteration >= MAX_ITERATIONS (3)`, the reviewer **forces progression** to `verified` status even on failure, printing a warning message. This limits the self-modification loop to at most 3 cycles—lower than FeaturePipeline's and CoderPipeline's 5—explicitly designed to "balance thoroughness with the risk of destructive self-modification."

The `MAX_ITERATIONS = 3` constant is a class-level attribute (`self_evolution.py:54`), making it easy to adjust. The rationale for 3 vs. 5 is that self-modification carries higher risk than feature implementation: each cycle modifies UMAF's own source code, and excessive iteration increases the chance of introducing bugs that the reviewer's test suite might miss.

#### 2.2 SelfEvolutionAnalyzerRole: Codebase + Log Analysis

The analyzer (`self_evolution/analyzer.py:13-159`) is unique among all 32 UMAF roles: it is the only role that analyzes UMAF itself. Its `build_task()` (`analyzer.py:26-87`) provides a 4-step structured prompt:

1. **Scan codebase structure**: Run `find` to map file layout, read key files to understand pipeline architecture, agent role definitions, tool registry, and test coverage
2. **Analyze agent logs**: Check `agent_log/` directories for execution patterns—which agents succeed vs fail, common failure modes (timeout, parse error, missing output), average iteration counts, tool call patterns
3. **Identify improvement opportunities**: Categorize findings into 6 buckets:
   - **Prompt Quality**: Vague, ambiguous, or missing instructions in `build_task()` methods
   - **Parameter Tuning**: Timeouts, `max_steps`, retry limits that are too low/high
   - **Error Handling**: Missing fallbacks, ungraceful failure modes
   - **Code Quality**: Duplication, dead code, missing type hints, naming inconsistencies
   - **Test Gaps**: Modules or behaviors with poor test coverage
   - **Configuration**: `tools_config.json` improvements, missing tool assignments
4. **Write `analysis_report.json`**: With `project_overview` (file count, pipelines, agent roles), `log_analysis` (logs found, common failure modes, average success rate), `improvement_opportunities` (categorized with ID, severity, files involved, suggested fix), and `summary`

**Parse result** (`analyzer.py:89-114`) uses the 3-tier strategy: disk file → agent messages → `_fallback_analyze()`. The fallback (`analyzer.py:117-158`) is notable for being both a fallback AND a pre-seeded improvement suggestion: it walks the project directory to count Python files and detect pipeline modules, then returns a hardcoded `improvement_opportunities` entry (SEO-001) suggesting "Expand test coverage for pipeline behavioral tests." This hardcoded suggestion served as the seed that produced the v1.8 test enhancement (175 new behavioral tests).

#### 2.3 SelfEvolutionPlannerRole: Concrete Improvement Planning

The planner (`self_evolution/planner.py:13-143`) reads the analysis report and produces an `implementation_plan.json` with per-improvement specifications. Unlike FeaturePlannerRole which plans abstract changes (files_to_create, files_to_modify), SelfEvolutionPlannerRole plans **concrete code modifications**:

- `action`: One of `modify`, `create`, or `delete`—explicitly declaring the type of code change
- `files_to_modify`: Each entry includes `path`, `section` (line range or function name), `current_code_snippet` (what the code looks like now), `new_code` (what it should become), and `reasoning` (why this change improves UMAF)
- `files_to_create`: Each entry includes `path`, `description`, and `content_outline`
- `verification`: How to verify—typically "Run `pytest test/ -q`"
- `estimated_impact`: 1-2 sentence summary of expected improvements
- `risk_assessment`: Any risks associated with changes

The `current_code_snippet` and `new_code` fields make this plan executable: the coder can implement changes by direct text substitution rather than interpreting abstract instructions. This is more detailed than FeaturePlannerRole's plan (which has `section` and `change` but not actual code snippets).

#### 2.4 SelfEvolutionCoderRole: In-Place Source Modification with Change Detection

The coder (`self_evolution/coder.py:15-124`) is the most architecturally significant SelfEvolution role. Its `build_task()` (`coder.py:24-81`) is structured as a modify-then-create ordered procedure with explicit UMAF conventions:

- **Python >= 3.11 syntax**: `X | None`, not `Optional[X]`
- **Import style**: `from __future__ import annotations` at the top
- **Documentation**: No multi-line docstrings—one short line max
- **Comments**: Only when the WHY is non-obvious
- **Testing**: Run `cd {project_dir} && python -m pytest test/ -x -q 2>&1 | tail -5` after each change

**Change detection** (`coder.py:83-123`) is the coder's most innovative parse_result method. It uses a **three-tier detection strategy**:

1. **Git diff** (`coder.py:88-94`): Runs `git -C {project_dir} diff --name-only` to detect modified tracked files. This catches line-level changes to existing files with 5-second timeout.

2. **Untracked files** (`coder.py:97-105`): Runs `git -C {project_dir} ls-files --others --exclude-standard` to detect new files not yet tracked by git. This catches newly created files.

3. **mtime fallback** (`coder.py:108-119`): If git commands fail (e.g., not a git repo, timeout), walks the project directory and checks `os.path.getmtime()` for files modified within the last 60 seconds. Directory exclusions (`.git`, `__pycache__`, `node_modules`, `.venv`) prevent false positives from cache files.

This three-tier approach is more robust than FeaturePipeline's file-existence checking: git-based detection captures actual code changes (not just file presence), and the mtime fallback handles edge cases where git is unavailable. The timeout protection (5s per git command, `subprocess.TimeoutExpired` catching) prevents hanging if the repository is in an unusual state.

The coder receives `review_issues` when in a retry iteration, rendered inline as "REVIEW ISSUES TO FIX," forcing the coder to address specific reviewer concerns before re-running verification.

#### 2.5 SelfEvolutionReviewerRole: Test-Driven Verification

The reviewer (`self_evolution/reviewer.py:15-116`) is the gatekeeper for self-modification safety. Its `build_task()` (`reviewer.py:24-65`) mandates a 4-step verification process:

1. **Run the test suite**: Execute `python -m pytest test/ -q`
2. **Check test results**: All tests must pass (0 failures), no new failures vs baseline
3. **Review code quality**: Sample changed files, check Python >= 3.11 syntax, unused imports, dead code, error handling
4. **Output verdict**: `REVIEW_PASSED` or `REVIEW_FAILED`

The prompt emphasizes: "Be thorough but fair. If the tests pass and the code follows conventions, this is a PASS." This instructs the reviewer to not be overly strict—the primary gate is test passing, with code quality as a secondary concern.

**Parse result** (`reviewer.py:67-116`) extends `scan_review_verdict()` with two additional capabilities:

- **Test result extraction** (`reviewer.py:80-92`): Uses regex `r"(\d+\s+passed.*?)(?:\s+in\s+[\d.]+s)?"` to parse pytest output from agent messages, capturing the "N passed" portion. This provides concrete evidence of test outcomes rather than relying solely on the agent's self-declared verdict.

- **Early failure detection** (`reviewer.py:70-72`): If `result.success` is False (agent crashed or timed out), immediately returns `review_passed=False` with issue "Agent did not complete successfully." This catches cases where the reviewer agent itself fails before producing a verdict.

- **Issue extraction** (`reviewer.py:95-109`): Similar to FeatureReviewerRole, extracts bullet-point issues from the `REVIEW_FAILED` message with a fallback to "Review failed—review message for details."

#### 2.6 SelfEvolutionWriterRole: Evolution Documentation

The writer (`self_evolution/writer.py:14-91`) produces `evolution_report.md` with 5 sections: Summary, Changes, Test Results, Impact, and Next Steps. It receives `changed_files`, `review_passed`, and `test_results` via context. The `_fallback_report()` (`writer.py:63-91`) generates a self-aware template noting "_No automated changes were made in this cycle_" when no files were changed, and includes "Next Steps" suggesting specific improvement goals.

### 3. Comparative Analysis: FeaturePipeline vs. SelfEvolutionPipeline vs. CoderPipeline/CoderPPPipeline

#### 3.1 Brownfield vs. Greenfield vs. Self-Modification

| Dimension | CoderPipeline | CoderPPPipeline | FeaturePipeline | SelfEvolutionPipeline |
|-----------|--------------|-----------------|-----------------|----------------------|
| **Development type** | Greenfield (single-file) | Greenfield (multi-file) | Brownfield (existing project) | Self-modification |
| **Entry point** | Coder | Head (decomposer) | Scanner | Analyzer |
| **Project awareness** | None | None (reads spec files) | Full (project_context.json) | Full (own codebase) |
| **Modification target** | New files only | New files only | New + existing files | Own source files |
| **Convention adherence** | Generic | Generic | Project-specific | Framework-specific |
| **Max review cycles** | 5 | 5 versions | 5 | 3 |
| **Change detection** | os.walk() | File existence + size | Plan-based existence check | git diff + mtime |
| **Safety mechanism** | Working directory | Assembly directory | Project directory (read-only context) | git checkout -- . |
| **Test execution** | Implicit in coder | Implicit in reviewer | Explicit in coder + reviewer | Explicit in reviewer (pytest) |

#### 3.2 The REVIEW_PASSED/REVIEW_FAILED Pattern: Consistency and Extensions

All four pipelines share the `scan_review_verdict()` function (`utils.py:71-85`), but each extends it differently:

- **CoderPipeline**: Direct use—`review_passed = scan_review_verdict(result.messages) or False`
- **CoderPPPipeline**: Two-tier verification—`scan_review_verdict()` on messages, then `review.md` file as authoritative override
- **FeaturePipeline**: Uses `scan_review_verdict()` in `FeatureReviewerRole.parse_result()`, then extracts structured issues from bullet points (capped at 20)
- **SelfEvolutionPipeline**: Uses `scan_review_verdict()` in `SelfEvolutionReviewerRole.parse_result()`, augmented with regex-based test result extraction and early failure detection via `result.success` check

This pattern of reusing a shared utility while adding pipeline-specific extensions is characteristic of UMAF's architecture: `utils.py` provides the common abstraction; each pipeline's `parse_result()` adds domain-specific enrichment.

#### 3.3 Tool Assignment Profiles

| Pipeline Role | Tools | Count | Notable Restrictions |
|--------------|-------|-------|---------------------|
| Feature Scanner | read_file, write_file, run_command | 3 | Needs `run_command` for `find`/`ls` |
| Feature Planner | read_file, write_file | 2 | No `run_command`—planning is read-only |
| Feature Coder | read_file, write_file, write_lines, run_command | 4 | Needs `write_lines` for code; no `web_search` (project is self-contained) |
| Feature Reviewer | read_file, run_command | 2 | Most restricted—read and test only; no file writing |
| Feature Writer | write_file | 1 | Most restricted overall—only writes report |
| SE Analyzer | read_file, write_file, run_command | 3 | Needs `run_command` for `find` and test running |
| SE Planner | read_file, write_file | 2 | Same profile as Feature Planner |
| SE Coder | read_file, write_file, write_lines, run_command | 4 | Same profile as Feature Coder |
| SE Reviewer | read_file, write_file, run_command | 3 | Can write files (unlike Feature Reviewer)—for in-place fixes |
| SE Writer | write_file | 1 | Same profile as Feature Writer |

Key difference: SelfEvolution reviewer has `write_file` access while Feature reviewer does not. This reflects the different trust models—SelfEvolution operates on UMAF's own code where the reviewer may need to fix minor issues directly, while FeaturePipeline operates on user projects where the reviewer should never modify code.

### 4. Safety Architecture for Self-Modification

SelfEvolutionPipeline's safety design is multilayered:

1. **Git reversibility**: The primary safety guarantee—all changes are revertible with `git checkout -- .`. This is stated in the pipeline docstring (`self_evolution.py:12-13`), the CLAUDE.md documentation, and is inherent in operating on the current git branch.

2. **Reduced iteration limit**: `MAX_ITERATIONS = 3` (vs. 5 for CoderPipeline and FeaturePipeline). Fewer self-modification cycles reduce the probability of introducing undetected bugs.

3. **Test suite as gate**: The reviewer runs the full test suite (`pytest test/`) and requires all tests to pass. This provides a fast, deterministic regression check. The 379-test suite covers all 7 pipelines.

4. **Change traceability**: The coder's `parse_result()` captures exact changed file lists via git diff. The writer documents all changes in `evolution_report.md`. This creates an audit trail for every self-modification.

5. **Fallback at every level**: Every role has a deterministic fallback that prevents pipeline crashes. The analyzer's fallback still identifies improvement opportunities; the planner's fallback still produces a plan; the coder's fallback (mtime scanning) still detects changes even without git.

6. **No new dependencies**: The coder prompt explicitly prohibits introducing new dependencies (`coder.py:78`). This ensures self-modifications don't expand the framework's external attack surface.

7. **Backward compatibility**: The coder prompt enforces "Keep changes backward-compatible" (`coder.py:79`). This prevents self-modifications from breaking existing pipeline interfaces.

### 5. Comparison with Related Work

#### 5.1 FeaturePipeline and Brownfield Development Systems

FeaturePipeline's scanner→planner→coder→reviewer topology maps to several emerging systems:

- **RepoAI** (ScienceDirect, 2025): Uses RAG-based retrieval to ground multi-agent code refactoring in repository context. FeaturePipeline's `project_context.json` serves the same purpose as RepoAI's retrieval index, but is generated once at pipeline start rather than queried dynamically—trading flexibility for deterministic, auditable context.

- **WIRL** (ASE 2025): An IDE-integrated agent for context-aware code adaptation that achieves 91.7% exact-match precision. FeaturePipeline's "modify file" workflow (read existing → make surgical change → write complete file) mirrors WIRL's code wiring approach but operates at the pipeline level rather than IDE-embedded.

- **Yellhorn MCP** / **plnr**: These tools generate workplans from full codebase context, similar to FeaturePlannerRole. The key difference is that FeaturePlannerRole's plan is machine-readable JSON consumed by downstream agents, while Yellhorn's workplans are GitHub issues for human developers.

#### 5.2 SelfEvolutionPipeline and Recursive Self-Improvement

SelfEvolutionPipeline occupies a pragmatic middle ground in the self-improvement landscape:

- **vs. Gödel Agent** (ACL 2025): Gödel Agent implements recursive self-modification through prompting alone, allowing the LLM to dynamically modify its own logic. SelfEvolutionPipeline is more constrained—self-modification follows a structured analyze→plan→implement→review pipeline with explicit safety gates, making it less flexible but more predictable and auditable.

- **vs. GBase** (2026): GBase provides full recursive self-improvement with quality gates, rollback, and auto-recovery. SelfEvolutionPipeline's `git checkout -- .` provides simpler rollback semantics without requiring dedicated infrastructure. GBase's `full_evolution_cycle()` with stability audits is more sophisticated than SelfEvolutionPipeline's 3-iteration loop.

- **vs. Socratic-SWE** (June 2026): Distills solving traces into structured "agent skills" for closed-loop coding agent improvement, achieving 50.40% on SWE-bench Verified. SelfEvolutionPipeline's analyzer→planner→coder loop is architecturally similar but operates on UMAF's own codebase rather than SWE-bench tasks.

- **vs. Autogenesis** (2026): Uses protocol layers (RSPL + SEPL) with propose→assess→commit improvements and rollback. SelfEvolutionPipeline's flow (analyze → plan → implement → review → write) implements a simplified version of this protocol without the formal resource/evolution layer abstraction.

#### 5.3 Token Scanning vs. Structured Verification

UMAF's `REVIEW_PASSED`/`REVIEW_FAILED` token scanning is a **linguistic self-assessment** approach. The broader ecosystem shows a trend toward more structured verification:

- **bop verification architecture** (GitHub): 5-stage pipeline with confidence thresholds (≥75% to block), explicit falsification attempts, and classification taxonomy. UMAF's binary pass/fail with no confidence scoring is simpler but more brittle.

- **ya-code-review**: 5-phase methodology with premise→execution tracing→claims→falsification→findings. Each claim must survive aggressive auto-invalidation. UMAF's approach trusts the reviewer agent's self-assessment without adversarial falsification.

- **PR-AF** (AgentField): Uses AST extraction to verify caller/import contexts programmatically, cross-correlating isolated findings into compound vulnerability syntheses. UMAF's reviewer reads files and runs tests but does not perform AST-level verification.

The key trade-off is between **simplicity** (UMAF's regex-based token scan works with any LLM backend and any codebase) and **reliability** (structured verification with falsification gates catches false positives/negatives that token scanning misses). UMAF's approach is appropriate for development-time code review where occasional false verdicts are acceptable; production-grade CI/CD integration would benefit from the structured verification approaches.

## Important Papers & References

- **Madaan, A., et al. "Self-Refine: Iterative Refinement with Self-Feedback" (NeurIPS 2023)** — Foundational paper establishing the generate→critique→refine loop. FeaturePipeline and SelfEvolutionPipeline's coder↔reviewer cycles are multi-agent instantiations of Self-Refine, with the key difference that UMAF uses separate agents with different tool profiles rather than a single model in different prompt modes. URL: https://arxiv.org/abs/2303.17651

- **Yin, X., Wang, X., Pan, L., et al. "Gödel Agent: A Self-Referential Agent Framework for Recursively Self-Improvement" (ACL 2025)** — Introduces the concept of LLM agents that recursively modify their own logic via prompting alone, inspired by Schmidhuber's Gödel Machine. SelfEvolutionPipeline implements a more constrained version of this idea with explicit pipeline stages, safety gates, and git-based reversibility. URL: https://aclanthology.org/2025.acl-long.1354/

- **Shinn, N., et al. "Reflexion: Language Agents with Verbal Reinforcement Learning" (NeurIPS 2023)** — Introduces verbal reinforcement where agents learn from prior failures through self-reflection stored in episodic memory. SelfEvolutionPipeline's analyzer examining agent logs for failure patterns implements a batch version of Reflexion's online learning. URL: https://arxiv.org/abs/2303.11366

- **Khanzadeh, M. "AgentMesh: A Cooperative Multi-Agent Generative AI Framework for Software Development Automation" (arXiv:2507.19902, Jul 2025)** — Planner→Coder→Debugger→Reviewer pipeline with blackboard-style shared state. FeaturePipeline's scanner→planner→coder→reviewer topology mirrors this pattern, with `project_context.json` serving as the shared blackboard. URL: https://ar5iv.labs.arxiv.org/html/2507.19902

- **"Towards Realistic Project-Level Code Generation via Multi-Agent Collaboration and Semantic Architecture Modeling" (arXiv:2511.03404, Nov 2025)** — Proposes ProjectGen with Semantic Software Architecture Tree for project-aware code generation. Achieves 57% improvement on DevBench. FeaturePipeline's `project_context.json` with file_manifest provides a simpler, flat alternative to SSAT's hierarchical tree structure. URL: https://arxiv.org/abs/2511.03404

- **Liu, X., et al. "DocAgent: Multi-Agent Collaborative Topological Code Processing" (ACL 2025)** — Uses topological sort on code dependency DAGs. FeaturePlannerRole's requirement for "no circular dependencies, all imports resolve" implements lightweight topological validation, while DocAgent performs full dependency graph analysis. URL: https://aclanthology.org/2025.acl-long.XX/

- **"Wired for Reuse: Automating Context-Aware Code Adaptation in IDEs via LLM-Based Agent" (ASE 2025)** — The WIRL system achieves 91.7% exact-match precision for code adaptation using RAG-based variable substitution. FeatureCoderRole's "read existing file before modifying" workflow is a manual analogue to WIRL's automated code wiring. URL: https://arxiv.org/abs/2507.01315

- **Kar, I., et al. "Towards AGI: A Pragmatic Approach Towards Self-Evolving Agent" (arXiv:2601.11658, Jan 2026)** — Hierarchical multi-agent framework integrating Base LLM, SLM agent, Code-Generation LLM, and Teacher-LLM with curriculum learning and genetic algorithm evolution. SelfEvolutionPipeline's simpler 5-node linear graph is a lightweight alternative to this hierarchical architecture. URL: https://export.arxiv.org/abs/2601.11658

- **Lin, M., et al. "Position: Agentic Evolution is the Path to Evolving LLMs" (arXiv:2602.00359, Feb 2026)** — Argues the evolution-scaling hypothesis: adaptation capacity scales with compute allocated to evolution. SelfEvolutionPipeline's 3-iteration budget represents a fixed compute allocation; the paper suggests this should scale dynamically. URL: https://browse-export.arxiv.org/abs/2602.00359

- **Xiao, C., Jiao, Z., et al. "Socratic-SWE: Self-Evolving Coding Agents via Trace-Derived Agent Skills" (arXiv:2606.07412, Jun 2026)** — Distills solving traces into structured agent skills for closed-loop coding agent improvement, reaching 50.40% on SWE-bench Verified. SelfEvolutionPipeline's analyzer analyzing agent logs is architecturally similar to Socratic-SWE's trace distillation. URL: https://export.arxiv.org/abs/2606.07412

- **Zhang, G., Ren, H., et al. "MemEvolve: Meta-Evolution of Agent Memory Systems" (ICML 2026)** — Jointly evolves experiential knowledge AND memory architecture, with EvolveLab providing 12 modular memory systems. UMAF's `CheckpointManager` and `agent_log/` directory provide a simpler memory substrate compared to EvolveLab's encode→store→retrieve→manage design. URL: https://icml.cc/virtual/2026/poster/61379

- **"When Parallelism Pays Off: Cohesion-Aware Task Partitioning for Multi-Agent Coding" (arXiv:2606.00953, 2026)** — Formalizes multi-agent orchestration as graph partitioning; cohesion-aware partitioning achieves 14% pass rate improvement, 2.1× speedup, 35% cost reduction. FeaturePipieline's sequential scanner→planner→coder topology could benefit from cohesion-aware decomposition when the feature spans multiple independent modules. URL: https://export.arxiv.org/abs/2606.00953

- **Pan, Z., Zhang, Y., and Liu, Y. "CodeCoR: Code Generation with Self-Reflective Multi-Agent Collaboration" (arXiv:2501.07811, Jan 2025)** — Four-agent framework achieving 77.8% Pass@1 on HumanEval. FeaturePipeline's 5-role topology (scanner, planner, coder, reviewer, writer) is more specialized than CodeCoR's 4-role topology but targets real-world project modification rather than competitive programming benchmarks. URL: https://arxiv.org/abs/2501.07811

- **Yao, S., et al. "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)** — The Thought→Action→Observation loop underlying every UMAF agent's tool-call execution. Both FeaturePipeline and SelfEvolutionPipeline's agents use ReAct within the broader pipeline graph—a second-order ReAct loop (coder↔reviewer) running on top of the first-order tool-call loop. URL: https://arxiv.org/abs/2210.03629

- **"Blueprint2Code: A Multi-Agent Pipeline for Reliable Code Generation via Blueprint Planning and Repair" (Frontiers in AI, 2025)** — 4-agent pipeline achieving 96.3% on HumanEval and 88.4% on MBPP through preview→blueprint→coding→debugging stages. FeaturePipeline's scanner (preview) → planner (blueprint) → coder (coding) → reviewer (debugging) maps directly to this architecture with the addition of brownfield project awareness. URL: https://doi.org/10.3389/frai.2025.1660912

- **"Towards Reliable ML Feature Engineering via Planning in Constrained-Topology of LLM Agents" (arXiv:2601.10820, Jan 2026)** — Planner-guided multi-agent framework achieving 38-150% improvement over manual workflows, reducing feature engineering cycles from 3 weeks to 1 day. The constrained-topology approach (planner represents team environment as a graph) is a more formal version of FeaturePlannerRole's JSON-based plan structure. URL: https://arxiv.org/abs/2601.10820

- **"Reflection-Driven Control for Trustworthy Code Agents" (arXiv:2512.21354, 2025)** — Elevates self-reflection into a first-class internal loop with Plan→Reflect→Verify stages, lightweight self-checker for SAFE/UNSAFE classification, and reflective memory repository. SelfEvolutionPipeline's analyzer→planner→coder→reviewer→writer implements a multi-agent, batch version of this reflection-driven control architecture. URL: https://arxiv.org/abs/2512.21354

## Open Questions & Future Directions

1. **Scanner granularity vs. scalability**: The FeatureScannerRole's `_fallback_scanner()` uses `find | head -2000`, which misses files in projects larger than 2000 files. For large codebases (monorepos, enterprise applications), a more sophisticated approach—incremental scanning with change detection, or embedding-based semantic indexing of the codebase for targeted retrieval—would be necessary. The cased-kit library's symbol extraction and dependency analysis provides one model; WIRL's RAG-based approach provides another.

2. **Cross-pipeline self-evolution**: Currently, SelfEvolutionPipeline only analyzes and improves UMAF's source code. A more powerful approach would be for SelfEvolutionPipeline to analyze the execution traces of OTHER pipelines (e.g., Research, CoderPP, Feature) and suggest improvements not just to the framework code but to pipeline-specific prompts, tool assignments, and parameter settings. This would require extending the analyzer to read arbitrary pipeline output directories and cross-reference agent success rates.

3. **Adaptive iteration limits**: Both FeaturePipeline (max 5 cycles) and SelfEvolutionPipeline (max 3 cycles) use fixed iteration caps. Research on Self-Refine shows diminishing returns after 2-3 iterations. An adaptive termination policy based on improvement delta—stop if the reviewer's issue count decreases below a threshold between cycles—would reduce API costs. The MemoCoder approach (detecting when an error pattern has been attempted before) could further improve efficiency.

4. **Reviewer confidence scoring**: The REVIEW_PASSED/REVIEW_FAILED token scan is a binary gate with no confidence information. A reviewer might pass code with low confidence (barely acceptable) or fail it with high confidence (serious bugs). Structured confidence scoring (as in bop's verification architecture with ≥75% thresholds) would enable more nuanced routing—low-confidence passes could trigger additional review, while high-confidence failures could skip directly to targeted fixes.

5. **SelfEvolution safety verification**: While `git checkout -- .` provides reversibility, there is no mechanism to verify that self-modifications are semantically safe beyond running the existing test suite. A test that passes after a self-modification might still fail in edge cases that the modification introduced. Techniques from the self-improvement safety literature—invariant checking, behavioral fuzzing, differential testing against a reference implementation—could strengthen the safety guarantee.

6. **FeaturePipeline for polyglot projects**: The scanner's `_fallback_scanner()` detects language based on file extension counts and classifies files by extension sets (`src_exts` includes `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, etc.), but the build_task() prompt is Python-centric (mentions `pyproject.toml`, `requirements.txt`, `snake_case`, `pytest`). For JavaScript/TypeScript projects, the scanner would correctly identify the language but the planner/coder/reviewer prompts would need language-appropriate conventions (camelCase, ESLint, Jest, `package.json`). Dynamic prompt generation based on detected language would make FeaturePipeline truly polyglot.

7. **Memory persistence across self-evolution cycles**: Each SelfEvolutionPipeline run starts fresh—the analyzer scans the current codebase state but has no memory of what was improved in previous self-evolution cycles. A persistent evolution memory (storing which improvements were attempted, which succeeded, and which introduced regressions) would prevent the analyzer from re-suggesting failed improvements and enable cumulative optimization. MemEvolve's meta-evolution of memory architecture provides a formal framework for this.

8. **FeaturePipeline-CoderPP integration**: FeaturePipeline handles brownfield development for existing projects with project-awareness, while CoderPP handles multi-file greenfield generation with dependency-aware decomposition. A merged pipeline could handle the case where a feature requires both modifications to existing files AND creation of multiple interdependent new modules—currently this would require running both pipelines separately.

9. **Adversarial self-review**: SelfEvolutionPipeline's reviewer is the same LLM family that wrote the changes. Research on cross-family LLM validation (RepoAI's approach) and adversarial falsification (ya-code-review, PR-AF) suggests that using a DIFFERENT model for review than for coding catches different categories of bugs. UMAF could extend SelfEvolutionPipeline to use DeepSeek for coding and Claude for reviewing (or vice versa) to achieve cross-model verification.

10. **Quantitative self-evolution metrics**: There is no built-in measurement of whether SelfEvolutionPipeline actually improves UMAF. Metrics like: pre-evolution test pass rate vs. post-evolution, code coverage change, average agent success rate change, and review pass rate change would provide quantitative evidence of improvement. An A/B testing framework comparing pipeline performance before and after self-evolution would validate the approach.

11. **Planner plan verification**: FeaturePlannerRole validates "no circular dependencies, all imports resolve" but this is done by the LLM's reasoning, not programmatically. A static analysis pass on the implementation plan—verifying that all referenced files exist, that dependency edges form a DAG, and that interface specifications are compatible—would catch plan errors before the coder begins implementing.

12. **Reviewer specialization**: Both FeaturePipeline and SelfEvolutionPipeline use a single reviewer role that checks all dimensions (completeness, correctness, conventions, tests). Research on multi-voter consensus review (bop, PR-AF, Nexus Agents) suggests that separate specialized reviewers—a correctness reviewer, a convention reviewer, and a security reviewer—could catch more issues through division of responsibility, at the cost of 3× API calls per review cycle.

## Relevance to Main Topic

FeaturePipeline and SelfEvolutionPipeline complete UMAF's code generation capability spectrum, extending the framework from greenfield generation (CoderPipeline, CoderPPPipeline) into brownfield development and meta-cognitive self-improvement. Together, these four pipelines form a comprehensive code generation capability: CoderPipeline handles single-file tasks (entry point for simple code generation), CoderPPPipeline handles multi-file greenfield projects (decompose→implement→review→assemble), FeaturePipeline handles brownfield modification of existing projects (scanner→planner→coder→reviewer with project context grounding), and SelfEvolutionPipeline handles self-modification (analyzer→planner→coder→reviewer with git-based safety).

The architectural patterns introduced by these pipelines are significant beyond UMAF. The scanner→planner→coder→reviewer topology with a `project_context.json` shared knowledge base demonstrates a practical approach to context-grounded multi-agent development: instead of each agent independently discovering project conventions, a dedicated scanner role produces a structured, auditable context file that all downstream agents reference. This pattern—**context bootstrapping via specialized scanner agents**—is applicable to any multi-agent system that operates on pre-existing artifacts, whether codebases, document collections, or databases.

SelfEvolutionPipeline's lightweight self-improvement approach—operating on the current git branch with `git checkout -- .` reversibility and a 3-iteration safety cap—demonstrates that meaningful self-improvement does not require complex sandboxing infrastructure (Docker containers, VM isolation, protocol layers). The combination of agent log analysis for opportunity identification, test suite execution for regression detection, and git-based rollback for safety provides a practical, low-overhead self-evolution capability that could be incorporated into any LLM agent framework with a version-controlled codebase.

For operators evaluating UMAF's safety properties, the tool restriction patterns in these pipelines are instructive: FeaturePipeline's reviewer has only 2 tools (read_file, run_command)—the most restricted reviewer profile—reflecting the trust boundary of operating on user projects. SelfEvolutionPipeline's reviewer has 3 tools (read_file, write_file, run_command)—more permissive because it operates on UMAF's own code where in-place fixes are acceptable. These differentiated trust models, expressed through tool assignment, demonstrate how `tools_config.json` as single source of truth enables security policies to be encoded declaratively and enforced automatically.

For researchers studying multi-agent code generation systems, UMAF's four-pipeline taxonomy provides a useful framework for understanding capability coverage: greenfield single-file (Coder), greenfield multi-file (CoderPP), brownfield modification (Feature), and self-modification (SelfEvolution). Most academic systems (CodeCoR, Blueprint2Code, ProjectGen) target only one of these quadrants; UMAF demonstrates that a unified architecture with consistent patterns (AgentRole ABC, ToolRegistry, status-based routing, token scanning) can span all four while maintaining shared infrastructure and safety guarantees.
