# ResearchPipeline: Dependency-Ordered Workers, Version-Bump Retries, and LaTeX Report Generation

## Overview

The ResearchPipeline is the UMAF framework's most architecturally sophisticated pipeline, implementing a 4-node LangGraph workflow (`head → workers → reviewer → writer`) that transforms unstructured research topics into scored, ranked, and LaTeX-formatted research proposals. The pipeline is designed around three interlocking resilience mechanisms: **dependency-ordered parallel execution** via topological level grouping with stop-on-failure, **version-bump retry with checkpoint-based context reuse** that preserves prior reasoning across retry attempts, and **honest success reporting** where `parse_result()` verifies file existence on disk via `os.path.isfile()` rather than trusting the LLM's self-reported success. Together, these mechanisms address the fundamental reliability problem in autonomous LLM research agents: LLMs frequently claim to have written files they haven't, fail on complex multi-step tasks due to context loss, and produce outputs with interdependent dependencies that traditional flat-parallel execution cannot satisfy.

The pipeline's motivation extends beyond simple task decomposition. Research synthesis is inherently a map-reduce problem — a complex topic must be decomposed into independent sub-topics, each researched in parallel, then aggregated and scored for quality. However, unlike classical MapReduce where workers are stateless and deterministic, LLM research workers are stochastic, stateful, and prone to failure. The ResearchPipeline addresses this through a three-tier resilience architecture: (1) the head agent's decomposition scales sub-topic count 2-12 based on topic complexity with a keyword-based fallback, (2) workers execute in dependency-respecting topological levels with stop-on-failure blocking downstream dependents, backed by a version-bump retry system (max 6 versions, 5 worker retries) that preserves message context across attempts via `CheckpointManager.load_previous()`, and (3) the reviewer applies a 5-dimension scoring rubric (depth, accuracy, relevance, clarity, originality, each 1-10) that ranks outputs for the LaTeX writer, which handles 10 LaTeX special characters via `_latex_escape()` and falls back to a templated document when LLM generation fails.

What distinguishes the ResearchPipeline from simpler multi-agent orchestration patterns is its **always-forward routing** philosophy: the `researched_partial` status — indicating that some workers failed to produce output even after all retries — is still routed to the reviewer rather than aborting the pipeline. This design choice recognizes that partial research results are often more valuable than no results, and the scoring system naturally down-ranks incomplete work. The pipeline's flow dict (`decomposed → workers`, `worker_retry → workers`, `researched → reviewer`, `researched_partial → reviewer`, `reviewed → writer`, `written → END`) encodes this philosophy — the only terminal states are successful completion or three specific error conditions (`error_no_subtasks`, `error_no_reviewable`, `error_no_scored_works`), none of which involve partial worker success.

## Key Methods & Approaches

### 1. Dynamic Decomposition Algorithm: Complexity-Scaled Topic Splitting

The head node (`_head_node`, `pipeline/research.py:204-225`) orchestrates topic decomposition with a three-tier fallback chain designed to guarantee forward progress even when the LLM decomposer fails.

**Tier 1 — LLM Decomposition via `ResearchDecomposerRole`** (`research/head_agent.py:8-131`): The decomposer inherits from `BaseDecomposerRole` (`agent.py:1155`), which implements the Template Method pattern for decomposition. The concrete `ResearchDecomposerRole` overrides five template methods to specialize behavior:

- **`_role_prompt()`**: Frames the agent as a "research coordinator" with instructions to analyze topic complexity and produce 2-12 sub-topics.
- **`_sizing_guide()`**: Provides a three-tier complexity heuristic:
  - Narrow/specific topics (single method/technique) → 3-5 sub-topics
  - Moderate topics (family of techniques, one research area) → 5-8 sub-topics
  - Broad/complex topics (whole field, multi-paradigm comparison) → 8-12 sub-topics
- **`_sub_unit_requirements()`**: Mandates that sub-topics be self-contained with clear scope boundaries, include concrete research questions, cover different angles, declare dependencies on other sub-topics (by id), and be suitable for a 25-minute focused research session.
- **`_json_template()`**: Specifies the output schema `[{"id": int, "title": str, "description": str, "dependencies": []}]`.
- **`_backend_instructions()`**: For `claude_cli`, instructs the agent to "Decompose the topic into sub-topics. Output the JSON array, then write TASK_COMPLETE." For DeepSeek, adds the constraint "Output ONLY the JSON array, nothing else before or after."

The `BaseDecomposerRole.parse_result()` implements a three-tier extraction strategy shared with `CoderPPDecomposerRole`: (1) extract inline JSON array from agent messages via `extract_json_array()`, (2) fall back to reading `decomposition.json` from disk, (3) fall back to `_fallback_decompose()` for programmatic generation.

**Tier 2 — Timeout Protection**: The head node wraps the decomposition call in a `ThreadPoolExecutor` with `HEAD_TIMEOUT=300` seconds (`pipeline/research.py:213-218`). If the decomposer times out or raises an exception, execution continues to the fallback.

**Tier 3 — Programmatic Fallback** (`_fallback_decompose`, `research/head_agent.py:67-117`): When LLM decomposition fails or produces fewer than 2 sub-topics, the fallback uses keyword splitting:
1. Split the input topic on commas, "and", "vs", and semicolons to extract keyword phrases.
2. For each keyword (up to 12), create a "deep-dive" sub-topic: `"{kw}: Mechanisms, Methods, and Key Results"`.
3. Append two synthesis sub-topics: one for comparative analysis and one for open problems/emerging directions.
4. If fewer than 2 sub-topics result, add an overview sub-topic as a final safety net.
5. Cap at 8 sub-topics (via `templates[:8]`).

This three-tier design guarantees that decomposition never produces zero sub-topics — the fallback always generates at least 2 (keyword deep-dive + comparative analysis), and if the topic itself has no parseable keywords (fewer than 2 characters), the overview sub-topic is added. This is architecturally critical because an empty `sub_tasks` list would trigger the `error_no_subtasks` terminal error, aborting the entire pipeline.

**Comparison with related work**: The Federation of Agents (FoA) framework (CDS CERN, 2025) uses a similar "score compatible agents, collaboratively build a DAG" approach but with a 6-phase pipeline including cluster formation and intra-cluster consensus — far more complex than UMAF's single-agent decomposer. D³MAS (arXiv:2510.10585, 2025) uses a 3-layer heterogeneous graph (Decompose, Deduce, Distribute) that explicitly targets the 47.3% average knowledge duplication problem, a concern UMAF addresses through the simpler mechanism of deduplication via MD5 fingerprinting at the worker output merge stage. A-MapReduce (Chen et al., 2025) introduces a task matrix + template approach (`Θ_q = (M_q, P_q, B_q)`) that is more structured than UMAF's template but also more rigid — UMAF's LLM-driven decomposition can adapt to arbitrary topics without pre-defined templates.

### 2. Dependency-Ordered Worker Execution via `_topological_levels()`

The worker node (`_workers_node`, `pipeline/research.py:227-378`) is the most complex node in the pipeline, implementing a state machine that handles fresh execution, retry-only-failed, and max-retries-exceeded final-attempt modes.

**Topological Level Grouping** (`pipeline/base.py:195-303`): The `BasePipeline._topological_levels()` static method transforms a flat list of sub-tasks with `dependencies` declarations into a list of execution levels, where all tasks within a level are independent and can run in parallel, and levels execute sequentially. The algorithm:

1. **Quick path**: If no task declares dependencies, return a single level containing all tasks — degenerate to flat parallelism.
2. **Key resolution**: Dependencies can reference tasks by `id` (int), `module_name` (str), or `dict` with either field. A unified `_task_key()` function normalizes to string keys, and `_dep_keys()` resolves dependency references to the same key space.
3. **Level construction loop**: While tasks remain, identify all tasks whose dependencies are disjoint from the remaining set (all upstream tasks completed) and group them as the current level.
4. **Cycle detection and breaking**: If no task is eligible but tasks remain, a dependency cycle exists. The algorithm:
   - Logs a warning with the cycle participants.
   - Iteratively removes edges: picks the task with the most intra-cycle dependencies, removes its **last** dependency (reasoning that earlier dependencies are more likely to be foundational/load-bearing).
   - Re-checks eligibility after each edge removal, up to `len(cycle_tasks) * 2` attempts.
   - If the cycle persists after all attempts, falls back to running all cycle tasks in parallel.
5. **Returns**: `list[list[dict]]` where inner lists are levels.

**Dependency Injection in `_run_workers_with_deps()`** (`pipeline/base.py:412-501`): This static method consumes the topological levels and executes agents with context passing:

1. Calls `_topological_levels()` to obtain levels; if only one level, delegates directly to `_run_parallel_agents()`.
2. For multi-level execution, maintains a `completed` dict mapping task identifiers to their output dicts.
3. Before each level executes, injects `_dependency_outputs` into tasks that declare dependencies by resolving dependency references against `completed`.
4. After each level completes, registers outputs by both `sub_task_id` (int) and `module_name` (str) — dual-key registration supporting both Research pipeline (int IDs) and CoderPP pipeline (string module names).
5. **Stop-on-failure**: If any task in a level fails and there are downstream levels remaining, the execution breaks out of the level loop, deferring downstream dependents for retry. This prevents cascading failures where dependent tasks run on missing or corrupted upstream outputs.

In the ResearchPipeline's `_workers_node`, the stop-on-failure behavior integrates with the version-bump retry system: when `_run_workers_with_deps()` breaks early due to failures, the returned `total_failed > 0` triggers the `worker_retry` status, which causes the LangGraph router to loop back to the workers node for a retry attempt with the next version.

**Comparison with related work**: UMAF's dependency-aware execution is architecturally similar to D³MAS's "Deduce layer" (which uses dependency edges for reasoning reuse) but simpler — UMAF passes completed output file paths as context, while D³MAS uses a heterogeneous graph with knowledge nodes. The Joint Task Graph Generation approach (TDCommons, Oct 2025) uses a "joint planning phase where agents propose and merge subtasks together," which is more collaborative but also more communication-intensive than UMAF's single-agent decomposition.

### 3. Version-Bump Retry System with Checkpoint-Based Context Reuse

The version-bump retry system is the ResearchPipeline's most distinctive resilience mechanism, enabling failed workers to retry with full context from their previous attempt while getting a fresh step budget.

**State machine in `_workers_node`** (`pipeline/research.py:227-378`): The node implements three execution modes:

**Mode A — Fresh Start** (`current_status != "worker_retry"`):
- `new_version = version` (typically 1)
- `tasks_to_run = list(sub_tasks)` (all tasks)
- `kept_outputs = []` (no preserved outputs)
- `worker_retry_count = 0`

**Mode B — Retry Only Failed Workers** (`current_status == "worker_retry"` and `worker_retry_count < RESEARCH_MAX_WORKER_RETRIES`):
- `failed = [st for st in sub_tasks if st["id"] not in id_to_output]` — identifies workers that produced no output file.
- `new_version = version + 1` — bumps the version for checkpoint-based context reuse.
- `tasks_to_run = failed` — only re-runs failed workers.
- `kept_outputs = list(id_to_output.values())` — preserves previously successful outputs.
- `worker_retry_count += 1`

**Mode C — Max Retries Exceeded** (`worker_retry_count >= RESEARCH_MAX_WORKER_RETRIES`):
- Prints a diagnostic message showing how many workers have output.
- Makes one final attempt with remaining workers via `_run_workers_with_deps()` at `final_version = version + 1`.
- Any workers still without output get placeholder entries with `summary: "Worker did not produce output after all retries."`
- Sets status to `researched_partial` — the pipeline moves forward with whatever was produced.

**CheckpointManager context reuse** (`agent.py:88-97, 570-589`): The `CheckpointManager.load_previous(current_version)` method is the core mechanism enabling retry with context:

1. When `version=2`, `load_previous(2)` finds the highest version < 2 (version 1) and loads its checkpoint.
2. The loaded checkpoint contains the complete message history from the previous attempt — every system prompt, human message, AI response, and tool result.
3. `BaseAgent._run_deepseek()` (for DeepSeek backend) at `agent.py:570-589`:
   - Loads the previous version's messages via `load_previous()`.
   - Resets `iterations` to 0 — providing a fresh step budget (`max_steps=40` for workers).
   - Inserts a context injection message: "This is version {V} retry. Review what went wrong in the previous version and improve."
   - Ensures the system prompt remains at position [0].
4. The agent now has: (a) complete reasoning history from the failed attempt, (b) awareness that this is a retry, and (c) a fresh budget of 40 steps — it can learn from what went wrong without being constrained by the previous attempt's exhausted step count.

**Constants** (`pipeline/research.py:24-25`):
- `RESEARCH_MAX_VERSIONS = 6`: Maximum number of version increments across all workers. When `new_version >= 6`, the retry loop exits even if workers are still failing.
- `RESEARCH_MAX_WORKER_RETRIES = 5`: Maximum number of retry cycles. This is separate from `RESEARCH_MAX_VERSIONS` — version counts across all workers while retry counts per-cycle.
- `WORKER_TIMEOUT = 900`: Individual worker timeout in seconds (15 minutes). Enforced at both the `_run_parallel_agents` level (via `future.result(timeout=0)`) and within `BaseAgent._run_claude_cli()` (via `threading.Timer` with 600s timeout).
- `HEAD_TIMEOUT = 300`: Head agent timeout (5 minutes).

**Output file verification** (`pipeline/research.py:329-334`): After `_run_workers_with_deps()` returns, the worker node verifies that every reported output file actually exists on disk with non-zero size:
```python
for out in outputs:
    of = out.get("output_file", "")
    if of:
        fp = os.path.join(working_dir, of)
        if not os.path.exists(fp) or os.path.getsize(fp) == 0:
            out["output_file"] = ""
```
This post-hoc verification closes a gap where `parse_result()` might report success but the file is empty or missing — a common LLM hallucination pattern.

**Duplicate detection** (`pipeline/research.py:340-354`): After merging preserved outputs with new outputs, MD5 fingerprinting of worker summaries (first 200 characters, whitespace-normalized) detects duplicate outputs and marks them as "Skipped: duplicate output." This addresses the problem where multiple workers independently produce identical or near-identical summaries.

**Comparison with related work**: UMAF's version-bump retry is architecturally closest to Conductor's (Netflix OSS) durable execution model, where "every step is persisted: LLM calls, tool calls, human approvals, loop state." The key difference is that Conductor provides **guaranteed completion** through automatic resume and duplicate prevention, while UMAF provides **best-effort retry** with explicit limits — it will stop after 6 versions and move forward with partial results. The Diagrid analysis (2026) of 5 agent frameworks found that "Every framework saves state. None of them guarantee completion. Checkpointing is a storage operation, not a reliability guarantee." UMAF's version-bump retry sits at an intermediate point: it provides more reliability than LangGraph's basic checkpointing (which requires manual resume) but less than Temporal/Restate's durable execution guarantees.

### 4. Honest Success Reporting via `parse_result()` with `os.path.isfile()`

The `ResearchWorkerRole.parse_result()` method (`research/worker_agent.py:33-57`) implements a critical piece of the pipeline's reliability architecture: it verifies that the output file actually exists on disk before reporting success.

```python
def parse_result(self, result: AgentResult, working_dir: str,
                 sub_task: dict | None = None, output_file: str = "", **context) -> dict:
    actual_file = output_file if (
        output_file and os.path.isfile(os.path.join(working_dir, output_file))
    ) else ""
    return {
        "sub_task_id": sub_task["id"],
        "title": sub_task["title"],
        "output_file": actual_file,  # Empty string if file doesn't exist
        "summary": summary,
    }
```

The design is deliberately binary: a worker either has an output file (success) or doesn't (failure). There is no intermediate "partial success" state at the individual worker level. The empty string `output_file` propagates through the pipeline:
- In `_workers_node`, workers with empty `output_file` are classified as failed and trigger retry.
- In `_run_workers_with_deps()`, the `_dependency_outputs` injection only passes outputs with non-empty `output_file` to dependent tasks — a task that depends on a failed worker receives an empty dependency list and must research independently.
- In `_reviewer_node`, only workers with `output_file` or non-timeout/non-skipped summaries are considered reviewable.

This design was introduced in v1.4 as the "Honest `parse_result`" fix (CLAUDE.md v1.4): "ResearchWorkerRole.parse_result() checks `os.path.isfile()`" — replacing the previous behavior where workers could claim success based solely on the LLM's self-reported TASK_COMPLETE, even if no file was actually written.

### 5. Abstract and Summary Extraction

The summary extraction in `parse_result()` (`research/worker_agent.py:36-43`) scans messages in reverse order to find the last substantive AIMessage:

```python
for msg in reversed(result.messages):
    if type(msg).__name__ != "AIMessage":
        continue
    content = msg.content if hasattr(msg, "content") else str(msg)
    if len(content) > 100:
        summary = content[:500] + "..." if len(content) > 500 else content
        break
```

This approach prioritizes the **last** substantive response (the agent's final synthesis) over earlier intermediate tool-call outputs. The 100-character minimum filters out brief acknowledgments like "TASK_COMPLETE" or short error messages. The 500-character cap prevents overly verbose summaries from bloating downstream prompts (reviewer, writer).

### 6. The 5-Dimension Reviewer Scoring System

The reviewer node (`_reviewer_node`, `pipeline/research.py:381-409`) implements a quality assurance gate with fallback ranking.

**Reviewer Agent** (`research/reviewer_agent.py:9-72`): The `ResearchReviewerRole`:
- Receives the list of worker outputs with file paths.
- Is instructed to read each file, score on 5 dimensions (depth, accuracy, relevance, clarity, originality, each 1-10), calculate total_score (max 50), rank by total_score descending, and write `scoring_report.json`.
- `parse_result()` (`research/reviewer_agent.py:99-125`) implements a two-tier extraction:
  1. **Primary**: Read `scoring_report.json` from disk, validate it's a non-empty list, sort by total_score.
  2. **Fallback**: Search agent messages in reverse for JSON containing `"sub_task_id"`, parse and sort.

**Pipeline-Level Fallback** (`pipeline/research.py:396-408`): If the reviewer agent fails entirely (returns empty list), the `_reviewer_node` implements its own programmatic fallback that:
1. Filters to workers with output files that exist on disk.
2. Ranks by: (a) file existence on disk, (b) summary length (longer summaries proxy for more thorough research).
3. Assigns uniform neutral scores (5/10 on all dimensions, total_score=25) with justification "Auto-ranked (reviewer was unable to score)."

This three-tier scoring system (LLM reviewer → message parsing → pipeline-level programmatic ranking) ensures the pipeline always produces scored outputs, even if the LLM reviewer fails entirely.

**Reviewable filtering** (`pipeline/research.py:384-387`): The reviewer excludes workers whose summary contains "timed out" or "skipped" — these workers produced no meaningful content and would waste reviewer context.

### 7. LaTeX Report Generation: Sectioned Output with Fallback Template

The writer node (`_writer_node`, `pipeline/research.py:412-432`) generates a publication-quality LaTeX document using a sectioned output strategy to avoid the monolithic write_file truncation problem.

**Writer Agent** (`research/writer.py:53-138`): The `WriterRole`:
- `max_steps=40` — the highest step budget in the pipeline (matching workers), reflecting the complexity of generating syntactically correct LaTeX.
- **Sectioned output strategy**: The writer prompt (`build_task()`, lines 77-126) explicitly instructs the agent to write each research work as a **separate** `.tex` section file (`section_01_title.tex`, `section_02_title.tex`, etc.) and a main file (`research_proposal.tex`) with `\input{}` commands. This strategy exists because the complete LaTeX document with all sections often exceeds the LLM's single `write_file` token budget — the JSON tool-call escaping of a multi-thousand-line LaTeX document is prohibitively error-prone for the DeepSeek backend.
- **parse_result()** (`research/writer.py:128-138`): Verifies the main `.tex` file exists, has >200 bytes (non-trivial content), and contains `\input{` or `\include{` commands. If any check fails, falls back to `_fallback_latex()`.

**LaTeX Escape Function** (`research/writer.py:181-185`): The `_latex_escape()` function handles all 10 LaTeX special characters:
```python
_LATEX_ESCAPE_MAP = {
    "\\": r"\textbackslash ",
    "~": r"\textasciitilde ",
    "^": r"\textasciicircum ",
}

def _latex_escape(text: str) -> str:
    for char in ("\\", "{", "}", "$", "&", "#", "_", "%", "~", "^"):
        text = text.replace(char, _LATEX_ESCAPE_MAP.get(char, "\\" + char))
    return text
```
The escape order is critical: `\` is escaped first (converting `\` → `\textbackslash `) to prevent double-escaping of the other characters that receive `\` + char replacements (e.g., `{` → `\{`). For `~` and `^`, which require special LaTeX text-mode commands rather than simple backslash prefixes, a dedicated map provides the correct LaTeX commands.

**Fallback LaTeX Template** (`research/writer.py:188-264`): When the LLM writer fails, `_fallback_latex()` generates a complete document using a pre-defined template (`_LATEX_TEMPLATE`, lines 7-50) with:
- Full preamble (`\documentclass[11pt,a4paper]{article}`, `hyperref`, `graphicx`, `amsmath`, `enumitem`, `titlesec`, `fancyhdr`, `cite` packages).
- Auto-generated abstract summarizing the research pipeline's output.
- Per-research-work sections with escaped titles and score/rank headers.
- A comprehensive scoring table using `tabular` with all 5 dimensions + total scores.
- Auto-generated `thebibliography` entries with `\bibitem` for each work.
- Three placeholders (`__CONTENT_PLACEHOLDER__`, `__SCORES_PLACEHOLDER__`, `__BIB_PLACEHOLDER__`) replaced via string substitution.

The template ensures that even when LLM LaTeX generation fails, the pipeline produces a compilable document with all research outputs. The content is basic (each work gets a `\section` with score header and a reference to the accompanying `.md` file), but the structure is complete and the document is technically valid LaTeX.

**Checkpoint merge on completion** (`pipeline/research.py:422-431`): After the writer completes, the pipeline merges all agent checkpoints into `_merged.json` files for: (1) every worker, (2) the head decomposer, and (3) the reviewer. This provides a consolidated audit trail for the entire pipeline run, enabling post-hoc debugging and analysis.

**Comparison with related work**: UMAF's LaTeX generation approach is simpler than dedicated paper-writing systems like PaperOrchestrator (7-stage pipeline with length-adaptive control, consistency polishing, and BibTeX reference generation — achieving 90% compilation rate and BERTScore F1 0.674) or Agent Laboratory (3-phase workflow with integrated arXiv/HuggingFace/LaTeX tools — 5,600+ GitHub stars). However, UMAF's sectioned output strategy and guaranteed fallback template address the same core problem (LLMs produce syntactically incomplete LaTeX) through different means — while PaperOrchestrator uses iterative refinement, UMAF guarantees a valid document via template fallback. DeepAgents PrintShop's quality gate system (Content ≥80, LaTeX ≥85, Overall ≥80) represents a more rigorous quality assurance approach, while UMAF relies on the reviewer's scoring and the writer's parse_result verification.

### 8. Flow Dict Routing: Always-Forward State Machine

The flow dict (`pipeline/research.py:434`) defines the complete state machine:

```python
flow = {
    "decomposed": "workers",
    "worker_retry": "workers",
    "researched": "reviewer",
    "researched_partial": "reviewer",
    "reviewed": "writer",
    "written": END,
}
terminal = {"error_no_subtasks", "error_no_reviewable", "error_no_scored_works"}
```

The routing is implemented via `BasePipeline._status_router()` (`pipeline/base.py:504-514`), which builds a LangGraph conditional edge function:
```python
def router(state: dict) -> Literal["__end__"] | str:
    status = state.get("status", "")
    if status in terminal:
        return END
    if status in flow_map:
        return flow_map[status]
    return END
```

Key routing properties:
- **`worker_retry → workers`**: The self-loop that implements retry. When any worker fails and retry budget remains, the workers node sets `status="worker_retry"`, and the router sends execution back to the workers node with an incremented version.
- **`researched_partial → reviewer`**: The "always forward" property. Even if all retries are exhausted and some workers produced no output, the pipeline proceeds to the reviewer. The reviewer filters non-reviewable outputs (timeouts, skips) and scores only what's available.
- **Terminal errors**: Only three conditions abort the pipeline: no sub-tasks at all (decomposition failure), no reviewable outputs (all workers timed out/skipped), and no scored works (reviewer failure cascading to writer failure). These represent unrecoverable states where proceeding would produce meaningless output.

The conditional edges are applied uniformly to all 4 nodes (`pipeline/research.py:443-444`):
```python
for node in ("head", "workers", "reviewer", "writer"):
    workflow.add_conditional_edges(node, router,
        {"workers": "workers", "reviewer": "reviewer", "writer": "writer", END: END})
```
This means every node can route to any other node — the routing is purely status-driven. In practice, the flow is linear with the workers self-loop, but the uniform edge setup allows future extension (e.g., a `reviewed_retry → workers` path if the reviewer finds systemic quality issues requiring worker re-execution).

### 9. Resume State Reconstruction

The `_try_load_resume_state()` method (`pipeline/research.py:71-197`) enables the `--resume` flag to restart an interrupted pipeline from its last checkpointed state:

1. Reads `decomposition.json` to recover the `sub_tasks` list.
2. Validates the dependency graph via `BasePipeline._validate_dependencies()`, warning about cycles.
3. Scans `agent_log/` for worker checkpoint files, extracting the maximum version and per-worker success flags from checkpoint metadata.
4. Reconstructs `worker_outputs` from `.md` files on disk, gated by checkpoint success — a worker is considered successful if its checkpoint declares `success` or `has_written_output`, or if it has no checkpoint but its output file exists on disk.
5. Scans for `scoring_report.json` and `research_proposal.tex` to determine the resume status.
6. Sets the status to the latest completed stage (`written` > `reviewed` > `researched` > `worker_retry` > `decomposed`), enabling the pipeline to skip completed stages.

This resume mechanism is critical for long-running research pipelines where workers may take 15+ minutes each — a pipeline with 8 workers × multiple versions could run for hours, and losing all progress on a mid-pipeline crash would be prohibitively expensive.

### 10. Backend-Aware Worker Task Generation

The `build_task()` method of `ResearchWorkerRole` (`research/worker_agent.py:17-31`) generates fundamentally different prompts for DeepSeek vs. Claude CLI:

**DeepSeek prompt** (`_build_worker_task_deepseek`, lines 86-163): Includes:
- Explicit instructions for each available tool (`web_search`, `download_file`, `web_fetch`, `call_claude`), with the two-step pattern (`download_file` → `read_file`) for arxiv.org access.
- Numbered instruction format with tool-specific steps.
- The `call_claude` tool for "deep reasoning and synthesis on specific questions."

**Claude CLI prompt** (`_build_worker_task_claude_cli`, lines 166-228): Includes:
- Pre-downloaded reference material section (files pre-fetched by `_prefetch_arxiv_sources()` at the framework level).
- Instructions to use `Read`, `Write`, `WebSearch`, and `Bash` (Claude CLI native tools).
- Explicit warning: "For any arxiv.org or academic URLs found in search results: do NOT try to fetch them directly (domain verification will block them). Instead, note the URLs in your findings."
- The agent is told "Synthesize your findings using your own reasoning. Identify patterns, compare approaches, and note trade-offs." — in contrast to DeepSeek which delegates deep reasoning to `call_claude`.

Both prompts share the same output structure (Overview, Key Methods, Important Papers, Open Questions, Relevance sections) and the same emphasis on "technical depth, concrete details, and accuracy."

**Pre-fetch layer** (`_prefetch_arxiv_sources`, `research/worker_agent.py:232-275`): Searches for the sub-task's title + description (truncated to 200 chars), extracts arxiv.org URLs via regex, downloads up to 3 to local `agent_log/prefetched_NN_*.html` files via `download_file()` (which uses Python urllib, bypassing Claude Code's cc-switch), and returns the local paths for prompt injection.

## Important Papers & References

- **Dean, J. and Ghemawat, S. "MapReduce: Simplified Data Processing on Large Clusters" (OSDI 2004)** — The foundational paper establishing the Map phase (decomposition into independent sub-problems) and Reduce phase (aggregation of results) pattern. UMAF's ResearchPipeline directly implements this pattern adapted for LLM-based research agents: decompose (head) → map (parallel workers) → reduce (reviewer scoring + writer synthesis). The key adaptation is handling stochastic, stateful workers rather than deterministic, stateless mappers.

- **Besta, M. et al. "Graph of Thoughts: Solving Elaborate Problems with Large Language Models" (AAAI 2024)** — Introduces modeling LLM reasoning as a graph where thoughts are nodes and dependencies are edges, enabling topological ordering of reasoning steps. UMAF's `_topological_levels()` and `_run_workers_with_deps()` implement a similar dependency-graph execution model, though applied at the task level rather than the reasoning-step level. The Federation of Agents (FoA) framework explicitly cites Graph of Thoughts as inspiration.

- **Yao, S. et al. "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)** — The canonical Thought→Action→Observation loop underlying all of UMAF's agent interactions. The ResearchPipeline's workers follow this pattern: reason about the sub-topic (Thought) → search/download/read (Action) → incorporate findings (Observation), iterating until TASK_COMPLETE.

- **Chen, Y. et al. "A-MapReduce: Executing Wide Search via Agentic MapReduce" (arXiv:2602.01331, 2025)** — Introduces task matrix + template decomposition with experiential memory for evolution. Achieves 45.8% faster execution and $1.10 cost savings through adaptive batching and manager-driven orchestration. UMAF's simpler, template-free decomposition approach trades the efficiency gains of pre-defined templates for flexibility across arbitrary research topics.

- **Chao, Y. et al. "LLM×MapReduce-V3: Enabling Interactive In-Depth Survey Generation through a MCP-Driven Hierarchically Modular Agent System" (EMNLP 2025)** — A 3-agent system (Analysis, Skeleton, Writing) with MCP-based hierarchical orchestration and human-in-the-loop interaction. Demonstrates the trajectory from fully autonomous (UMAF) to interactive survey generation, where human feedback refines the decomposition and synthesis.

- **D³MAS: Decompose, Deduce, and Distribute (arXiv:2510.10585, 2025)** — Identifies 47.3% average knowledge duplication across multi-agent communications and proposes a 3-layer heterogeneous graph architecture to eliminate overlap. Achieves 8.7-15.6% accuracy improvement across HumanEval, MMLU, HotpotQA. UMAF's simpler MD5-fingerprint deduplication and dependency-injection-based context sharing address the same problem at lower architectural complexity.

- **Agent-Oriented Planning in Multi-Agent Systems (ICLR 2025, HKUST)** — Three design principles (Solvability, Completeness, Non-Redundancy) for task decomposition in multi-agent systems. The Non-Redundancy principle directly motivates UMAF's dependency declarations and dependency-injection mechanism, which prevent workers from duplicating upstream research.

- **PaperOrchestrator: An LLM-Orchestrated Multi-Agent Pipeline for Automated End-to-End Scientific Paper Writing (Springer, 2025)** — A 7-stage pipeline achieving 90.0% LaTeX compilation rate and BERTScore F1 0.674. Notably uses "section-by-section robust LaTeX conversion" — the same sectioned output strategy UMAF employs in its writer role, though PaperOrchestrator uses a dedicated conversion stage rather than generating LaTeX directly.

- **Schmidgall, S. et al. "Agent Laboratory: Using LLM Agents as Research Assistants" (arXiv:2501.04227, 2025)** — An end-to-end autonomous research workflow with Literature Review, Experimentation, and Report Writing phases. Uses specialized agents with integrated arXiv, HuggingFace, Python, and LaTeX tools. The 3-phase structure mirrors UMAF's head→workers→reviewer→writer decomposition, though Agent Laboratory includes an experimentation phase (code execution) that UMAF's ResearchPipeline does not.

- **Gamma, E. et al. "Design Patterns: Elements of Reusable Object-Oriented Software" (Addison-Wesley, 1994)** — The Template Method pattern is the architectural foundation of `BaseDecomposerRole` and the `AgentRole` ABC. The State pattern (representing pipeline status as a state machine with status-driven routing) governs the flow dict router.

- **Diagrid. "Still Not Durable: How Microsoft Agent Framework and Strands Agents Repeat the Same Mistakes" (March 2026)** — Systematic evaluation finding that all 5 major agent frameworks persist state but none guarantee completion. Identifies the gap between "checkpointing" (storage) and "durable execution" (guaranteed completion). UMAF's version-bump retry occupies an intermediate position — more reliable than basic checkpointing but without the auto-resume, duplicate prevention, and distributed locking of true durable execution engines.

- **LangChain. "LangGraph: A Library for Building Stateful, Multi-Actor Applications with LLMs" (2024)** — The `StateGraph` abstraction, `TypedDict` state management, and conditional edge routing used by all UMAF pipelines. UMAF extends this with `BasePipeline._status_router()` for declarative flow map-based routing and `BasePipeline._run_workers_with_deps()` for topology-aware parallel execution.

## Open Questions & Future Directions

1. **Partial-work recovery within workers**: The current retry system is all-or-nothing at the worker level — a worker that times out at 14 minutes of a 15-minute task must restart from scratch. True incremental recovery (resume a worker from its last completed subtask within its own multi-step research process) would require finer-grained checkpointing within the worker's agent loop, not just between versions.

2. **Dynamic parallelism scaling**: The current system uses a fixed `max_workers=len(level_tasks)` — running all tasks in a level simultaneously. For broad decompositions (8+ workers), this could overwhelm API rate limits or local resources. An adaptive parallelism controller that scales concurrency based on observed API latency and error rates would improve throughput under resource constraints.

3. **Cross-worker knowledge sharing during execution**: Currently, workers only receive dependency outputs from completed upstream tasks. There is no mechanism for concurrently executing workers (within the same topological level) to share intermediate findings. A shared knowledge bus or blackboard architecture could allow workers to benefit from each other's partial discoveries, potentially reducing redundant work.

4. **Scoring calibration and inter-reviewer consistency**: The reviewer's 5-dimension scoring is a single LLM's judgment with no calibration against human evaluations or inter-reviewer consistency checks. Multiple reviewer agents scoring independently and comparing results (similar to PaperWrite AI's 3-persona reviewer voting) would produce more reliable rankings.

5. **Automatic dependency inference**: Currently, dependency declarations must be specified manually by the head agent in the decomposition JSON. An automatic dependency inference step — where the head agent or a separate analyzer examines sub-topic descriptions for semantic dependencies (e.g., "sub-topic B cannot be researched without understanding sub-topic A's findings") — would improve decomposition quality for complex topics where the head agent fails to identify implicit dependencies.

6. **Cost-aware worker scheduling**: The pipeline has no mechanism to estimate or limit per-worker API costs. A worker researching a broad sub-topic could consume significantly more tokens than a worker on a narrow sub-topic. Cost estimation based on sub-topic complexity (using the same heuristics as `_sizing_guide()`) and per-worker token budgets would enable predictable pipeline costs.

7. **LaTeX compilation verification**: The current `parse_result()` for the writer checks that the `.tex` file exists and contains `\input{}` commands, but does not attempt to compile it with `pdflatex`. A post-generation compilation step (as implemented by PaperWrite AI and PaperOrchestrator) would catch syntax errors and trigger automatic fixes, improving the 90% target compilation rate.

8. **Streaming output during research**: The pipeline is entirely batch-oriented — users see no output until the entire pipeline completes or a node fails. Streaming intermediate results (worker summaries as they complete, reviewer scores as they're calculated) would improve UX for long-running (multi-hour) research pipelines and enable early termination if results are sufficient.

9. **Adaptive version limits**: `RESEARCH_MAX_VERSIONS=6` and `RESEARCH_MAX_WORKER_RETRIES=5` are hardcoded constants. Different research topics have different failure characteristics — a topic with clear, well-defined sub-topics might succeed in 1-2 versions, while an ambiguous, cutting-edge topic might benefit from more retry attempts. Adaptive limits based on observed progress (improvement in output quality between versions) would optimize resource allocation.

10. **Cross-pipeline feedback loop**: The ResearchPipeline's outputs (scored works, LaTeX report) could feed into other pipelines — for example, the SelfEvolution pipeline could analyze research reviewer scores to identify systematic weaknesses in worker prompt design, or the Feature pipeline could implement code changes recommended by research findings. This would close the loop between research and implementation within UMAF.

## Relevance to Main Topic

The ResearchPipeline is the most architecturally complete demonstration of UMAF's design philosophy — the layered resilience architecture (decomposition fallback → dependency ordering → version-bump retry → honest parse_result → scoring fallback → LaTeX template fallback) implements the principle that **every stage must have a programmatic fallback**, ensuring the pipeline never deadlocks or produces empty output. This philosophy is consistent across all 7 UMAF pipelines (coder↔reviewer loops have max iteration limits, skill detectors have deterministic fallbacks, etc.), but the ResearchPipeline implements it at the greatest depth with the most stages.

The pipeline's dependency-ordered execution via `_topological_levels()` and `_run_workers_with_deps()` is a shared infrastructure capability used by both the ResearchPipeline and CoderPPPipeline — demonstrating that UMAF's `BasePipeline` design successfully abstracts cross-pipeline concerns. The version-bump retry system with `CheckpointManager.load_previous()` is also shared across pipelines, though the ResearchPipeline uses it most extensively (6 versions × 5 retries vs. CoderPP's 5 versions × 5 retries).

For the broader multi-agent LLM research community, the ResearchPipeline represents a pragmatic middle-ground design point: it provides significantly more reliability than simple sequential or flat-parallel agent execution through its dependency management and checkpoint-based retry, but stops short of the architectural complexity of full durable execution engines (Temporal, Conductor, Restate) that require external infrastructure. This makes it suitable for research and development contexts where occasional pipeline failures are acceptable but the cost of total failure (losing all progress on a multi-hour run) is not. The key design insight — that **checkpointing is necessary but not sufficient for reliability, and must be paired with orchestration-layer retry logic, honest success verification, and programmatic fallbacks** — is applicable to any multi-agent LLM system operating at scale.
