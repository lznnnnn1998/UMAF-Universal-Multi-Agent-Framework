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
| Unit/smoke tests | 0 | 0 | 0 | 8/8 | 8/8 | 8/8 | 15/15 | 42/42 | **97/97** |
| Python | 3.9 | 3.9 | 3.9 | 3.11 | 3.11 | 3.11 | 3.11 | 3.11 | 3.11 |
| Pipelines | 2 | 2 | 2 | 2 | 2 | 3 | 3 | 5 | **6** |

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

### Related
[[version_diffs]], [[architecture_progress]], [[oop_refactoring]]
