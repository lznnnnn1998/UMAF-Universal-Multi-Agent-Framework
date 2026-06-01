---
name: pipeline-robustness
description: "v1.4 pipeline robustness — dependency stop-on-failure, version-bump retry with context reuse, honest parse_result, worker retry state machine"
metadata:
  type: project
---

## v1.4 Pipeline Robustness (June 2026)

### Three-Layer Bug Fix

The research pipeline had a three-layer failure in its retry/dependency system. All three layers had to be fixed for the system to work.

**Layer 1 — Same-version retry has no context:** `_run_parallel_agents` retried failed workers with the same version number. `BaseAgent(version=1)` → `CheckpointManager.load_previous(1)` finds nothing (only looks for version < 1) → starts fresh with no memory of the previous attempt. The agent repeats the same mistakes. Fix: removed `retry_failures=True` from the `_run_workers_with_deps` call; version-bump retries (`version=2,3,4`) correctly trigger `load_previous(version)` in `agent.py:489`, which restores messages, resets iterations, and injects a retry context message.

**Layer 2 — Dependency DAG doesn't stop on failure:** `_run_workers_with_deps` grouped subtasks into topological levels and ran them in order, but a failure in level 1 didn't stop level 2 from starting. Worker 03 (which depends on worker 02's output) would start while worker 02 was still failing. Fix: after each level completes, if `failed > 0` and more levels remain, break out of the loop — downstream tasks are deferred for retry.

**Layer 3 — `parse_result` always reports success:** `ResearchWorkerRole.parse_result()` always set `output_file` in its return dict, even when the file was never written. Workers counted as "succeeded" → stop-on-failure never triggered. Fix: check `os.path.isfile(os.path.join(working_dir, output_file))` before reporting; empty string if the file doesn't exist.

### Worker Retry State Machine

New `worker_retry` status in the research flow dict creates a dedicated retry loop:
```
workers → (all succeeded?) → researched → reviewer
       → (some failed?) → worker_retry → workers (version+1, failed only)
       → (max retries?) → researched_partial → reviewer
```

Constants: `RESEARCH_MAX_VERSIONS=4`, `RESEARCH_MAX_WORKER_RETRIES=3`, `WORKER_TIMEOUT=600`.

### Verified (June 2026)

Topic: "Propose a brand new optimized attention mechanism"
- 7/7 workers produced output (100%)
- Scores: 48, 47, 45, 44, 43, 39, 38 out of 50
- Writer: 60KB LaTeX, 11 sections, 17 equations, 13 tables, 47 references
- Pipeline time: 443s
- Several workers needed v03/v05 version-bump retries — mechanism worked

**Why:** Without these fixes, the dependency graph was decorative — workers ran in order but failures didn't propagate, and retries started from scratch with no memory of prior attempts.

**How to apply:** When adding new pipeline stages with dependencies, use `_run_workers_with_deps` (not raw `_run_parallel_agents`), implement honest `parse_result` that checks file existence, and route through the `worker_retry` state machine for automatic retry.

### Related
[[version_diffs]], [[key_updates]], [[architecture_progress]]
