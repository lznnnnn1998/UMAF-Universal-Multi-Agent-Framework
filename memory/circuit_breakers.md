---
name: circuit-breakers
description: "Resilience patterns — agent-level error spirals, graph-level thread timeouts, dedup detection, forced progress"
metadata: 
  node_type: memory
  type: project
  originSessionId: d4200744-181c-4ba7-9d5b-36d64631acd6
---

The framework uses circuit breakers at two levels to prevent deadlocks, infinite loops, and resource waste.

## Agent-Level (agent.py)

All in `_run_with_deepseek()`:

1. **Force wrap-up at ≤3 steps remaining**: Appends urgency message to the conversation. At 0 steps: "This is your LAST step — you must wrap up NOW."

2. **Unknown tool warnings**: When the LLM tries a tool not in TOOL_MAP, it's added to a blocklist. Next iteration warns: "the following tools are NOT available (do not attempt them again)."

3. **Error spiral detection**: Consecutive tool errors matching `_PERSISTENT_ERRORS` (`"timed out"`, `"not found"`, `"no such file"`, `"permission denied"`) are tracked. At 3 consecutive errors, forces wrap-up: "CRITICAL: The last several tool calls all failed... Write your best-effort findings."

4. **Post-loop forced summary**: After exhausting all steps without TASK_COMPLETE, one final message forces a summary + TASK_COMPLETE.

5. **Retry on Claude CLI timeout**: `_run_with_claude_cli` retries once on timeout/stderr with a simplified prompt. The retry uses the translated task text (fix #1).

## Graph-Level (research/graph.py)

1. **Head agent timeout**: `HEAD_TIMEOUT_SECONDS = 120`. ThreadPoolExecutor with `future.result(timeout=120)`. Falls back to `_fallback_decompose()`.

2. **Worker timeout**: `WORKER_TIMEOUT_SECONDS = 300` per worker. ThreadPoolExecutor with individual `future.result(timeout=300)`. Timed-out workers are marked as failed; other workers continue.

3. **Worker parallelism**: `max_workers = min(len(sub_tasks), 4)`. Workers run concurrently via ThreadPoolExecutor (was sequential before fix #3).

4. **Deduplication**: `MAX_DUPLICATE_WORKERS = 2`. MD5 fingerprint of worker summary (first 200 chars normalized). Post-hoc: marks duplicates but doesn't break the loop (parallel workers can't be cancelled mid-flight).

5. **Forced progress**: Router always moves forward. Status `researched_partial` → reviewer. Reviewer auto-ranks (25/50) if LLM scoring fails. Writer falls back to template.

## Coder/Reviewer Graph (graph.py)

1. **Max 5 iterations**: `state["iteration"] >= 5` → END. Prevents infinite coder↔reviewer loops.

2. **Stale review_passed reset**: Coder always sets `review_passed=False` on each run (fix #6), ensuring the reviewer re-evaluates new code before termination.

**Why:** Without these, the pipeline deadlocks (workers hang forever, head hangs, reviewers see stale state). Each breaker was added after observing a real failure mode.
**How to apply:** When adding new graph nodes, always add timeout + fallback. Related: [[project-overview]], [[research-pipeline]].
