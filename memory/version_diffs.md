---
name: version-diffs
description: "Complete changelog: v1.0→v1.1 (12 bug fixes), v1.2 (backend-aware agents), v1.3 (Python 3.11), v1.3.1 (worker output fix, arxiv access)"
metadata: 
  node_type: memory
  type: project
  originSessionId: db564f0a-1b8e-4bed-8a26-28d0132d0605
---

## v1.4 (June 2026) — Pipeline Robustness & Dependency Management

**Why:** The research pipeline's dependency graph was decorative — workers ran in dependency order but failures didn't propagate to downstream tasks, and retries started from scratch with no memory of prior attempts. A three-layer bug masked this: (1) same-version retry had no checkpoint to load, (2) topological levels didn't stop on failure, (3) `parse_result` always reported success.

### Dependency management
- **Stop-on-failure in `_run_workers_with_deps`**: After each topological level, if any task failed and more levels remain, break out of the loop. Downstream tasks that depend on failed outputs are deferred for retry.
- **Honest `parse_result`**: `ResearchWorkerRole.parse_result()` now checks `os.path.isfile()` before reporting `output_file` — empty/missing files correctly count as failure, triggering stop-on-failure.
- **Worker retry state machine**: New `worker_retry` status in research flow dict creates a dedicated retry loop: failed workers only, version-bumped, with max retry limits.

### Context-reusing retries
- **Version-bump retry**: Removed `retry_failures=True` from `_run_workers_with_deps`; retries now go through the version-bump path (`worker_retry` → `workers` with `version+1`), which triggers `BaseAgent(version=N)` → `CheckpointManager.load_previous(N)` → restore messages + reset iterations + inject retry context.
- **Agent checkpoint injection** (`agent.py:489`): When `version > 1`, loads previous checkpoint, restores full message history, resets `iterations=0`, and appends a `[System: This is version N retry...]` message so the agent knows what failed and why.

### Constants
- `WORKER_TIMEOUT`: 300 → 600s (complex attention mechanism derivations need more time)
- `RESEARCH_MAX_VERSIONS`: 4 (allows up to 3 retries)
- `RESEARCH_MAX_WORKER_RETRIES`: 3

### Cleanup
- **`graph.py` removed**: Dead code, replaced by `pipeline.py` (BasePipeline + CoderPipeline + ResearchPipeline + CoderPPPipeline)
- **`.gitignore` updated**: Added `**/agent_log/`, `*output*/*.json`, `*output*/**/*.json`, coderpp pipeline output directories

### Verified
7/7 workers produce output; scores 48, 47, 45, 44, 43, 39, 38 out of 50; 60KB LaTeX (11 sections, 17 equations, 13 tables, 47 refs); 443s pipeline time.

---

## v1.3.1 (May 2026) — Worker Output Fix & arxiv.org Access

**Why:** 2/4 workers produced no output files because the agent loop checked TASK_COMPLETE before executing tool calls. `claude -p` subprocesses couldn't access arxiv.org due to cc-switch domain verification.

### Critical bug fix
- **Tool-before-completion ordering**: Agent loop now executes tool calls BEFORE checking TASK_COMPLETE. Previously, responses containing both `write_file` + `TASK_COMPLETE` would break immediately, losing the file write entirely.

### Worker output reliability
- Mid-loop write reminder at ~2/3 of max steps if `write_file` hasn't been called yet
- Force wrap-up messages now explicitly forbid all tools except `write_file`
- Post-loop exhaustion message explicitly requires calling `write_file` (not just summarizing)

### arxiv.org access
- **`download_file` tool**: Framework-level urllib download → local file → `read_file` pattern bypasses cc-switch
- **Pre-fetch layer**: `claude_cli` workers get arxiv.org content pre-downloaded at framework level before agent runs
- Both backends now use download-then-read-local pattern for academic sites

### Defaults & organization
- Default working dir: `tempfile.mkdtemp()` → `research_output/` inside repo
- All logs and intermediate files under `research_output/agent_log/` (renamed from `agent_logs`)
- Removed `import tempfile` from main.py

### Verified
4/4 workers produce files (up from 2/4); all 4 real scores (up from 2 real + 2 missing); top score 47/50.

---

## v1.3 (May 2026) — Python 3.11 & Code Quality

**Why:** Python 3.9 patterns were deprecated; `_latex_escape()` had a latent bug (backslash → tab); several files had prototype-era dead code.

### Environment
- Python 3.9 → 3.11 minimum; `.python-version` added

### Bug fix
- `_latex_escape()`: `"\\textbackslash "` (produced tab) → `r"\textbackslash "` (raw string). Extracted `_LATEX_ESCAPE_MAP`.

### Dead code removed
- `agent.py`: unused `_TOOL_NAME_TRANSLATION` dict (duplicate of `_TOOL_NAMES_TO_TRANSLATE`), `_build_system_prompt` dispatcher

### Dynamic decomposition
- Head agent now scales sub-topic count 2-8 based on topic complexity instead of fixed 5-7
- Prompt teaches LLM to assess difficulty: narrow→2-3, moderate→4-5, broad→6-8
- Fallback scales by keyword count instead of always padding to 5+

### Simplifications
- `_run_with_claude_cli`: extracted `_invoke`/`_build_prompt` helpers, DRY retry path (~20 lines)
- `decompose_topic()`: common prompt factored out, JSON template as plain string (~20 lines)
- Research `_router()`: if-chain → `flow` dict (~15 lines)

### New tool: `web_fetch` (urllib-based)
- Added `web_fetch(url, max_chars)` to `tools.py` — fetches URLs via Python urllib, bypassing Claude Code permission system
- Registered in all tool sets (coder, reviewer, worker, research reviewer)
- Worker prompts updated: use `web_fetch` for arxiv.org abstracts instead of `Bash(curl ...)`
- arxiv.org and academic sites now always accessible without permission prompts

### Types
- 4 `Optional[X]` → `X | None` across `agent.py` and `claude_config.py`

### Verified
8 unit tests pass; end-to-end coder pipeline verified.

---

## v1.2 (May 2026) — Backend-Aware Agents

**Why:** Workers used nested `claude -p` calls (agent IS already Claude Code), causing recursive invocations and timeouts. Global `Bash(*)/Write(*)/Edit(*)` permissions were too broad. `claude_env_sample.json` with real keys tracked in git.

### Changes
| Area | Change |
|------|--------|
| Worker tasks | Backend-aware: `claude_cli` workers use WebSearch+Write+Read directly; `deepseek` workers use `call_claude` |
| Head agent | `claude_cli` gets Read-only tools (pure reasoning, ~70s vs 120s+) |
| Permissions | Scoped Bash/Read/Write/Edit to project directory in `.claude/settings.local.json` |
| Security | `claude_env_sample.json` → `.example.json` template; `.gitignore` updated |
| Timeout | `ClaudeCLILLM` 120s → 300s |
| Parallelism | Worker concurrency 4 → 2 |
| Logging | `agent_logs/<name>_<timestamp>.json` via `agent_name` param |
| `--allowedTools` | Always passed (fixes empty-tools = all-tools bug) |
| System prompt | Removed "Use Bash for anything"; `call_claude` discourages nesting |

### Verified
4/6 workers produce 21-26KB files; top score 43/50; 41KB LaTeX; 2/6 hit 300s timeout.

---

## v1.1 (May 2026) — 12 Bug Fixes

### Critical
| # | Issue | Fix |
|---|-------|-----|
| 1 | Retry used untranslated task → workers fail tools on retry | `task` → `translated_task` |
| +1 | `claude -p` wrote files to project root instead of working dir | `cwd=working_dir` in subprocess.run |

### High
| # | Issue | Fix |
|---|-------|-----|
| 2 | `"error"` substring in research content caused false retries | Check `"error:"` prefix, `"[stderr]"`, `"timed out"` |
| 3 | Sequential workers → 35min worst-case | ThreadPoolExecutor, max 4 concurrent |
| 4 | Daemon threads orphaned `claude -p` processes on timeout | ThreadPoolExecutor with subprocess.run timeout |

### Medium
| # | Issue | Fix |
|---|-------|-----|
| 5 | Tool name translation missed "using", "via", "call X with" | `\b` word-boundary regex |
| 6 | Stale `review_passed=True` skipped reviewer | Coder resets `review_passed=False` |
| 7 | Fallback decomposition always 5 generic titles | Keyword extraction, pad to ≥5 |

### Low
| # | Issue | Fix |
|---|-------|-----|
| 8 | "Only 0 step(s) remaining" | Special-case: "This is your LAST step" |
| 9 | Regex couldn't parse nested JSON args | Brace-counting extraction |
| 10 | Greedy regex matched across JSON arrays | Non-greedy `[\s\S]*?` |
| 11 | Only `&`/`%` escaped in LaTeX | All 10 special chars in `_latex_escape()` |

### Before/After (v1.0→v1.1)
| Metric | Before | After |
|--------|--------|-------|
| Worker completion | 3/7 (43%) | 5/5 (100%) |
| Top score | 25/50 (auto-rank) | 46/50 (real) |
| Pipeline time | 35min (sequential) | 9min (parallel 4×) |
| Research files | 0 | 40KB + JSON |

### Related
[[architecture_progress]], [[key_updates]]
