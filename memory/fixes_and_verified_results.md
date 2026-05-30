---
name: fixes-and-verified-results
description: "The 12 bugs found and fixed during code review — before/after metrics, test verification"
metadata: 
  node_type: memory
  type: project
  originSessionId: d4200744-181c-4ba7-9d5b-36d64631acd6
---

## 12 Fixes Applied (May 2026)

| # | Severity | Issue | File:Line | Fix |
|---|----------|-------|-----------|-----|
| 1 | Critical | Retry prompt used untranslated `task` — workers guaranteed to fail tools on retry | agent.py:327 | `task` → `translated_task` |
| 2 | High | `"error"` substring matched research content, causing false retries | agent.py:315-319 | Check `"error:"` prefix, `"[stderr]"`, `"timed out"` |
| 3 | High | Sequential workers → 35min worst-case (7×300s) | research/graph.py:71-171 | ThreadPoolExecutor, max 4 concurrent |
| 4 | High | Daemon threads orphaned `claude -p` subprocesses on timeout | research/graph.py:39-67 | ThreadPoolExecutor in head + workers (subprocess.run timeout kills processes) |
| 5 | Medium | Tool name translation missed "using", "via", "call X with" phrasings | agent.py:79-91 | `\b` word-boundary regex replacement |
| 6 | Medium | Stale `review_passed=True` could skip reviewer after coder modifies code | graph.py:99 | Coder resets `review_passed=False` |
| 7 | Medium | Fallback decomposition always 5 generic titles | head_agent.py:86-133 | Keyword extraction from topic, pad to ≥5 |
| 8 | Low | "Only 0 step(s) remaining" | agent.py:186-187 | Special-case: "This is your LAST step" |
| 9 | Low | Regex `[^}]+` couldn't parse nested JSON args | agent.py:129-157 | Brace-counting extraction |
| 10 | Low | Greedy `[\s\S]*` could match across JSON arrays | reviewer_agent.py:121 | `[\s\S]*?` non-greedy |
| 11 | Low | Only `&` and `%` escaped in LaTeX — `_`, `#`, `{`, etc. missed | writer.py:134-145 | `_latex_escape()`: all 10 special chars |
| +1 | Critical | `claude -p` wrote files to project root instead of temp working dir | llm.py:47, agent.py:307 | `cwd=working_dir` in subprocess.run |

## Verified Results

Test: `python3 main.py -m research -b claude_cli "model quantization: QAT, PTQ, stochastic rounding"`

| Metric | Before Fixes | After Fixes |
|--------|-------------|-------------|
| Worker completion | 3/7 (43%) | **5/5 (100%)** |
| Decomposition | Hardcoded fallback | LLM-generated |
| Top score | 25/50 (auto-ranked) | **46/50** (real scoring) |
| Worker parallelism | Sequential (35min max) | Parallel 4× (9min wall) |
| Research files on disk | 0 | Comprehensive 40KB report + JSON |

## Root Causes Fixed

- **Workers timing out silently**: Caused by: (a) retry using untranslated task names (fix #1), (b) no cwd sandboxing so files went to wrong directory (fix +1), (c) sequential execution magnifying delays (fix #3)
- **Low-quality scoring**: Caused by reviewer unable to find files → forced 25/50 auto-rank. Fixed by cwd sandboxing so files land in working directory
- **Pipeline deadlocks**: Caused by daemon threads and sequential timeout accumulation. Fixed by ThreadPoolExecutor with proper timeouts (fixes #3, #4)

**Why:** Documents what broke, why, and how it was fixed. Prevents regressions and guides future contributors.
**How to apply:** Before adding new features, verify the pipeline still passes with `python3 main.py -m research -b claude_cli --working-dir research_output "test topic"`. Related: [[project-overview]], [[circuit-breakers]], [[research-pipeline]], [[v1.2-changes]].

## v1.2 Fixes (May 2026)

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 1 | Critical | Workers used nested `claude -p` calls (agent IS already Claude Code) → recursive invocations, timeouts | Backend-aware worker tasks: `claude_cli` uses WebSearch+Write+Read directly |
| 2 | Critical | Empty tools list → no `--allowedTools` → all tools available → permission denied on Write/Bash | Head agent gets Read-only tools; `--allowedTools` always passed |
| 3 | Critical | Global `Bash(*)`/`Write(*)`/`Edit(*)` — wildcard permissions | Scoped to project directory in `.claude/settings.local.json` |
| 4 | High | `claude_env_sample.json` with real API keys tracked in git | Removed from tracking, replaced with `.example.json` placeholder |
| 5 | High | Head agent used WebSearch for decomposition → slow (120s+) | Read-only tools, pure reasoning (~70s) |
| 6 | Medium | No debugging visibility into agent failures | Conversation logger: `agent_logs/<name>_<timestamp>.json` |
| 7 | Medium | Worker parallelism 4 → system resource contention | Reduced to 2 concurrent workers |
| 8 | Medium | `ClaudeCLILLM` timeout 120s → workers timing out | Increased to 300s |
| 9 | Low | System prompt said "Use Bash for any command-line operation" | Removed; `call_claude` spec discourages nesting |

### Verified v1.2 Results

Test: `python3 main.py -m research -b claude_cli --working-dir research_output "Flash Attention, Multi-Query Attention, Grouped Query Attention, Paged Attention, Ring Attention"`

| Metric | Before (v1.1) | After (v1.2) |
|--------|-------------|-------------|
| Research files on disk | 0 | **4/6 (21-26KB each)** |
| Top score | 25/50 (auto-ranked) | **43/50** (real scoring) |
| Agent logs | 0 | 9 logs (all agents) |
| LaTeX output | 3.6KB (hollow template) | **41KB** (equations, tables) |
| Head agent time | 120s+ (timeout) | ~70s (success) |
