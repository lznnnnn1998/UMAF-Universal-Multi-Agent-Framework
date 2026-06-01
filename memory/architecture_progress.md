---
name: architecture-progress
description: "Architecture evolution — two pipelines, two backends, circuit breakers, tool system, fallback strategy"
metadata: 
  node_type: memory
  type: project
  originSessionId: db564f0a-1b8e-4bed-8a26-28d0132d0605
---

## Architecture Overview

UMAF is a LangChain + LangGraph multi-agent framework with two LLM backends and two pipelines.

```
main.py → pipeline.py → agent.py → llm.py              (all pipelines)
              │               │            ├── ChatOpenAI (deepseek-chat)
              ▼               ▼            └── ClaudeCLILLM (subprocess)
        BasePipeline    AgentRole ABC
        ├── CoderPipeline     ├── CoderRole
        ├── ResearchPipeline  ├── ResearchWorkerRole
        └── CoderPPPipeline   ├── ResearchDecomposerRole
                              ├── ResearchReviewerRole
                              ├── WriterRole
                              └── ... (10 roles total)
```

## Two Backends

**DeepSeek** (default): `ChatOpenAI` with `deepseek-chat`, temp=0.3. Agent loop: JSON tool-call → parse → execute → loop. Brace-counting parser for nested objects.

**Claude CLI**: `ClaudeCLILLM` shells out to `claude -p`. Env injected from `claude_env_sample.json` (routes through DeepSeek's Anthropic-compatible proxy). Single invocation per agent (CLI is itself multi-turn). Tool names translated: Python names → native names via `\b` word-boundary regex.

## Two Pipelines

### Coder/Reviewer (`pipeline.py` — `CoderPipeline`)
Coder (all 6 tools) ↔ Reviewer (no write_file). Max 5 cycles. Coder resets `review_passed=False` each run.

### Research (`pipeline.py` — `ResearchPipeline`)
```
head (decompose) → workers (dependency-ordered) → reviewer (score) → writer (LaTeX) → END
```
- **Head**: Backend-aware (v1.2). `claude_cli` = Read-only, pure reasoning. `deepseek` = search tools. Dynamically scales 2-8 sub-topics by complexity (v1.3). 120s timeout, fallback on failure.
- **Workers**: Backend-aware (v1.2). Dependency-ordered via `_topological_levels()` (v1.4). `claude_cli` = WebSearch+Write+Read directly. `deepseek` = `call_claude` for reasoning. 600s timeout each (v1.4). MD5 dedup. Stop-on-failure blocks downstream dependents (v1.4). Version-bump retry with context reuse via `CheckpointManager.load_previous()` (v1.4).
- **Reviewer**: 5-dimension scoring (depth, accuracy, relevance, clarity, originality, each 1-10). Writes `scoring_report.json`. Falls back to auto-rank (25/50).
- **Writer**: LLM generates LaTeX. Falls back to Python template with `_latex_escape()` (10 special chars). Template-based with scoring table and bibliography.

## Five Tools → Six Tools (`tools.py`)
All accept `working_dir` for path sandboxing:
- `read_file`, `write_file` (creates parent dirs)
- `run_command` (30s timeout)
- `call_claude` (120s, env-injected subprocess)
- `web_search` (DuckDuckGo Lite scraping, no API key)
- `web_fetch` (urllib-based, 20s timeout — bypasses Claude Code permission checks; arxiv.org and academic sites always accessible)

Tool definitions (metadata for prompts) separated from implementations (`TOOL_MAP`).

## Circuit Breakers

**Agent-level** (`agent.py`):
- Force wrap-up at ≤3 steps remaining (urgency messages)
- Error spiral detection: 3 consecutive persistent errors → forced best-effort summary
- Unknown tool blocklist: warns, doesn't repeat unavailable tools
- Post-loop forced summary if all steps exhausted
- Claude CLI retry on timeout/error (translated prompt)
- Version-bump retry: `load_previous()` restores messages, resets iterations, injects retry context (v1.4)

**Graph-level** (`pipeline.py`):
- Head agent: 120s ThreadPoolExecutor timeout → fallback decomposition
- Workers: 600s individual timeout (v1.4), max 2 concurrent
- Stop-on-failure: dependency levels block downstream tasks on failure (v1.4)
- MD5 fingerprint dedup (first 200 chars normalized)
- Router always moves forward (`researched_partial` accepted)
- Worker retry state machine: `worker_retry` → `workers` with version+1, max 3 retries (v1.4)
- Reviewer auto-ranks (25/50) if LLM scoring fails; Writer falls back to template

## Key Design Decisions
- Explicit `working_dir` parameter — no global state
- Python >= 3.11: `X | None` syntax (v1.3)
- Fallbacks at every pipeline stage (decomposition, research, scoring, LaTeX)
- Backend-aware task generation (v1.2): no nested `claude -p`
- Conversation logger: `agent_log/<name>_<timestamp>.json` (v1.2)
- Pre-fetch layer (v1.3.1): arxiv.org content pre-downloaded at framework level before `claude_cli` agents run
- Tool-before-completion ordering (v1.3.1): execute tools before checking TASK_COMPLETE to avoid lost writes

## Evolution

| Version | Focus |
|---------|-------|
| v1.0 | Initial: two pipelines, two backends, five tools |
| v1.1 | 12 bug fixes: cwd sandboxing, translation, timeouts, parallel workers |
| v1.2 | Backend-aware agents, scoped permissions, conversation logger, security |
| v1.3 | Python 3.11, dead code removal, simplifications, _latex_escape fix, dynamic decomposition, web_fetch |
| v1.3.1 | Tool-before-TASK_COMPLETE fix, download_file + pre-fetch for arxiv access, 4/4 workers produce files |
| v1.4 | Dependency stop-on-failure, version-bump retry with context reuse, honest parse_result, worker retry state machine, OOP refactoring (5-layer class hierarchy), 7/7 workers |

### Related
[[version_diffs]], [[key_updates]]
