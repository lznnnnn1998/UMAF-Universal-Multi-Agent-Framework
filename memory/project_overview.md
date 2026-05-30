---
name: project-overview
description: "Universal Multi-Agent Framework — full architecture, both backends, all tools, circuit breakers"
metadata: 
  node_type: memory
  type: project
  originSessionId: d4200744-181c-4ba7-9d5b-36d64631acd6
---

The Universal Multi-Agent Framework is built with LangChain + LangGraph, supporting two LLM backends (DeepSeek API and Claude CLI subprocess) and two pipelines (coder/reviewer and research).

## Architecture

```
main.py ──▶ graph.py ──▶ agent.py ──▶ llm.py   (Phase 1: coder/reviewer)
                │            │            │
                ▼            ▼            ├── ChatOpenAI (deepseek-chat)
          MultiAgentState   tools.py      └── ClaudeCLILLM (subprocess)

main.py ──▶ research/graph.py ──▶ research/head_agent.py    (Phase 2: research)
                │                   research/worker_agent.py
                ▼                   research/reviewer_agent.py
          ResearchState             research/writer.py
```

## Modules

- **`llm.py`**: Two backends. DeepSeek via `ChatOpenAI` (deepseek-chat, temp=0.3). Claude CLI via `ClaudeCLILLM` (shells out to `claude -p`, injects env from `claude_env_sample.json`, supports `--allowedTools` and `cwd`).
- **`tools.py`**: Five tools — `read_file`, `write_file`, `run_command` (30s timeout), `call_claude` (120s timeout, env injection), `web_search` (DuckDuckGo Lite scraping). All accept `working_dir` for path sandboxing. Exported as `TOOL_MAP`.
- **`agent.py`**: Core `run_agent()` function. DeepSeek path: JSON tool-call loop with brace-counting parser, circuit breakers (force wrap-up at ≤3 steps, error spiral detection, unknown tool warnings). Claude CLI path: single `claude -p` invocation with tool name translation (Python names → native names via word-boundary regex), retry on timeout/error, `cwd` sandboxing.
- **`graph.py`**: Coder/reviewer pipeline. `MultiAgentState` TypedDict. Coder has all 5 tools, reviewer has no write_file. Max 5 cycles. Coder resets `review_passed=False` on each run.
- **`main.py`**: CLI with `--mode coder|research`, `--backend deepseek|claude_cli`, `--working-dir`.
- **`claude_config.py`**: Loads `claude_env_sample.json` env vars (ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN, ANTHROPIC_MODEL, etc.). `merge_claude_env()` merges with os.environ.

## Research Sub-Module

- **`research/head_agent.py`**: `decompose_topic()` via `run_agent`. Fallback: keyword extraction from topic (splits on commas/`and`/`vs`), generates 5+ specific sub-tasks.
- **`research/worker_agent.py`**: `research_subtask()` with all 5 tools. Returns `{sub_task_id, title, output_file, summary}`.
- **`research/reviewer_agent.py`**: `review_and_score()` — 5-dimension rubric (depth, accuracy, relevance, clarity, originality, each 1-10). Writes `scoring_report.json`. Non-greedy JSON extraction.
- **`research/writer.py`**: `write_proposal()` — LLM generates LaTeX. `_fallback_latex()` with `_latex_escape()` (all 10 special chars escaped). Template-based with scoring table and bibliography.
- **`research/graph.py`**: `ResearchState` TypedDict. ThreadPoolExecutor for head (120s timeout) and workers (parallel, max 4, 300s each). Post-hoc MD5 dedup. Router always moves forward; partial results accepted.

## Key Design Decisions
- Tool definitions separate from implementations (metadata + TOOL_MAP)
- Explicit `working_dir` parameter — no global state
- Circuit breakers at agent level (error spirals) and graph level (thread timeouts, dedup)
- Python 3.9 compatible (Optional[X], no walrus operator in hot paths)
- Fallbacks at every pipeline stage

**Why:** Full framework documentation after phase 2 completion.
**How to apply:** New contributors should start here to understand the architecture, then read [[backend-architecture]] for LLM details and [[circuit-breakers]] for resilience patterns.
