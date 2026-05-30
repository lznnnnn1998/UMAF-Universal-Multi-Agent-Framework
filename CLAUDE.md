# Universal Multi-Agent Framework

A LangChain + DeepSeek framework that mimics Claude Code's autonomous behavior with multi-agent collaboration. Supports two pipelines: **coder** (code generation with review loop) and **research** (head→workers→reviewer→writer for LaTeX research proposals).

## Architecture

```
main.py ──▶ graph.py ──▶ agent.py ──▶ llm.py   (Phase 1: coder/reviewer)
                │            │
                ▼            ▼
          MultiAgentState   tools.py (TOOL_MAP)

main.py ──▶ research/graph.py ──▶ research/head_agent.py    (Phase 2: research)
                │                   research/worker_agent.py
                ▼                   research/reviewer_agent.py
          ResearchState             research/writer.py
```

## Modules

### `llm.py` — LLM Configuration (two backends)

**DeepSeek backend** (default):
- LangChain `ChatOpenAI` with `base_url=https://api.deepseek.com/v1`
- Model: `deepseek-chat`, temp=0.3, max_tokens=4096
- Loads `DEEPSEEK_API_KEY` from `.env`

**Claude CLI backend** (`--backend claude_cli`):
- `ClaudeCLILLM` class shells out to `claude -p` subprocess
- Reads env vars from `claude_env_sample.json` via `claude_config.py`
- Routes through DeepSeek's Anthropic-compatible API proxy
- 120s subprocess timeout, `--output-format text`, `--allowedTools` support
- Accepts `cwd` kwarg for working directory sandboxing

Exports `get_llm(backend)` factory — returns the appropriate LLM instance.

### `tools.py` — Core Tools

Five tools available to agents. All accept `working_dir` for path sandboxing:
- `read_file(path)` — read file contents
- `write_file(path, content)` — write content (creates parent dirs)
- `run_command(command)` — shell command with 30s timeout
- `call_claude(prompt)` — shell out to `claude -p` with env injection (120s timeout)
- `web_search(query, max_results)` — DuckDuckGo Lite scraping (no API key needed)

Exported as `TOOL_MAP: dict[str, Callable]`.

### `agent.py` — Single Agent Loop

`run_agent(task, working_dir, tools, tool_map, max_steps, backend)` runs an autonomous agent. Two code paths:

**DeepSeek path** (`_run_with_deepseek`):
- JSON tool-call loop: LLM returns `{"tool": "...", "args": {...}}`
- Brace-counting JSON parser handles nested objects/arrays in args
- Circuit breakers: force wrap-up at ≤3 steps remaining, error spiral detection at 3 consecutive persistent errors, unknown tool warnings

**Claude CLI path** (`_run_with_claude_cli`):
- Single `claude -p` invocation (the CLI itself is multi-turn with tool access)
- Tool name translation: Python names (`read_file`, `call_claude`, etc.) → Claude CLI native names (`Read`, `Bash`, etc.) via word-boundary regex
- Retry on timeout/error with simplified prompt
- Passes `cwd=working_dir` so files land in the correct directory

Returns `{"messages", "iterations", "success"}`.

### `graph.py` — Coder/Reviewer Pipeline (Phase 1)

Two agents in a review loop:
- **Coder** (all 5 tools): implements the requirement
- **Reviewer** (read_file, run_command, call_claude, web_search — no write_file): reviews code

State: `MultiAgentState` with messages, current_agent, requirement, working_dir, review_passed, iteration.
- Coder resets `review_passed=False` on each run (prevents stale-pass termination)
- Max 5 coder↔reviewer cycles, then terminates

```bash
python3 main.py -m coder "write a hello world script in Python"
```

### `research/` — Research Pipeline (Phase 2)

**`research/head_agent.py`** — Task decomposition:
- `decompose_topic()` calls `run_agent` with DECOMPOSE_TOOLS (run_command, call_claude, web_search)
- Extracts JSON array from LLM output; falls back to `_fallback_decompose()` on failure
- Fallback splits topic on commas/`and`/`vs` to extract keywords, generates 5+ specific sub-tasks

**`research/worker_agent.py`** — Research execution:
- `research_subtask()` calls `run_agent` with WORKER_TOOLS (all 5 tools)
- Task instructs agent to research, write `research_NN_Title.md`, verify, and signal TASK_COMPLETE
- Returns `{sub_task_id, title, output_file, summary}`

**`research/reviewer_agent.py`** — Scoring and ranking:
- `review_and_score()` reads worker output files, scores on 5 dimensions (1-10 each, max 50):
  depth, accuracy, relevance, clarity, originality
- Writes `scoring_report.json`; returns top 3 ranked items
- `_extract_scores()` reads the JSON file first, then falls back to message parsing

**`research/writer.py`** — LaTeX generation:
- `write_proposal()` calls `run_agent` with WRITER_TOOLS (read_file, write_file)
- LLM generates complete LaTeX with preamble, sections, abstract, scoring table
- Falls back to `_fallback_latex()` template if LLM fails
- Full LaTeX escaping via `_latex_escape()` (handles \\, {, }, $, &, #, _, %, ~, ^)

**`research/graph.py`** — Pipeline orchestration with circuit breakers:
```
head (decompose) → workers (parallel research) → reviewer (score) → writer (LaTeX) → END
```
- Head agent: 120s timeout via ThreadPoolExecutor, fallback on timeout
- Workers: parallel execution via ThreadPoolExecutor (max 4 concurrent), 300s timeout each
- Deduplication: MD5 fingerprint of summaries, marks duplicates post-hoc
- Status flow: `decomposed → researched/researched_partial → reviewed → written → END`
- Router always moves forward; partial results accepted at every stage

```bash
python3 main.py -m research -b claude_cli "model quantization: QAT, PTQ, stochastic rounding"
```

### `main.py` — Entry Point

```
python3 main.py [--mode coder|research] [--backend deepseek|claude_cli] [--working-dir PATH] "requirement"
```

Reads requirement from argument, stdin, or interactive prompt. Creates temp working directory.

### `claude_config.py` — Environment Setup

Loads `claude_env_sample.json` which defines 12 env vars for Claude CLI subprocess routing:
- `ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic`
- `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_MODEL`, `ANTHROPIC_SMALL_FAST_MODEL`, etc.

`merge_claude_env()` merges these with `os.environ` for subprocess calls.

## Setup

```bash
pip install -r requirements.txt
# Set DEEPSEEK_API_KEY in .env (for deepseek backend)
# Ensure claude CLI is installed: npm install -g @anthropic-ai/claude-code
# Copy and configure claude_env_sample.json for the claude_cli backend

# Grant subprocess permissions (required for claude_cli backend):
# ~/.claude/settings.json must have:
#   "permissions": { "WebSearch": "*", "Bash": "*", "Read": "*", "Write": "*", "Edit": "*" }
```

## Key Design Decisions

- **Tool definitions separate from implementations**: metadata list for prompts + TOOL_MAP for execution
- **Explicit `working_dir` parameter**: no global state for path sandboxing
- **Claude CLI tool name translation**: Python names (`call_claude`) → native names (`Bash`) via regex
- **Circuit breakers at two levels**: agent-level (error spirals, forced wrap-up) and graph-level (thread timeouts, dedup)
- **Reviewer (coder mode) has no write_file**: read + shell only for safety
- **Reviewer (research mode) has write_file**: needs to write `scoring_report.json`
- **Python 3.9 compatible**: uses `Optional[X]` not `X | None`, avoids walrus operator in hot paths
- **DuckDuckGo Lite for web_search**: no API key required, HTML scraping via urllib
- **Fallbacks at every stage**: decomposition, worker research, scoring, LaTeX generation all have Python fallbacks

## Known Limitations

- Workers writing output files: `claude -p` may not always write to the exact filename requested; files may appear under different names
- Nested `claude -p` calls: worker tasks instruct agents to use `Bash` for `claude -p`, causing nested invocations (2-3min each)
- DuckDuckGo scraping: regex-based HTML parsing is fragile to layout changes
- DeepSeek backend: `run_agent` uses JSON tool-call format which may not be as reliable as native tool calling
- Worker parallelism: max 4 concurrent to avoid overwhelming the API; no cancellation of already-running workers
