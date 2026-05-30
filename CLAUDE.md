# Universal Multi-Agent Framework (UMAF) v1.2

A LangChain + DeepSeek framework that mimics Claude Code's autonomous behavior with multi-agent collaboration. Supports two pipelines: **coder** (code generation with review loop) and **research** (head→workers→reviewer→writer for LaTeX research proposals). Both backends route through DeepSeek API — `claude_cli` uses the Anthropic-compatible proxy, `deepseek` uses the native API.

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
- Reads env vars from `claude_env_sample.json` via `claude_config.py` (falls back to `claude_env_sample.example.json`)
- Routes through DeepSeek's Anthropic-compatible API proxy
- 300s subprocess timeout, `--output-format text`, `--allowedTools` support
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

`run_agent(task, working_dir, tools, tool_map, max_steps, backend, agent_name)` runs an autonomous agent. Two code paths:

**DeepSeek path** (`_run_with_deepseek`):
- JSON tool-call loop: LLM returns `{"tool": "...", "args": {...}}`
- Brace-counting JSON parser handles nested objects/arrays in args
- Circuit breakers: force wrap-up at ≤3 steps remaining, error spiral detection at 3 consecutive persistent errors, unknown tool warnings

**Claude CLI path** (`_run_with_claude_cli`):
- Single `claude -p` invocation (the CLI itself is multi-turn with tool access)
- Tool name translation: Python names (`read_file`, `call_claude`, etc.) → Claude CLI native names (`Read`, `Bash`, etc.) via word-boundary regex
- Retry on timeout/error with simplified prompt
- Passes `cwd=working_dir` so files land in the correct directory
- `call_claude` spec discourages nested `claude -p` calls (the agent IS already Claude Code)

**Conversation Logger** (`_save_agent_log`):
- All agents save prompts, responses, success status, and elapsed time to `agent_logs/` under the working directory
- Log files named `<agent_name>_<timestamp>.json` for debugging
- Controlled by the `agent_name` parameter on `run_agent()` — pass descriptive names like `"head_decompose"`, `"worker_01"`, `"reviewer"`, `"writer"`

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
- Backend-aware: `claude_cli` uses Read-only tool set (pure reasoning, no web search needed — the model already knows the topics)
- `deepseek` backend uses DECOMPOSE_TOOLS_DEEPSEEK (run_command, call_claude, web_search)
- Extracts JSON array from LLM output; falls back to `_fallback_decompose()` on failure
- Fallback splits topic on commas/`and`/`vs` to extract keywords, generates 5+ specific sub-tasks

**`research/worker_agent.py`** — Research execution:
- **Backend-aware tasks** (v1.2): `claude_cli` workers use WebSearch + Write + Read directly — no nested `claude -p` calls. The agent IS already Claude Code.
- `deepseek` workers use `call_claude` for deep reasoning (the Python function shells out to `claude -p`)
- Task instructs agent to research, write `research_NN_Title.md`, verify with Read, and signal TASK_COMPLETE
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
- Workers: parallel execution via ThreadPoolExecutor (max 2 concurrent, v1.2), 300s timeout each
- Deduplication: MD5 fingerprint of summaries, marks duplicates post-hoc
- File verification: checks `os.path.exists()` and `os.path.getsize() > 0` after each worker
- Status flow: `decomposed → researched/researched_partial → reviewed → written → END`
- Router always moves forward; partial results accepted at every stage

```bash
python3 main.py -m research -b claude_cli --working-dir research_output "Flash Attention, Multi-Query Attention, Grouped Query Attention"
```

### `main.py` — Entry Point

```
python3 main.py [--mode coder|research] [--backend deepseek|claude_cli] [--working-dir PATH] "requirement"
```

Reads requirement from argument, stdin, or interactive prompt. Creates temp working directory.

### `claude_config.py` — Environment Setup

Loads `claude_env_sample.json` (12 env vars for Claude CLI subprocess routing). Falls back to `claude_env_sample.example.json` if the main config doesn't exist.

`merge_claude_env()` merges these with `os.environ` for subprocess calls.

## Setup

```bash
pip install -r requirements.txt

# Set DEEPSEEK_API_KEY in .env (for deepseek backend)
# Ensure claude CLI is installed: npm install -g @anthropic-ai/claude-code

# For claude_cli backend:
# 1. Copy the example config and add your API key:
cp claude_env_sample.example.json claude_env_sample.json
# Edit claude_env_sample.json: replace YOUR_DEEPSEEK_API_KEY with your real key

# 2. Configure .claude/settings.local.json with scoped permissions:
#   "permissions": {
#     "allow": [
#       "Bash(python3 *)",
#       "Bash(claude *)",
#       "Bash(curl *)",
#       "Bash(ls *)",
#       "Bash(cat *)",
#       "Bash(mkdir *)",
#       "Bash(wc *)",
#       "Read(/path/to/project/**)",
#       "Write(/path/to/project/**)",
#       "Edit(/path/to/project/**)",
#       "WebSearch(*)",
#       "WebFetch(*)"
#     ]
#   }
#
# Global ~/.claude/settings.json only needs:
#   "permissions": { "allow": ["Read(*)", "WebSearch(*)", "WebFetch(*)"] }
```

## Key Design Decisions

- **Tool definitions separate from implementations**: metadata list for prompts + TOOL_MAP for execution
- **Explicit `working_dir` parameter**: no global state for path sandboxing
- **Claude CLI tool name translation**: Python names (`call_claude`) → native names (`Bash`) via regex
- **Backend-aware task generation** (v1.2): `claude_cli` workers get different tasks than `deepseek` workers — no nested `claude -p` calls
- **Circuit breakers at two levels**: agent-level (error spirals, forced wrap-up) and graph-level (thread timeouts, dedup)
- **Reviewer (coder mode) has no write_file**: read + shell only for safety
- **Reviewer (research mode) has write_file**: needs to write `scoring_report.json`
- **Head agent for claude_cli uses Read-only tools** (v1.2): pure reasoning decomposition, no web search needed
- **Python 3.9 compatible**: uses `Optional[X]` not `X | None`, avoids walrus operator in hot paths
- **DuckDuckGo Lite for web_search**: no API key required, HTML scraping via urllib
- **Fallbacks at every stage**: decomposition, worker research, scoring, LaTeX generation all have Python fallbacks
- **Conversation logging** (v1.2): every agent saves prompt/response to `agent_logs/` for debugging

## Known Limitations

- Workers writing output files: `claude -p` may write to a slightly different filename than requested (v1.2 improved this but it's still not 100% reliable — graph verifies file existence)
- Worker timeouts: `claude -p` subprocess timeout is 300s; complex research tasks may need more time (use shorter topics or increase `ClaudeCLILLM.timeout`)
- DeepSeek backend: `run_agent` uses JSON tool-call format which may not be as reliable as native tool calling
- Worker parallelism: max 2 concurrent (v1.2 reduced from 4) to avoid overwhelming API and system resources
- DuckDuckGo scraping: regex-based HTML parsing is fragile to layout changes
- `claude -p` permissions: the subprocess needs matching permission patterns in `.claude/settings*` files scoped to its working directory; run `main.py` with `--working-dir` under the project root so the subprocess finds project `.claude/` settings

## v1.2 Changes (May 2026)

- **Backend-aware worker tasks**: `claude_cli` workers use WebSearch+Write+Read directly (no nested `claude -p`)
- **Head agent**: Read-only tool set for `claude_cli` — pure reasoning, ~70s decomposition
- **Conversation logger**: all agents save prompts/responses to `agent_logs/` (`agent_name` parameter on `run_agent`)
- **Permissions scoped**: Bash/Write/Read scoped to working directory (removed blanket `Bash(*)`/`Write(*)`/`Edit(*)` from global)
- **Timeout**: `ClaudeCLILLM` 120s→300s; worker parallelism 4→2; `--allowedTools` always passed (fixes empty-tools = all-tools bug)
- **Security**: `claude_env_sample.json` removed from git tracking, replaced with `.example.json` template; `claude_config.py` auto-falls back to example
- **System prompt**: removed "Use Bash for anything" encouragement; `call_claude` spec discourages nesting
- **Verified**: pipeline produces 4/6 worker research files (21-26KB each), top score 43/50, 41KB LaTeX
