# Universal Multi-Agent Framework (UMAF) v1.3.1

LangChain + DeepSeek multi-agent framework with two pipelines and two backends.

## Architecture

```
main.py → graph.py → agent.py → llm.py        (coder/reviewer)
              │          │
              ▼          ▼
        MultiAgentState tools.py (TOOL_MAP)

main.py → research/graph.py → research/head_agent.py      (research pipeline)
              │                 research/worker_agent.py
              ▼                 research/reviewer_agent.py
        ResearchState           research/writer.py
```

## Modules

### `llm.py` — Two backends
- **DeepSeek** (default): `ChatOpenAI` with `deepseek-chat`, temp=0.3, max_tokens=4096. Reads `DEEPSEEK_API_KEY` from `.env`.
- **Claude CLI**: `ClaudeCLILLM` shells out to `claude -p` subprocess (300s timeout). Injects env from `claude_env_sample.json`. Accepts `cwd` and `allowed_tools`.

Factory: `get_llm(backend)`.

### `tools.py` — Seven tools (all accept `working_dir`)
`read_file`, `write_file`, `run_command` (30s timeout), `call_claude` (120s, env-injected), `web_search` (DuckDuckGo Lite, no API key), `web_fetch` (urllib, 20s timeout), `download_file` (urllib, 30s timeout, saves to local file). Exported as `TOOL_MAP`.

### `agent.py` — Single agent loop
`run_agent(task, working_dir, tools, tool_map, max_steps, backend, agent_name)`:
- **DeepSeek path**: JSON tool-call loop with brace-counting parser. Circuit breakers: force wrap-up at ≤3 steps, error spiral detection (3 consecutive persistent errors), unknown tool warnings.
- **Claude CLI path**: Single `claude -p` invocation with tool name translation (word-boundary regex). Retries once on timeout/error.
- **Conversation logger**: `_save_agent_log()` writes to `agent_log/<name>_<timestamp>.json`.
- **Pre-fetch layer**: For `claude_cli` workers, arxiv.org content is pre-downloaded at the framework level (via `download_file`) before the agent runs, avoiding Claude Code's cc-switch domain verification.

### `graph.py` — Coder/Reviewer pipeline
Coder (all 6 tools) → Reviewer (no write_file). Max 5 cycles. Coder resets `review_passed=False` each run.

### `research/` — Research pipeline
```
head (decompose) → workers (parallel, max 2) → reviewer (score) → writer (LaTeX) → END
```
- **head_agent.py**: Backend-aware decomposition. Dynamically scales sub-topic count 2-8 based on topic complexity (narrow→2-3, moderate→4-5, broad→6-8). Falls back to `_fallback_decompose()`.
- **worker_agent.py**: Backend-aware tasks + pre-fetch layer. Claude CLI: framework pre-downloads arxiv.org content to local files (bypasses cc-switch domain verification), then agent reads locally with WebSearch+Write+Read. DeepSeek: uses `download_file`→`read_file`→`call_claude` pipeline.
- **reviewer_agent.py**: 5-dimension scoring (depth, accuracy, relevance, clarity, originality, each 1-10). Writes `scoring_report.json`.
- **writer.py**: LaTeX generation. Falls back to `_fallback_latex()` template. `_latex_escape()` handles all 10 special chars.
- **graph.py**: ThreadPoolExecutor with timeouts (head 120s, worker 300s). MD5 dedup. Router always moves forward.

### `main.py` — Entry point
```
python3 main.py [--mode coder|research] [--backend deepseek|claude_cli] [--working-dir PATH] "requirement"
```

### `claude_config.py` — Env setup
Loads `claude_env_sample.json` (12 env vars). Falls back to `.example.json`. `merge_claude_env()` merges with `os.environ`.

## Setup

```bash
pip install -r requirements.txt
# Set DEEPSEEK_API_KEY in .env
# For claude_cli: cp claude_env_sample.example.json claude_env_sample.json (edit API key)
# Scope permissions in .claude/settings.local.json
```

## Key Design Decisions

- Tool metadata + TOOL_MAP separation; explicit `working_dir` (no global state)
- Tool name translation for Claude CLI (Python names → native names via regex)
- Backend-aware task generation (v1.2): no nested `claude -p` for claude_cli workers
- Circuit breakers at agent level (error spirals, forced wrap-up) and graph level (timeouts, dedup, forced progress)
- **Python >= 3.11**: `X | None` syntax, no deprecated `Optional[X]` or `Union[X, Y]`
- Fallbacks at every stage; DuckDuckGo Lite (no API key); all agents logged for debugging

## Known Limitations

- `claude -p` may write to slightly different filenames than requested (graph verifies existence)
- Worker timeouts: complex tasks may exceed 300s (increase `ClaudeCLILLM.timeout`)
- DeepSeek JSON tool-call format less reliable than native tool calling
- DuckDuckGo scraping is regex-based and fragile to layout changes
- Subprocess needs `.claude/` settings scoped to its working directory
- Claude Code's cc-switch blocks arxiv.org domain verification → workaround: `download_file` pre-fetches content at framework level before agents run

## Version History

### v1.3.1 (May 2026) — Worker Output Fix & arxiv.org Access
- **Bug fix**: Reordered agent loop to execute tool calls BEFORE checking TASK_COMPLETE — responses containing both a `write_file` call and `TASK_COMPLETE` now execute the write first. Fixes missing worker output files.
- **Mid-loop write reminder**: Agent now gets nudged at ~2/3 of max steps if it hasn't called `write_file` yet.
- **Stronger force wrap-up**: Final steps explicitly forbid all tools except `write_file`; post-loop exhaustion message requires writing the file immediately.
- **`download_file` tool**: Framework-level urllib download → local file → `read_file`. Bypasses Claude Code's cc-switch domain verification for arxiv.org.
- **Pre-fetch layer**: `claude_cli` workers get arxiv.org content pre-downloaded at framework level before the agent runs.
- **Default working dir**: Changed from `tempfile.mkdtemp()` → `research_output/` inside repo; all logs now under `agent_log/`.
- **Verified**: 4/4 workers produce files (up from 2/4); scores 47, 46, 43, 41/50.

### v1.3 (May 2026) — Code Quality & Modernization
- **Python >= 3.11**: `Optional[X]` → `X | None` across all files; `.python-version` set to 3.11
- **Bug fix**: `_latex_escape()` backslash producing tab instead of `\textbackslash` (raw string fix)
- **Removed dead code**: unused `_TOOL_NAME_TRANSLATION` dict, `_build_system_prompt` dispatcher
- **Simplified**: `_run_with_claude_cli` retry (shared `_invoke`/`_build_prompt` helpers), head_agent prompt (shared `common` text), research router (flow dict instead of if-chain)
- All 8 unit tests pass; end-to-end coder pipeline verified

### v1.2 (May 2026) — Backend-Aware Agents
- Backend-aware worker tasks (no nested `claude -p`), head agent Read-only for claude_cli
- Conversation logger, scoped permissions, security cleanup (`.example.json` template)
- Timeout 120s→300s, parallelism 4→2, `--allowedTools` always passed
- Verified: 4/6 workers (21-26KB each), top score 43/50, 41KB LaTeX