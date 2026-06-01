# Universal Multi-Agent Framework (UMAF) v1.4

LangChain + DeepSeek multi-agent framework with three pipelines and two backends. OOP architecture with 5-layer class hierarchy.

## Architecture

```
main.py → pipeline.py → agent.py → llm.py        (all pipelines)
              │               │          ├── ChatOpenAI (deepseek-chat)
              ▼               ▼          └── ClaudeCLILLM (subprocess)
        BasePipeline    AgentRole ABC
        ├── CoderPipeline     ├── CoderRole
        ├── ResearchPipeline  ├── ResearchWorkerRole
        └── CoderPPPipeline   ├── ResearchDecomposerRole
                              ├── ResearchReviewerRole
                              ├── WriterRole
                              └── ... (10 roles total)
```

## Modules

### `llm.py` — Two backends
- **DeepSeek** (default): `ChatOpenAI` with `deepseek-chat`, temp=0.3, max_tokens=4096. Reads `DEEPSEEK_API_KEY` from `.env`.
- **Claude CLI**: `ClaudeCLILLM` shells out to `claude -p` subprocess (300s timeout). Injects env from `claude_env_sample.json`. Accepts `cwd` and `allowed_tools`.

Factory: `get_llm(backend)`.

### `tools.py` — Seven tools + ToolRegistry
`read_file`, `write_file`, `run_command` (30s timeout), `call_claude` (120s, env-injected), `web_search` (DuckDuckGo Lite, no API key), `web_fetch` (urllib, 20s timeout), `download_file` (urllib, 30s timeout, saves to local file). `ToolRegistry` class centralizes tool specs with 12 role-specific methods — no duplicated tool definitions.

### `agent.py` — Agent core
- **`BaseAgent`**: Autonomous agent loop with circuit breakers (force wrap-up, error spiral detection, unknown tool warnings).
- **`AgentRole`** (ABC): Template method — `tools_for_backend()`, `build_task()`, `parse_result()`, `execute()`. Subclass for new agent types.
- **`CheckpointManager`**: Saves/loads agent state. `load_previous(version)` restores messages, resets iterations for context-reusing retries.
- **Conversation logger**: `_save_agent_log()` writes to `agent_log/<name>_<timestamp>.json`.
- **Pre-fetch layer**: For `claude_cli` workers, arxiv.org content is pre-downloaded at the framework level (via `download_file`) before the agent runs, avoiding Claude Code's cc-switch domain verification.

### `pipeline.py` — Three pipelines (replaces `graph.py`)
**`BasePipeline`**: Output dir management, double-check confirmation, `_topological_levels()`, `_run_workers_with_deps()`, `_run_parallel_agents()`.

**`CoderPipeline`**: Coder (all 6 tools) → Reviewer (no write_file). Max 5 cycles. Coder resets `review_passed=False` each run.

**`ResearchPipeline`**:
```
head (decompose) → workers (dependency-ordered) → reviewer (score) → writer (LaTeX) → END
```
- **head_agent.py**: Backend-aware decomposition. Dynamically scales sub-topic count 2-8 based on topic complexity. Falls back to `_fallback_decompose()`.
- **worker_agent.py**: Backend-aware tasks + pre-fetch layer. Dependency-ordered via `_topological_levels()`. Stop-on-failure blocks downstream tasks. Version-bump retry with context reuse via `CheckpointManager.load_previous()`. `parse_result()` checks `os.path.isfile()` before reporting success. Worker timeout 600s.
- **reviewer_agent.py**: 5-dimension scoring (depth, accuracy, relevance, clarity, originality, each 1-10). Writes `scoring_report.json`.
- **writer.py**: LaTeX generation. Falls back to `_fallback_latex()` template. `_latex_escape()` handles all 10 special chars.
- **Flow dict**: `decomposed → workers`, `worker_retry → workers` (version+1, failed only), `researched → reviewer`, `researched_partial → reviewer`, `reviewed → writer`, `written → END`.
- **Constants**: `HEAD_TIMEOUT=120`, `WORKER_TIMEOUT=600`, `RESEARCH_MAX_VERSIONS=4`, `RESEARCH_MAX_WORKER_RETRIES=3`.

**`CoderPPPipeline`**: Multi-file code generation with organizer → workers → reviewer.

### `main.py` — Entry point
```
python3 main.py [--mode coder|research|coderpp] [--backend deepseek|claude_cli] [--working-dir PATH] "requirement"
```

### `claude_config.py` — Env setup
Loads `claude_env_sample.json` (12 env vars). Falls back to `.example.json`. `merge_claude_env()` merges with `os.environ`.

### `research/` — Research pipeline agents
- `head_agent.py`: `ResearchDecomposerRole` — topic decomposition with dynamic scaling
- `worker_agent.py`: `ResearchWorkerRole` — sub-topic research + pre-fetch, `research_subtask()` entry point
- `reviewer_agent.py`: `ResearchReviewerRole` — 5-dimension scoring
- `writer.py`: `WriterRole` — LaTeX generation

### `coderpp/` — CoderPP pipeline agents
- `head_agent.py`, `worker_agent.py`, `reviewer_agent.py`, `organizer.py`

## Setup

```bash
pip install -r requirements.txt
# Set DEEPSEEK_API_KEY in .env
# For claude_cli: cp claude_env_sample.example.json claude_env_sample.json (edit API key)
# Scope permissions in .claude/settings.local.json
```

## Key Design Decisions

- OOP class hierarchy: Data types → Infrastructure → Agent core → Concrete roles → Pipeline classes
- `AgentRole` ABC + `ToolRegistry` centralization (no duplicated tool definitions)
- Tool metadata + TOOL_MAP separation; explicit `working_dir` (no global state)
- Tool name translation for Claude CLI (Python names → native names via regex)
- Backend-aware task generation (v1.2): no nested `claude -p` for claude_cli workers
- **Python >= 3.11**: `X | None` syntax, no deprecated `Optional[X]` or `Union[X, Y]`
- Fallbacks at every stage; DuckDuckGo Lite (no API key); all agents logged for debugging
- Dependency-aware execution (v1.4): stop-on-failure blocks downstream workers, version-bump retries reuse context via checkpoints
- Router always moves forward: `researched_partial` accepted at reviewer stage

## Known Limitations

- `claude -p` may write to slightly different filenames than requested (graph verifies existence)
- Worker timeouts: complex tasks may exceed 600s (increase `ClaudeCLILLM.timeout`)
- DeepSeek JSON tool-call format less reliable than native tool calling
- DuckDuckGo scraping is regex-based and fragile to layout changes
- Subprocess needs `.claude/` settings scoped to its working directory
- Claude Code's cc-switch blocks arxiv.org domain verification → workaround: `download_file` pre-fetches content at framework level before agents run

## Version History

### v1.4 (June 2026) — Pipeline Robustness & Dependency Management
- **Stop-on-failure**: `_run_workers_with_deps` breaks out of topological level loop when a level has failures, blocking downstream dependents.
- **Version-bump retry with context reuse**: Failed workers retry with `version+1` → `CheckpointManager.load_previous(version)` restores messages, resets iterations, injects retry context.
- **Honest `parse_result`**: `ResearchWorkerRole.parse_result()` checks `os.path.isfile()` before reporting `output_file` — missing files correctly count as failure.
- **Worker retry state machine**: `worker_retry` status in research flow dict, max 3 retries (`RESEARCH_MAX_WORKER_RETRIES`), max 4 versions (`RESEARCH_MAX_VERSIONS`).
- **Timeout**: Worker timeout 300s → 600s for complex attention mechanism derivations.
- **Cleanup**: `graph.py` removed (dead code, replaced by `pipeline.py`). `.gitignore` updated with agent_log, JSON output, and coderpp patterns.
- **Verified**: 7/7 workers produce output (100%); scores 48, 47, 45, 44, 43, 39, 38/50; 60KB LaTeX; 443s pipeline time.

### v1.3.1 (May 2026) — Worker Output Fix & arxiv.org Access
- **Bug fix**: Reordered agent loop to execute tool calls BEFORE checking TASK_COMPLETE — responses containing both a `write_file` call and `TASK_COMPLETE` now execute the write first. Fixes missing worker output files.
- **Mid-loop write reminder**: Agent now gets nudged at ~2/3 of max steps if it hasn't called `write_file` yet.
- **Stronger force wrap-up**: Final steps explicitly forbid all tools except `write_file`; post-loop exhaustion message requires writing the file immediately.
- **`download_file` tool**: Framework-level urllib download → local file → `read_file`. Bypasses Claude Code's cc-switch domain verification for arxiv.org.
- **Pre-fetch layer**: `claude_cli` workers get arxiv.org content pre-downloaded at framework level before the agent runs.
- **Default working dir**: Changed from `tempfile.mkdtemp()` → `research_output/` inside repo; all logs now under `xxxx_output/agent_log/`.
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
