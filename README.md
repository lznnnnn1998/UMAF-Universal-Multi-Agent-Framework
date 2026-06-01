# UMAF — Universal Multi-Agent Framework

A modular, extensible multi-agent framework supporting autonomous code generation and AI-powered research with LLM-based review loops. Built on [LangChain](https://www.langchain.com/) + [LangGraph](https://langchain-ai.github.io/langgraph/).

## Overview

UMAF orchestrates multiple autonomous agents that collaborate through structured pipelines. Each agent reasons, uses tools, and produces verifiable output — mimicking the workflow of a human engineering or research team.

### Two Pipelines → Three Pipelines

| Pipeline | Command | Flow |
|----------|---------|------|
| **Coder** | `-m coder` | Coder implements → Reviewer reviews → loop until pass or max cycles |
| **Research** | `-m research` | Head decomposes topic → Workers research (dependency-ordered) → Reviewer scores → Writer generates LaTeX |
| **CoderPP** | `-m coderpp` | Multi-file code generation with organizer → workers → reviewer |

### Two LLM Backends

| Backend | Flag | Mechanism |
|---------|------|-----------|
| **DeepSeek API** | `-b deepseek` | LangChain `ChatOpenAI`, JSON tool-call loop with circuit breakers |
| **Claude CLI** | `-b claude_cli` | Subprocess `claude -p`, native tool calling, DeepSeek proxy |

## Architecture

```
main.py
├── pipeline.py (all pipelines — Coder, Research, CoderPP)
│   └── agent.py (autonomous agent loop + AgentRole ABC)
│       ├── llm.py (DeepSeek API / Claude CLI subprocess)
│       └── tools.py (7 tools + ToolRegistry: read, write, shell,
│                     web_search, web_fetch, download_file, call_claude)
│
├── research/
│   ├── head_agent.py (task decomposition, 2-8 sub-topics)
│   ├── worker_agent.py (dependency-ordered research, pre-fetch layer)
│   ├── reviewer_agent.py (5-dimension scoring & ranking)
│   └── writer.py (LaTeX generation)
│
└── coderpp/ (multi-file code generation pipeline agents)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure DeepSeek backend (default)
echo 'DEEPSEEK_API_KEY=your-key-here' > .env

# Or use Claude CLI backend (routes through DeepSeek proxy)
cp claude_env_sample.example.json claude_env_sample.json
# Edit claude_env_sample.json with your credentials

# Coder mode: generate code with review
python3 main.py -m coder "write a Python script to parse CSV files"

# Research mode: generate a LaTeX research proposal
python3 main.py -m research -b claude_cli "model quantization: QAT, PTQ, stochastic rounding"
```

### CLI

```
python3 main.py [--mode coder|research] [--backend deepseek|claude_cli] [--working-dir PATH] "requirement"
```

If no argument is given, reads from stdin or interactive prompt. Requires **Python >= 3.11**.

## Features

- **Autonomous tool use**: agents can read/write files, run shell commands, search the web, fetch URLs, download files, and delegate to nested Claude instances
- **Multi-agent review loops**: coder↔reviewer with up to 5 cycles; research pipeline with 5-dimension scoring (depth, accuracy, relevance, clarity, originality)
- **Dependency-aware execution**: workers run in topological order; stop-on-failure blocks downstream tasks; version-bump retries reuse full conversation context via checkpoints
- **Circuit breakers**: agent-level error spiral detection (3 consecutive persistent errors), force wrap-up at ≤3 steps remaining, mid-loop write reminders, version-bump context-reusing retries; graph-level thread timeouts, MD5 dedup, dependency stop-on-failure, worker retry state machine
- **Parallel workers**: research workers run concurrently via `ThreadPoolExecutor` (max 2) with individual timeouts (600s)
- **Graceful fallbacks**: every pipeline stage has a fallback — decomposition, research, scoring (auto-rank 25/50), LaTeX generation
- **Working directory sandboxing**: all file operations scoped to `research_output/` inside the repo (configurable)
- **Pre-fetch layer**: arxiv.org and academic content pre-downloaded at framework level for `claude_cli` workers, bypassing subprocess domain restrictions
- **Conversation logging**: every agent run logged to `agent_log/<name>_<timestamp>.json`
- **Backend-aware task generation**: `claude_cli` workers use native tools directly (no nested `claude -p`); `deepseek` workers use `call_claude` for reasoning

## Tools

| Tool | Description | Timeout |
|------|-------------|---------|
| `read_file(path)` | Read file contents | — |
| `write_file(path, content)` | Write file (creates parent dirs) | — |
| `run_command(command)` | Shell command | 30s |
| `call_claude(prompt)` | Delegate to Claude CLI subprocess | 120s |
| `web_search(query)` | DuckDuckGo Lite search (no API key) | 15s |
| `web_fetch(url)` | Fetch URL content via urllib (bypasses Claude Code permission checks) | 20s |
| `download_file(url, path)` | Download URL to local file via urllib (for arxiv.org PDFs/HTML) | 30s |

All tools accept `working_dir` for path sandboxing.

## Research Pipeline Details

```
Head Agent (120s timeout)
  └─ Decomposes topic → 2-8 sub-topics (dynamic, scaled by complexity; LLM or fallback)

Worker Agents (2 concurrent, 600s timeout each, dependency-ordered)
  └─ Pre-fetch: arxiv.org content downloaded at framework level (claude_cli backend)
  └─ Each researches one sub-topic → writes research_NN_Title.md
  └─ Dependency-aware: stop-on-failure blocks downstream dependents
  └─ Retry: version-bump retries reuse full context via checkpoints (max 3 retries)
  └─ Circuit breakers: mid-loop write reminders, force wrap-up, post-loop forced write

Reviewer Agent
  └─ Scores all outputs on 5 dimensions (depth, accuracy, relevance,
     clarity, originality, each 1-10) → writes scoring_report.json

Writer
  └─ Synthesizes top 3 into publication-quality LaTeX → research_proposal.tex
```

### Verified Results (v1.4, June 2026)

Topic: *"Propose a brand new optimized attention mechanism"*

- 7 sub-tasks generated by LLM (not fallback)
- **7/7 workers produced output files (100%)** — with dependency ordering and version-bump retries
- Top scores: **48/50**, 47/50, 45/50, 44/50, 43/50, 39/50, 38/50 (all real LLM scoring)
- Pipeline time: ~7.4 minutes (443s)
- Research files: 10-28 KB each; LaTeX: ~60KB (11 sections, 17 equations, 13 tables, 47 references)

### Verified Results (v1.3.1, May 2026)

Topic: *"model quantization: QAT, PTQ, stochastic rounding during training"*

- 4 sub-tasks generated by LLM (not fallback)
- **4/4 workers produced output files (100%)** — up from 2/4 in prior version
- Top scores: **47/50**, 46/50, 43/50, 41/50 (all real LLM scoring)
- Pipeline time: ~12 minutes
- Research files: 11-21 KB each; LaTeX output generated

## Configuration

### DeepSeek Backend
Set `DEEPSEEK_API_KEY` in `.env`.

### Claude CLI Backend
Requires the `claude` CLI installed (`npm install -g @anthropic-ai/claude-code`). Copy `claude_env_sample.example.json` to `claude_env_sample.json` and set credentials. Scope permissions in `.claude/settings.local.json`:

```json
{
  "permissions": {
    "Bash": "./research_output/**",
    "Read": "./research_output/**",
    "Write": "./research_output/**",
    "Edit": "./research_output/**"
  }
}
```

Set `CLAUDE_ENV_PATH` env var for a custom config path.

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| **v1.4** | Jun 2026 | Dependency stop-on-failure, version-bump retry with context reuse, honest parse_result, worker retry state machine, 3 pipelines, OOP refactoring (5-layer class hierarchy), 7/7 workers |
| **v1.3.1** | May 2026 | Tool-before-TASK_COMPLETE fix (4/4 workers), `download_file` + pre-fetch for arxiv access, default working dir in repo |
| **v1.3** | May 2026 | Python 3.11, dynamic decomposition (2-8 sub-topics), `web_fetch` tool, dead code removal, 8 unit tests |
| **v1.2** | May 2026 | Backend-aware agents (no nested `claude -p`), scoped permissions, conversation logger, timeout 120→300s |
| **v1.1** | May 2026 | 12 bug fixes: cwd sandboxing, parallel workers (4→2), tool name translation, dedup |
| **v1.0** | May 2026 | Initial release: two pipelines, two backends, five tools |

## Limitations

- **Filename deviation**: `claude -p` may write to slightly different filenames than requested (reviewer scans working dir to find outputs)
- **Worker timeouts**: complex research tasks may exceed 600s timeout (configurable in `ClaudeCLILLM`)
- **DeepSeek JSON parsing**: enforcing structured tool calls via JSON prompts is less reliable than native tool calling
- **Web search fragility**: DuckDuckGo Lite HTML scraping is regex-based and fragile to layout changes
- **No mid-execution cancellation**: already-running parallel workers can't be cancelled mid-execution

## Possible Optimizations

### Near-term

1. **DuckDuckGo → dedicated search API**: Replace HTML scraping with a proper search API (SerpAPI, Brave Search) for reliability and structured results.

2. **Caching**: Cache decomposition results and worker outputs keyed by topic hash. Repeated runs skip redundant work.

### Medium-term

4. **DeepSeek native tool calling**: Use LangChain's `bind_tools()` instead of JSON prompt engineering for the DeepSeek backend, matching Claude CLI reliability.

5. **Incremental LaTeX compilation**: Validate generated `.tex` files with `pdflatex --draftmode` and surface errors back to the writer for correction.

6. **Streaming output**: Stream agent progress to the CLI (currently all output arrives at pipeline completion).

### Longer-term

7. **Dynamic worker allocation**: Vary the number of workers based on topic complexity — already partially done at the decomposition level (2-8 sub-topics), but worker slots are fixed at 2.

8. **Reviewer feedback loop**: Feed reviewer critique back to workers for a second research pass before final scoring — analogous to the coder↔reviewer loop.

9. **Multi-model routing**: Use smaller/faster models for decomposition and larger models for deep research and scoring.

10. **Plugin system**: Let users register custom agent roles and pipeline topologies without modifying core framework code.

## Future Work

### Phase 3 — Production Hardening
- Comprehensive test suite (unit tests for each module, integration tests for full pipelines)
- Structured logging with trace IDs across agent boundaries
- Graceful shutdown: signal handling to clean up subprocesses on interrupt
- Configurable timeouts and retry policies via config file

### Phase 4 — New Pipeline Topologies
- **Debate mode**: two agents argue opposing positions, judge agent synthesizes consensus
- **Fact-check mode**: writer agent drafts, fact-checker verifies each claim against search results
- **Self-improving code**: coder writes tests first, implements, reviewer checks, coder refactors based on review

### Phase 5 — Observability & UI
- Web dashboard showing pipeline progress, agent transcripts, and scoring breakdowns
- Cost tracking: token usage and API call counts per pipeline run
- Export formats: PDF (via `pdflatex`), HTML, Markdown in addition to LaTeX

## Project Structure

```
.
├── main.py                  # CLI entry point
├── agent.py                 # Autonomous agent loop + AgentRole ABC + CheckpointManager
├── llm.py                   # DeepSeek + Claude CLI LLM backends
├── tools.py                 # 7-tool toolkit + ToolRegistry with 12 role-specific methods
├── pipeline.py              # All pipelines (Coder, Research, CoderPP) + BasePipeline
├── claude_config.py         # Claude CLI env injection
├── claude_env_sample.json   # Claude CLI credentials (git-ignored)
├── claude_env_sample.example.json  # Template for credentials
├── requirements.txt         # Python dependencies
├── CLAUDE.md                # Project documentation for AI assistants
├── .python-version          # Python 3.11
├── research/
│   ├── __init__.py
│   ├── head_agent.py        # Topic decomposition (2-8 sub-topics)
│   ├── worker_agent.py      # Dependency-ordered research + pre-fetch layer
│   ├── reviewer_agent.py    # 5-dimension scoring
│   └── writer.py            # LaTeX proposal generation
└── coderpp/
    ├── head_agent.py        # Multi-file code generation decomposition
    ├── worker_agent.py      # Code file generation
    ├── reviewer_agent.py    # Code review
    └── organizer.py         # Post-generation organization
```

## License

MIT