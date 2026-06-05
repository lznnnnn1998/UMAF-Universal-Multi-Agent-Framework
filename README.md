# UMAF — Universal Multi-Agent Framework

A modular, extensible multi-agent framework supporting autonomous code generation, AI-powered research, topology optimization, and skill scanning. Built on [LangChain](https://www.langchain.com/) + [LangGraph](https://langchain-ai.github.io/langgraph/).

## Overview

UMAF orchestrates multiple autonomous agents that collaborate through structured pipelines. Each agent reasons, uses tools, and produces verifiable output — mimicking the workflow of a human engineering or research team.

### Six Pipelines

| Pipeline | Command | Flow |
|----------|---------|------|
| **Coder** | `-m coder` | Coder implements → Reviewer reviews → loop until pass or max cycles |
| **Research** | `-m research` | Head decomposes topic → Workers research (dependency-ordered) → Reviewer scores → Writer generates LaTeX |
| **CoderPP** | `-m coderpp` | Multi-file code generation with organizer → workers → reviewer |
| **Topology** | `-m topology` | Analyzer assesses task → Designer proposes topologies → Evaluator scores → Writer produces spec + report |
| **Skill** | `-m skill` | Scanner scans project → 4 parallel detectors → Aggregator deduplicates → Writer produces inventory + report |
| **Feature** | `-m feature` | Scanner analyzes project → Planner creates plan → Coder implements ↔ Reviewer reviews (max 5 cycles) → Writer produces report |

### Two LLM Backends

| Backend | Flag | Mechanism |
|---------|------|-----------|
| **DeepSeek API** | `-b deepseek` | LangChain `ChatOpenAI`, JSON tool-call loop with circuit breakers |
| **Claude CLI** | `-b claude_cli` | Subprocess `claude -p`, native tool calling, DeepSeek proxy |

## Architecture

```
main.py
├── pipeline/ (6 pipeline classes + BasePipeline)
│   ├── base.py (output dir mgmt, topological sort, parallel agents)
│   ├── coder.py, research.py, coderpp.py
│   ├── topology.py, skill.py, feature.py
│   └── agent.py (autonomous agent loop + AgentRole ABC + CheckpointManager)
│       ├── llm.py (DeepSeek API / Claude CLI subprocess)
│       └── tools/ (7 tools + ToolRegistry with 18+ role-specific methods)
│           ├── registry.py, functions.py, feature_tools.py
│
├── feature/ (5 roles)           ├── research/ (4 roles)
│   ├── scanner.py               │   ├── head_agent.py
│   ├── planner.py               │   ├── worker_agent.py
│   ├── coder.py                 │   ├── reviewer_agent.py
│   ├── reviewer.py              │   └── writer.py
│   └── writer.py                │
│                                ├── coderpp/ (4 roles)
├── topology/ (4 roles)          │   ├── head_agent.py
│   ├── analyzer.py              │   ├── worker_agent.py
│   ├── designer.py              │   ├── reviewer_agent.py
│   ├── evaluator.py             │   └── organizer.py
│   └── writer.py                │
│                                └── skill/ (4 roles)
└── utils.py                         ├── scanner.py
                                     ├── detectors.py
                                     ├── aggregator.py
                                     └── writer.py
```

23 AgentRole subclasses, 6 pipeline classes, 2 backends, 7 tools — all with deterministic fallbacks.

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

# Topology mode: design optimal agent topology for a task
python3 main.py -m topology -b claude_cli "multi-source data fusion with real-time validation"

# Skill mode: scan a project for skill inventory
python3 main.py -m skill -b claude_cli .
```

### CLI

```
python3 main.py [--mode coder|research|coderpp|topology|skill|feature] [--backend deepseek|claude_cli] [--working-dir PATH] "requirement"
```

Requires **Python >= 3.11**.

## Features

- **Autonomous tool use**: agents can read/write files, run shell commands, search the web, fetch URLs, download files, and delegate to nested Claude instances
- **Multi-agent review loops**: coder↔reviewer (max 5 cycles); 5-dimension research scoring (depth, accuracy, relevance, clarity, originality); topology evaluation (5 dimensions, sorted by total_score)
- **Dependency-aware execution**: topological ordering; stop-on-failure blocks downstream tasks; version-bump retries reuse full context via checkpoints
- **Circuit breakers**: error spiral detection (2 consecutive), force wrap-up at ≤3 steps, mid-loop write reminders, post-loop forced write; pipeline-level timeouts, MD5 dedup, worker retry state machine (max 3 retries, max 4 versions)
- **Parallel workers**: ThreadPoolExecutor (max 2) with individual 600s timeouts; 4 parallel detectors in Skill pipeline
- **Fan-out/fan-in topologies**: Skill Summarizer uses domain-parallel detection; Topology Optimizer can design arbitrary agent graphs (sequential, fan_out_fan_in, debate_consensus, hierarchical)
- **Graceful fallbacks**: every pipeline stage has a deterministic fallback — decomposition, research, scoring, LaTeX, topology analysis, design, evaluation, skill scanning, detection, aggregation
- **Working directory sandboxing**: all file operations scoped to output directories (configurable)
- **Pre-fetch layer**: arxiv.org and academic content pre-downloaded at framework level for `claude_cli` workers, bypassing subprocess domain restrictions
- **Conversation logging**: every agent run logged to `agent_log/<name>_<timestamp>.json`
- **Backend-aware task generation**: `claude_cli` workers use native tools directly (no nested `claude -p`); `deepseek` workers use `call_claude` for reasoning

## Tools

All tools are defined in `tools/registry.py` with implementations in `tools/functions.py`. `ToolRegistry` provides 18+ role-specific classmethods — no duplicated tool definitions.

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

## Pipeline Details

### CoderPipeline
```
Coder (all tools) ↔ Reviewer (no write_file) — max 5 cycles
```
Simple but effective: generate, review, iterate. Coder resets `review_passed=False` on each run to prevent stale-pass.

### ResearchPipeline
```
Head Agent (120s timeout)
  └─ Decomposes topic → 2-8 sub-topics (dynamic, LLM or fallback)

Worker Agents (2 concurrent, 600s timeout each, dependency-ordered)
  └─ Pre-fetch: arxiv.org content downloaded at framework level (claude_cli backend)
  └─ Each researches one sub-topic → writes research_NN_Title.md
  └─ Stop-on-failure blocks downstream dependents
  └─ Version-bump retries with context reuse via checkpoints (max 3 retries, max 4 versions)

Reviewer Agent → 5-dimension scoring (each 1-10) → scoring_report.json

Writer → synthesizes top 3 into LaTeX → research_proposal.tex
```

**Verified (v1.4):** 7/7 workers (100%); top score 48/50; 60KB LaTeX (11 sections, 17 equations, 13 tables, 47 references); 443s pipeline time.

### CoderPPPipeline
```
Organizer → Workers (parallel) → Reviewer — multi-file code generation
```
Reads `.md` and `.tex` spec files. Decomposes into modules, generates code per module, reviews, assembles.

### TopologyPipeline (v1.5)
```
Analyzer (6 complexity factors)
  └─ Designer (2-4 candidate topologies, 4 patterns)
      └─ Evaluator (5-dimension scoring, sorted by total_score)
          └─ Writer → topology_spec.json + topology_report.md
```

Designs optimal agent topology for any task description. Supports 4 patterns: sequential, fan_out_fan_in, debate_consensus, hierarchical. Evaluates on latency, reliability, cost_efficiency, simplicity, scalability.

**Verified:** Produced valid topology design for Skill Summarizer (fan-out/fan-in with domain-specific parallel detectors — an excellent architecture).

### SkillPipeline (v1.5)
```
Scanner (project scan → project_scan.json)
  └─ 4 parallel detectors (Python, JS, Infra, ConfigDocs)
      └─ Aggregator (deduplicate, categorize)
          └─ Writer → skills.json + skills_report.md
```

Scans a project directory and produces a structured skill inventory with proficiency levels (detected/used/extensively-used). Domain-parallel detection means each detector specializes in its ecosystem.

**Verified on this repo:** 33 skills across 11 categories — langchain (extensively-used), DeepSeek API (extensively-used), LangGraph (used), ThreadPoolExecutor (used), DuckDuckGo (used), argparse (used), subprocess (extensively-used), Claude CLI (extensively-used), urllib (extensively-used), pytest (detected), dataclasses (extensively-used), etc.

### FeaturePipeline (v1.6)
```
Scanner (project analysis) → Planner (implementation plan) → Coder ↔ Reviewer (max 5 cycles) → Writer (feature report) → END
```

Adds or modifies code in an existing project. The scanner analyzes the project structure and conventions, the planner creates an implementation plan with both `files_to_create` and `files_to_modify`, the coder implements changes, and the reviewer validates them. Falls back gracefully at every stage.

**Verified:** Created `prime_check.py` + tests in 1 iteration (review passed); created `palindrome.py` + tests in 1 iteration (claude_cli).

## Configuration

### DeepSeek Backend
Set `DEEPSEEK_API_KEY` in `.env`.

### Claude CLI Backend
Requires the `claude` CLI installed (`npm install -g @anthropic-ai/claude-code`). Copy `claude_env_sample.example.json` to `claude_env_sample.json` and set credentials. Scope permissions in `.claude/settings.local.json`:

```json
{
  "permissions": {
    "Bash": "./*_output/**",
    "Read": "./*_output/**",
    "Write": "./*_output/**",
    "Edit": "./*_output/**"
  }
}
```

Set `CLAUDE_ENV_PATH` env var for a custom config path.

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| **v1.6.1** | Jun 2026 | Dependency injection fixes: Coder (reviewer receives coder_files), Skill (upstream data passed to detectors/aggregator/writer), CoderPP (dependency_outputs injected in workers_node) |
| **v1.6** | Jun 2026 | Feature Pipeline, modular package structure (pipeline/, tools/, test/), 23 roles, 97 tests |
| **v1.5** | Jun 2026 | Topology Optimizer + Skill Summarizer pipelines (5 total), meta-programming via CoderPP, 18 AgentRoles, 42 tests |
| **v1.4.1** | Jun 2026 | 8 bug fixes: agent loop reorder, force wrap-up, checkpoint context, error spiral threshold, smoke tests |
| **v1.4** | Jun 2026 | OOP refactoring (5-layer hierarchy, AgentRole ABC, ToolRegistry), pipeline robustness (stop-on-failure, version-bump retry), CoderPP pipeline |
| **v1.3.1** | May 2026 | Tool-before-TASK_COMPLETE fix (4/4 workers), `download_file` + pre-fetch for arxiv access, default working dir in repo |
| **v1.3** | May 2026 | Python 3.11, dynamic decomposition (2-8 sub-topics), `web_fetch` tool, dead code removal, 8 unit tests |
| **v1.2** | May 2026 | Backend-aware agents (no nested `claude -p`), scoped permissions, conversation logger, timeout 120→300s |
| **v1.1** | May 2026 | 12 bug fixes: cwd sandboxing, parallel workers (4→2), tool name translation, dedup |
| **v1.0** | May 2026 | Initial release: two pipelines, two backends, five tools |

## Limitations

- **Filename deviation**: `claude -p` may write to slightly different filenames than requested (pipeline verifies existence)
- **Worker timeouts**: complex tasks may exceed 600s timeout (configurable in `ClaudeCLILLM`)
- **DeepSeek JSON parsing**: enforcing structured tool calls via JSON prompts is less reliable than native tool calling
- **Web search fragility**: DuckDuckGo Lite HTML scraping is regex-based and fragile to layout changes
- **No mid-execution cancellation**: already-running parallel workers can't be cancelled mid-execution
- **CoderPP worker hang**: workers can get stuck on TaskOutput framework calls when modifying pipeline.py (CoderPP works best for generating new agent role files, less so for pipeline integration)

## Possible Optimizations

### Near-term

1. **DuckDuckGo → dedicated search API**: Replace HTML scraping with a proper search API (SerpAPI, Brave Search) for reliability.

2. **Caching**: Cache decomposition results and worker outputs keyed by topic hash. Repeated runs skip redundant work.

### Medium-term

3. **DeepSeek native tool calling**: Use LangChain's `bind_tools()` instead of JSON prompt engineering for the DeepSeek backend.

4. **Incremental LaTeX compilation**: Validate generated `.tex` files with `pdflatex --draftmode` and surface errors back to the writer.

5. **Streaming output**: Stream agent progress to the CLI (currently all output arrives at pipeline completion).

### Longer-term

6. **Dynamic worker allocation**: Vary the number of workers based on topic complexity — already partially done at decomposition (2-8 sub-topics), but worker slots are fixed at 2.

7. **Reviewer feedback loop**: Feed reviewer critique back to workers for a second research pass before final scoring.

8. **Multi-model routing**: Use smaller/faster models for decomposition and larger models for deep research and scoring.

9. **Plugin system**: Let users register custom agent roles and pipeline topologies without modifying core framework code.

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
├── main.py                   # CLI entry point (6 modes)
├── agent.py                  # Autonomous agent loop + AgentRole ABC + CheckpointManager
├── llm.py                    # DeepSeek + Claude CLI LLM backends
├── utils.py                  # Shared helpers (extract_json_object, safe_read)
├── claude_config.py          # Claude CLI env injection
├── claude_env_sample.json    # Claude CLI credentials (git-ignored)
├── claude_env_sample.example.json  # Template for credentials
├── requirements.txt          # Python dependencies
├── CLAUDE.md                 # Project documentation for AI assistants
├── .python-version           # Python 3.11
├── pipeline/                 # Pipeline classes (7 files)
│   ├── __init__.py           # Re-exports all pipeline classes
│   ├── base.py               # BasePipeline + shared helpers (topological sort, parallel agents)
│   ├── coder.py              # CoderPipeline + CoderRole + ReviewerRole
│   ├── research.py           # ResearchPipeline (head → workers → reviewer → writer)
│   ├── coderpp.py            # CoderPPPipeline (head → workers → reviewer → organizer)
│   ├── topology.py           # TopologyPipeline (analyzer → designer → evaluator → writer)
│   ├── skill.py              # SkillPipeline (scanner → 4 detectors → aggregator → writer)
│   └── feature.py            # FeaturePipeline (scanner → planner → coder ↔ reviewer → writer)
├── tools/                    # Tool system (4 files)
│   ├── __init__.py           # Re-exports + auto-applies feature tools
│   ├── registry.py           # ToolSpec dataclass + ToolRegistry (18+ role methods)
│   ├── functions.py          # 7 tool implementations + TOOL_MAP
│   └── feature_tools.py      # Feature pipeline tool methods
├── feature/                  # Feature pipeline agent roles (5 files)
│   ├── scanner.py            # Project analysis → project_context.json
│   ├── planner.py            # Implementation plan with files_to_create + files_to_modify
│   ├── coder.py              # Creates and modifies files, writes tests
│   ├── reviewer.py           # REVIEW_PASSED/REVIEW_FAILED validation
│   └── writer.py             # Feature report generation
├── research/                 # Research pipeline agent roles (4 files)
│   ├── head_agent.py         # Topic decomposition (2-8 sub-topics)
│   ├── worker_agent.py       # Dependency-ordered research + pre-fetch layer
│   ├── reviewer_agent.py     # 5-dimension scoring
│   └── writer.py             # LaTeX proposal generation
├── coderpp/                  # CoderPP pipeline agent roles (4 files)
│   ├── head_agent.py         # Multi-file code generation decomposition
│   ├── worker_agent.py       # Code file generation
│   ├── reviewer_agent.py     # Code review
│   └── organizer.py          # Post-generation organization
├── topology/                 # Topology Optimizer agent roles (4 files, v1.5)
│   ├── analyzer.py           # Task complexity analysis (6 factors)
│   ├── designer.py           # Candidate topology generation (4 patterns)
│   ├── evaluator.py          # 5-dimension scoring, sorted ranking
│   └── writer.py             # topology_spec.json + topology_report.md
├── skill/                    # Skill Summarizer agent roles (4 files, v1.5)
│   ├── scanner.py            # Project directory scan → project_scan.json
│   ├── detectors.py          # 4 domain-specific detectors (Python, JS, Infra, ConfigDocs)
│   ├── aggregator.py         # Cross-domain skill dedup + categorization
│   └── writer.py             # skills.json + skills_report.md
└── test/                     # Test suite (4 files)
    ├── test_smoke.py         # 15 agent/pipeline core tests
    ├── test_topology.py      # 14 topology pipeline tests
    ├── test_skill.py         # 13 skill pipeline tests
    └── test_feature_v2.py    # 55 feature pipeline tests (97 total)
```

## License

MIT
