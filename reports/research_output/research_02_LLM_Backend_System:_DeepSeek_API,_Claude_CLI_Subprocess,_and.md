# LLM Backend System: DeepSeek API, Claude CLI Subprocess, and Backend-Aware Task Generation

## Overview

UMAF's LLM layer implements a dual-backend architecture that provides transparent access to two fundamentally different language model invocation paradigms: a traditional API-based approach via DeepSeek's OpenAI-compatible endpoint, and a subprocess-based approach that shells out to the Claude Code CLI (`claude -p`). This design is motivated by the observation that API-based models (DeepSeek via `ChatOpenAI`) offer cost efficiency and deterministic JSON tool-call parsing but struggle with complex multi-step reasoning and require extensive prompt engineering for reliable structured output, while the Claude Code CLI provides native tool calling, superior multi-step task execution, and stream-json event processing at the cost of higher latency, per-call overhead, and subprocess management complexity.

The architecture unifies both backends behind a common `LLMProvider` abstract base class (`llm.py:18`) with a single `invoke(messages) -> AIMessage` interface. A factory function `get_llm(backend)` returns the appropriate concrete instance, enabling pipelines and agents to swap backends at runtime without code changes. This factory pattern, combined with the `AgentRole` Template Method where `build_task(backend)` receives the backend identifier, enables **backend-aware task generation** — the same role produces different prompts for DeepSeek (emphasizing JSON tool-call formatting, escape rules, and specific write strategies) versus Claude CLI (leveraging native tool names, stream-json output, and bypassPermissions mode). Critically, when the Claude CLI backend is active, agents are instructed to use their own reasoning directly rather than spawning nested `claude -p` subprocesses, avoiding the exponential cost and latency of recursive invocation.

Beyond the core invoke mechanism, the LLM backend system encompasses three supporting subsystems: (1) a **pre-fetch layer** that downloads arxiv.org and other academic content at the framework level (via Python urllib) before agents run, bypassing Claude Code's cc-switch domain verification; (2) a **CheckpointManager** providing version-aware, file-based state persistence with incremental saves and cross-version context reuse for retry scenarios; and (3) a **conversation logger** that writes timestamped JSON logs to `agent_log/` for debugging, auditing, and pipeline-wide merge-on-completion. These subsystems collectively address the reliability and observability gaps inherent in running autonomous LLM agent loops over extended durations.

## Key Methods & Approaches

### 1. Dual-Backend Architecture: DeepSeekProvider and ClaudeCLILLM

UMAF's backend architecture follows a clean Strategy + Factory pattern pairing:

**Abstract Base — `LLMProvider` (`llm.py:18`):** Defines a single abstract method `invoke(messages: list[BaseMessage], **kwargs) -> AIMessage` that both concrete implementations must satisfy. This minimal interface ensures that the rest of the framework (notably `BaseAgent.run()`) operates identically regardless of which backend is active.

**DeepSeek Backend — `DeepSeekProvider` (`llm.py:28`):**
- Wraps LangChain's `ChatOpenAI` pointed at `https://api.deepseek.com/v1` with model `deepseek-chat`.
- Configuration: `temperature=0.3` (balanced determinism vs. creativity), `max_tokens=8192` (sufficient for multi-page code generation and research synthesis; configured as `max_tokens` internally, though CLAUDE.md notes 4096 for historical reasons — the actual code uses 8192).
- API key resolved from `DEEPSEEK_API_KEY` environment variable via `python-dotenv` (`load_dotenv()`).
- The `invoke()` method delegates directly to LangChain's `ChatOpenAI.invoke()`, which handles OpenAI-compatible request/response serialization, retry, and error mapping.
- A module-level singleton `deepseek = DeepSeekProvider()._llm` is maintained for backward compatibility, exposing the raw `ChatOpenAI` instance for code that predates the `LLMProvider` abstraction.

**Claude CLI Backend — `ClaudeCLILLM` (`llm.py:67`):**
- Constructs a `claude -p <prompt>` subprocess command with `--output-format stream-json`, `--verbose`, and `--permission-mode bypassPermissions`.
- **Timeout**: Default 600s (increased from 300s in v1.4). The `BaseAgent._run_claude_cli()` method additionally enforces a hard timeout via `threading.Timer` that kills the subprocess if it exceeds the threshold during long-running tool executions.
- **Environment injection**: Calls `merge_claude_env()` which merges the current process environment with variables defined in `claude_env_sample.json` (12 env vars for API keys, proxy settings, model configuration). This is critical because the `claude` CLI reads its API key and endpoint configuration from environment variables.
- **Working directory**: Accepts `cwd` parameter via kwargs, passed to `subprocess.Popen` / `subprocess.run` so file operations resolve relative to the pipeline's working directory.
- **Allowed tools**: Accepts `allowed_tools: list[str]` via kwargs, appended as `--allowedTools tool1,tool2,...` to the CLI command. This enables per-role tool restriction at the CLI level, enforced by Claude Code's own permission model.

**Two invocation modes:**
1. `invoke()` (`llm.py:81`): Text output mode (`--output-format text`). Used by `call_claude` tool and backward-compatible codepaths. Returns the complete response as a single `AIMessage`.
2. `stream_invoke()` (`llm.py:115`): Stream-json output mode. Launches the subprocess with `subprocess.Popen`, reads NDJSON lines from stdout, yields parsed event dicts (`assistant`, `user`, `result` types). Used by `BaseAgent._run_claude_cli()` for incremental processing and checkpointing.

**Static helper — `_messages_to_prompt()` (`llm.py:57`):** Converts LangChain message objects (`SystemMessage`, `HumanMessage`, `AIMessage`) into a flat prompt string with `[ROLE]` headers. This is necessary because the `claude -p` CLI accepts a single prompt string, not a structured message array — unlike the DeepSeek API which preserves the message structure through LangChain.

### 2. Factory Pattern: `get_llm()` and `get_llm_provider()`

UMAF provides two factory functions for backend instantiation:

**`get_llm(backend="deepseek")` (`llm.py:164`):** Returns a pre-instantiated, module-level singleton — either the raw `ChatOpenAI` instance (`deepseek`) or the `ClaudeCLILLM` singleton (`_claude_cli_instance`). This is the backward-compatible entry point used by `BaseAgent._run_deepseek()` and `BaseAgent._run_claude_cli()`.

**`get_llm_provider(backend="deepseek")` (`llm.py:175`):** Returns a **fresh** `LLMProvider` instance — either a new `DeepSeekProvider()` or a new `ClaudeCLILLM()`. This is used by `AgentRole.execute()` (line 1127 of `agent.py`) to create a provider for checking backend capabilities. Each call creates a new instance, avoiding state leakage between concurrent agent executions.

The factory design follows the established pattern seen across the broader LLM ecosystem (e.g., `LiteObject/llm-provider-abstraction`, `kubectl-ai`, `agent-orchestrator`), where a registry or switch dispatches to concrete implementations. UMAF's approach is deliberately simple — a two-branch conditional rather than a full registry — because only two backends are supported and the interface is minimal (single `invoke()` method). Adding a third backend (e.g., Anthropic API, Ollama) would require only adding a new concrete class and a new branch in the factory.

### 3. Backend-Aware Task Generation

This is one of UMAF's most important architectural innovations. Rather than generating a single, backend-agnostic prompt and hoping the LLM adapts, every `AgentRole.build_task(backend, ...)` receives the backend identifier and tailors the prompt accordingly. This exists at three layers:

**Layer 1 — `BaseAgent` Prompt Builders:** `BaseAgent._build_deepseek_prompt()` (`agent.py:318`) and `BaseAgent._build_claude_cli_prompt()` (`agent.py:365`) produce fundamentally different system prompts:
- **DeepSeek prompt** includes: 311 lines of JSON tool-call formatting rules (exact JSON object format, 4 escaping rules, write strategy hierarchy preferring `run_command` Python one-liners over `write_file` to avoid multi-line string escaping, 4 repair strategies for malformed JSON). This is necessary because DeepSeek does not support native tool calling via its ChatOpenAI-compatible API — UMAF must parse JSON tool calls from the model's text output.
- **Claude CLI prompt** includes: 30 lines of native tool instructions (mapping Python tool names to Claude CLI native names, instructing the model to use its native tool interface rather than outputting JSON). The prompt explicitly states "DO NOT output JSON tool call blocks. Use your native tool calling instead."

**Layer 2 — `AgentRole.build_task()` Per-Backend Branching:** Each concrete role's `build_task(backend, ...)` method checks the backend and forks:
- For **DeepSeek**: Emphasizes `call_claude` for deep reasoning subtasks (since DeepSeek's own reasoning through JSON tool-call loops is less reliable), uses `download_file` followed by `read_file` for arxiv.org access, and provides tool-specific instruction blocks that only mention actually-available tools.
- For **Claude CLI**: The agent IS a Claude Code instance, so there is no need for nested `claude -p` calls. The `call_claude` tool is mapped to a `Bash` instruction that says "you ARE the reasoning engine — do NOT spawn nested claude -p calls." Web search uses `WebSearch`, file operations use `Read`/`Write`, and web fetch uses `Bash` with inline Python urllib one-liners.

**Layer 3 — `BaseDecomposerRole._backend_instructions()`:** Head agents (research decomposer, coderpp decomposer) append backend-specific suffixes. For `claude_cli`, agents are told to "Use your own knowledge — do NOT search the web" and to read `.tex` files for decomposition ideas. For `deepseek`, they receive web search instructions and arxiv.org pre-fetch patterns.

**Tool name translation:** `BaseAgent._translate_task_for_claude()` (`agent.py:395`) uses regex substitution to replace Python tool names in the task description with Claude CLI native names (e.g., `write_file` → `Write`, `run_command` → `Bash`, `web_search` → `WebSearch`). This ensures the task itself references tools the CLI can understand.

**The nested `claude -p` problem:** Early versions of UMAF (pre-v1.2) suffered from exponential cost blowup — when DeepSeek was instructed to use `call_claude` for reasoning heavy tasks, and the Claude CLI backend was active, a single research worker could spawn multiple nested `claude -p` processes, each of which might itself attempt to call `claude -p`. v1.2 solved this by making task generation backend-aware: Claude CLI workers receive prompts that explicitly say "You ARE the reasoning engine — do NOT spawn nested claude -p calls," and the `call_claude` tool is translated to `Bash` with this instruction baked in.

### 4. The Pre-Fetch Layer: Bypassing cc-switch Domain Verification

A critical limitation of Claude Code's subprocess mode is that the `cc-switch` layer blocks domain verification for arxiv.org and certain other academic domains. When a Claude Code agent attempts to fetch `arxiv.org/abs/XXXX.XXXXX`, the network sandbox rejects the request — the domain cannot be verified through cc-switch's verification mechanism.

UMAF's pre-fetch layer (`research/worker_agent.py:232-275`) solves this by downloading academic content **at the framework level** (via Python `urllib` directly — outside Claude Code's sandbox) **before** the agent runs. The mechanism works as follows:

1. **Search phase**: `_prefetch_arxiv_sources()` calls `web_search()` with the sub-task's title and description (truncated to 200 chars), retrieving up to 8 search results.
2. **URL extraction**: Regex extracts `arxiv.org` URLs (`arxiv\.org/\S+`) and other academic URLs (`openreview.net`, `proceedings.mlr.press`, `papers.nips.cc`) from search results.
3. **Download phase**: Up to 3 academic URLs are downloaded via `download_file()` (which uses `urllib.request.urlopen`) to local HTML files in `agent_log/prefetched_NN_<name>.html`.
4. **Prompt injection**: The downloaded file paths are injected into the Claude CLI worker's prompt under a "Pre-downloaded Reference Material" section with explicit instructions: "Read them directly — do NOT try to fetch them again from the web."
5. **DeepSeek parallel**: For DeepSeek workers, the prompt instead instructs the agent to use `download_file` followed by `read_file` as a two-step pattern, since DeepSeek's `download_file` tool also runs at the framework level (in the tool execution function, not inside a subprocess sandbox).

This pattern represents a general solution to the "sandboxed agent can't access domain X" problem: pre-fetch at the orchestrator level and pass the content as local file references. It's applicable to any domain blocked by Claude Code's network verification, and the framework could be extended to pre-fetch from user-specified URL lists or domain patterns.

### 5. CheckpointManager: Versioned State Persistence and Cross-Version Context Reuse

The `CheckpointManager` (`agent.py:25-236`) provides the persistence backbone for UMAF's autonomous agent loop, enabling agents to survive failures, resume from intermediate states, and retain reasoning context across retry attempts.

**File patterns** under `{working_dir}/agent_log/`:
- `{safe_name}_v{version:02d}_checkpoint.json` — Per-version checkpoint
- `{safe_name}_merged.json` — Consolidated multi-version audit trail
- `{safe_name}_{timestamp}.json` — Conversation log

**Core operations:**

**`save(version, messages, iterations, max_steps, has_written_output, task, tools, extra)` (`agent.py:52`):** Serializes the agent's complete state — all messages (type-annotated via `serialize_messages()`), iteration count, step budget, write tracking flag, task string, tool specs, timestamp, and arbitrary extra metadata. Messages are serialized as `[{type: "SystemMessage|HumanMessage|AIMessage", content: "..."}]` for JSON-safe storage. Writes to the versioned file path.

**`load(version)` (`agent.py:77`):** Deserializes a checkpoint file back into a dict with reconstructed LangChain message objects (`SystemMessage`, `HumanMessage`, `AIMessage`). Includes a legacy migration path: if a versioned file doesn't exist, it checks for an older unversioned `_checkpoint.json` and renames it.

**`load_previous(current_version)` (`agent.py:88`):** The primary resume path. When an agent retries at version V, it loads version V-1 to recover prior reasoning. Lists all available versions (`list_versions()`), finds the highest below `current_version`, and returns it. This enables the "version-bump retry" pattern in ResearchPipeline and CoderPPPipeline: a failed worker is re-executed at version+1 with full message context from the previous attempt, plus a system message explaining this is a retry ("Review what went wrong in the previous version and improve").

**`merge()` (`agent.py:114`):** At pipeline completion, all version checkpoints for an agent are merged into a single `_merged.json` file. Uses MD5 hashing (`{type}{content[:200]}`) to deduplicate messages across versions while preserving the full timeline via `version_summary` — listing each version's iteration count, message count, new messages contributed, timestamp, and write status. This provides a compact audit trail while keeping the original per-version files for debugging.

**Checkpoint granularity differs by backend:**
- **DeepSeek**: Checkpoint saved after every tool execution (`agent.py:693-694` in `_execute_tool()`). This means the checkpoint captures the state after each LLM response + tool result pair, providing fine-grained recovery within the agent loop.
- **Claude CLI**: Checkpoint saved after every stream-json event (`agent.py:881-886` inline `_flush_ckpt()` function) — after each `assistant` message (text + tool calls) and each `user` message (tool results). This enables sub-step recovery even within a single long-running `claude -p` invocation, which is critical given the 600s timeout.

**Context reuse for retry (`agent.py:570-589`):** When `version > 1` and checkpointing is enabled, `_run_deepseek()`'s message initialization:
1. Loads the previous version's messages via `load_previous()`
2. Resets `iterations` to 0 (fresh step budget — avoids the agent immediately hitting max_steps)
3. Inserts a context injection message: "This is version {V} retry. Review what went wrong in the previous version and improve."
4. Ensures the system prompt is at position [0]

This design is important because it preserves the agent's prior reasoning and tool results while giving it a clean slate for the retry — the agent doesn't redo completed work but can learn from what went wrong.

**Integration with the broader durability landscape:** The checkpointing literature reveals a critical distinction: **checkpointing ≠ durable execution** (Diagrid, 2026). UMAF's `CheckpointManager` provides the storage half (save/load/merge) but delegates the orchestration half (auto-resume on failure, duplicate prevention, partial-work rollback) to the pipeline layer — specifically `_run_workers_with_deps()` which implements stop-on-failure, version-bump retry with explicit load_previous calls, and topological dependency management. This split is architecturally appropriate for a framework that orchestrates finite-duration agent runs rather than long-lived, always-on agents.

### 6. Conversation Logger: Debugging and Audit Trail

The conversation logger (`agent.py:1073-1080`, `CheckpointManager.save_log()` at line 170) writes timestamped JSON logs for every agent execution:

**`save_log(task, elapsed, success, messages, extra)` (`agent.py:170`):** Creates `{agent_name}_{timestamp}.json` in `agent_log/` with:
- `agent`: The agent's name (e.g., `worker_03`, `head`, `reviewer`)
- `timestamp`: `YYYYMMDD_HHMMSS` format
- `elapsed_seconds`: Wall-clock duration rounded to 0.1s
- `success`: Boolean indicating whether TASK_COMPLETE was detected
- `prompt`: The full task string
- `response`: The last AIMessage content (up to 10,000 chars), extracted by scanning messages in reverse order to find the final substantive agent response
- `extra`: Arbitrary metadata dict (backend type, retry flag, version, post-loop forced write flag)

DeepSeek logs include `extra={"success": true/false}`. Claude CLI logs include `extra={"backend": "claude_cli_stream", "retried": true/false}` — annotating whether the automatic retry-on-error mechanism was triggered.

These logs serve three purposes:
1. **Debugging**: When a pipeline produces unexpected results, individual agent logs show exactly what the LLM was asked and what it produced.
2. **Performance analysis**: Timestamps and elapsed times enable identification of slow agents and backend-specific latency patterns.
3. **Self-evolution feedback**: The SelfEvolutionPipeline's analyzer reads `agent_log/` to identify patterns of failure and improvement opportunities.

### 7. Backend Trade-offs: DeepSeek vs. Claude CLI

Based on the codebase analysis and external benchmarks, the trade-offs between the two backends are substantial:

| Dimension | DeepSeek (ChatOpenAI) | Claude CLI (Subprocess) |
|-----------|----------------------|------------------------|
| **Tool Calling Mechanism** | JSON parsing from text output — requires 4-strategy parser (markdown fences → standard order → reversed order → JSON repair with state-machine whitespace escaping). Prone to malformed JSON, especially with multi-line code strings. | Native tool calling via `stream-json` events — tools are first-class, type-safe, and validated by the CLI runtime. No JSON parsing errors. |
| **Tool Calling Reliability** | Moderate — requires extensive prompt engineering (311-line system prompt vs. 30-line for Claude CLI), write strategy hierarchy, and multi-tier JSON repair. Even with these, DeepSeek's JSON tool-call format is "less reliable than native tool calling" (CLAUDE.md). | High — tool calls are structurally validated by the CLI. The model uses its native tool interface, which has been explicitly trained for tool use. |
| **Multi-Step Task Execution** | Gets stuck in loops, loses state across steps, requires heavy scaffolding (external benchmarks confirm DeepSeek "tends to get stuck in loops, call the wrong tool, or lose state across steps" on 3+ sequential calls). | Maintains coherent task state across longer contexts, handles ambiguous tool outputs gracefully, fewer execution failures. |
| **Cost** | ~$0.28/M input, ~$0.42/M output (DeepSeek V3.2 pricing, early 2026). Approximately 10-15× cheaper than Claude at scale. | Higher — each subprocess invocation incurs Claude API costs at ~$3/M input, ~$15/M output (Sonnet tier), plus the overhead of full system prompt + tool descriptions on each invocation. |
| **Latency** | Fast — single API call per iteration, no subprocess overhead. | Slower — subprocess startup, stream-json overhead, tool execution time within the subprocess. The 600s timeout is frequently reached for complex tasks. |
| **Circuit Breakers** | Full intervention system: force wrap-up at max_steps-3, error spiral detection (3 consecutive persistent errors), unknown tool warnings, write reminders at max_steps-4, post-loop forced write (2 additional LLM calls to salvage uncompleted tasks). | Limited to hard timeout via `threading.Timer` and one automatic retry on error/timeout. The CLI manages its own internal iteration, so the framework has less control over when to force wrap-up. |
| **Checkpoint Granularity** | After every tool execution — fine-grained state capture within the agent loop. | After every stream-json event — even finer granularity, capturing mid-turn tool calls and results within a single subprocess invocation. |
| **Tool Restriction** | At the prompt level — tools are declared in the system prompt, and the agent can attempt any listed tool. Restrictions enforced through prompt instructions only. | At the CLI level — `--allowedTools` flag restricts which tools the subprocess can actually use. Hard enforcement by Claude Code runtime. |
| **Environment Isolation** | Single process — the agent runs in the same Python process as the framework. No subprocess isolation. | Subprocess isolation — the `claude` CLI runs in its own process with injected environment variables. If the CLI crashes, the framework continues. |
| **Error Recovery** | JSON parse failures trigger automatic retry with diagnostic feedback (error position, snippet, common causes). Failed tool calls are caught and reported. | Timeout/error triggers one automatic retry with a shortened prompt ("skip time-consuming steps"). Stream-json stderr is captured and appended to the response for diagnosis. |
| **Observability** | Complete message history in checkpoint files — every LLM response and tool result is recorded. | Stream-json events are recorded incrementally (text extracted as AIMessage, tool calls as HumanMessage annotations, tool results as HumanMessage with content suffix). Stderr captured in temp file and appended on error. |

**When each backend is appropriate:**

- **DeepSeek**: High-volume, cost-sensitive workloads; short agent chains (≤3 tool calls); tasks with well-defined schemas and predictable inputs; development and testing phases where fast iteration matters; when the framework needs fine-grained control over the agent loop (circuit breakers, write reminders).

- **Claude CLI**: Complex multi-step tasks (5+ sequential tool calls); research tasks requiring deep reasoning, judgment, and ambiguity handling; code generation where tool calling fidelity directly impacts output quality; when the task involves reading many files and making context-dependent decisions; production pipelines where failed runs have meaningful consequences.

- **Hybrid** (current UMAF default, configurable per-pipeline): Use DeepSeek for simple roles (decomposer, reviewer, writer — typically ≤5 tool calls) and Claude CLI for complex roles (research workers, coder agents). The `tools_config.json` per-role tool assignments and `max_steps` settings already enable this specialization.

### 8. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      AgentRole.execute()                     │
│  ┌─────────────────────┐    ┌────────────────────────────┐  │
│  │  build_task(backend) │    │  tools_for_backend(backend) │  │
│  │  - DeepSeek prompt   │    │  - DeepSeek: JSON tool dicts│  │
│  │  - Claude CLI prompt │    │  - Claude CLI: native tools │  │
│  └──────┬──────────────┘    └──────────┬─────────────────┘  │
│         │                              │                     │
│         ▼                              ▼                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   BaseAgent                           │   │
│  │  ┌─────────────────┐  ┌──────────────────────────┐   │   │
│  │  │ _run_deepseek() │  │    _run_claude_cli()     │   │   │
│  │  │ - JSON parse    │  │  - stream-json events    │   │   │
│  │  │ - 4 strategies  │  │  - threading.Timer       │   │   │
│  │  │ - circuit break │  │  - auto-retry on error   │   │   │
│  │  │ - tool execution│  │  - incremental ckpt      │   │   │
│  │  └────────┬────────┘  └──────────┬───────────────┘   │   │
│  │           │                      │                    │   │
│  └───────────┼──────────────────────┼────────────────────┘   │
│              │                      │                        │
│              ▼                      ▼                        │
│  ┌──────────────────────┐  ┌──────────────────────────┐     │
│  │   DeepSeekProvider   │  │     ClaudeCLILLM         │     │
│  │   ChatOpenAI wrapper │  │  subprocess: claude -p   │     │
│  │   api.deepseek.com   │  │  --output-format stream-  │     │
│  │   temp=0.3, max=8192 │  │  json --verbose           │     │
│  └──────────────────────┘  └──────────────────────────┘     │
│                                                              │
│  Supporting Systems:                                         │
│  ┌──────────────┐ ┌────────────────┐ ┌───────────────────┐  │
│  │CheckpointMgr │ │ConversationLog │ │  Pre-fetch Layer  │  │
│  │versioned .json│ │agent_log/*.json│ │  urllib arxiv.org │  │
│  │load_previous()│ │save_log()      │ │  → local .html    │  │
│  │merge()        │ │                │ │  → prompt inject  │  │
│  └──────────────┘ └────────────────┘ └───────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 9. JSON Tool-Call Parsing Pipeline (DeepSeek-Specific)

Since DeepSeek does not support native tool calling through the ChatOpenAI-compatible API, UMAF implements a sophisticated multi-strategy JSON parser in `BaseAgent._parse_tool_call()` (`agent.py:406`):

**Pre-processing:** Strip markdown code fences (` ```json ... ``` `) and attempt to parse their contents first (iterating in reverse order, since the most recent code block is usually the most relevant).

**Strategy 1 — Standard key order:** Search for `{"tool": "name", "args": {...}` pattern using regex. Find the matching closing brace via bracket-counting (`find_matching_delimiter` from `utils.py`). Attempt strict `json.loads()`. Success condition: parsed object has both `"tool"` and `"args"` keys.

**Strategy 2 — Reversed key order:** Search for `{"args": {...}, "tool": "name"}` pattern (some LLMs output args before tool name). Same bracket-counting and validation.

**JSON repair** (`_repair_json`, agent.py:491): If strict parsing fails:
1. Remove trailing commas before closing braces (`,\s*} → }`)
2. Remove trailing commas before closing brackets (`,\s*] → ]`)
3. Escape raw newlines, tabs, and carriage returns inside JSON string values using a state machine (`_escape_raw_whitespace_in_strings`, agent.py:512) that tracks whether the current position is inside a double-quoted string, accounting for escape characters.

**Diagnostic feedback:** When all strategies fail, the parser records the parse error position, message, and a 80-character snippet around the error position. This is injected into the next HumanMessage so the LLM can self-correct: "Your last response had JSON errors: pos 342: Expecting ',' delimiter near ...\nraw newline here... Common causes: unescaped backslashes..."

**Write strategy hierarchy** embedded in the DeepSeek system prompt (agent.py:348-362):
1. **Preferred**: `run_command` with Python one-liner (`python3 -c "open('out.py','w').write('''...code...''')"`) — avoids ALL JSON escaping issues since the code is inside Python triple-quoted strings, not JSON strings.
2. **Alternative**: `write_lines` — each line is a separate array element, avoiding multi-line string escaping.
3. **Last resort**: `write_file` — only for short content (<10 lines), since the content field is a single JSON string where all special characters must be escaped.

This hierarchy is itself a workaround for a fundamental limitation: expecting an LLM to generate valid JSON containing arbitrary code (with its own quotes, backslashes, and newlines) is inherently error-prone. The `run_command` escape hatch pragmatically routes around the problem rather than trying to solve it through better parsing.

## Important Papers & References

- **DeepSeek-V3 Technical Report (DeepSeek-AI, 2024)** — Describes the architecture of DeepSeek-V3, the model behind the `deepseek-chat` API endpoint. Introduces Multi-Token Prediction (MTP) and auxiliary-loss-free load balancing. Relevant to understanding the JSON generation reliability and tool-calling capabilities of the DeepSeek backend.

- **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (DeepSeek-AI, 2025)** — Introduces the `deepseek-reasoner` model with chain-of-thought reasoning. While UMAF uses `deepseek-chat` by default, the architecture is compatible with the reasoner model for roles requiring deeper reasoning.

- **The Claude Model Family (Anthropic, 2024-2026)** — The Claude models powering the Claude CLI backend. Native tool use, extended thinking, and stream-json output format are all features of the Claude API/SDK that UMAF accesses through the subprocess interface.

- **AutoGPT: An Autonomous GPT-4 Experiment (Richards, 2023)** — The canonical autonomous agent loop (Plan → Act → Observe → Decide) that UMAF's `_run_deepseek()` implements. While AutoGPT popularized the pattern, UMAF adds production-grade features: circuit breakers, checkpoint persistence, multi-strategy JSON repair, and backend-aware tool translation.

- **ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., ICLR 2023)** — The reasoning+acting paradigm underlying UMAF's agent loop design. The interleaving of thought (AIMessage) and action (tool call) in `BaseAgent._run_deepseek()` directly implements the ReAct pattern.

- **Template Method Pattern (Gamma et al., "Design Patterns: Elements of Reusable Object-Oriented Software", 1994)** — The `AgentRole` abstract class with `execute()` as the template method and `build_task(backend)` / `tools_for_backend(backend)` as primitive operations is a direct application of this classic design pattern, adapted for backend-aware multi-agent systems.

- **Durable Execution for AI Agents (Diagrid, 2026)** — Industry analysis benchmarking checkpoint/resume capabilities across five major frameworks. Highlights the gap between "checkpointing" (saving state) and "durable execution" (guaranteeing completion). UMAF's `CheckpointManager` + pipeline-layer retry logic addresses the checkpointing half; the full durability guarantees (auto-resume, duplicate prevention) remain future work.

- **Agentic Design Patterns (Gullí, Springer 2025)** — Comprehensive catalog of 21 agentic design patterns including prompt chaining, tool use, multi-agent collaboration, and self-correction. UMAF implements several of these: prompt chaining (pipeline node sequencing), tool use (8 tools), multi-agent collaboration (all 7 pipelines), and self-correction (coder↔reviewer loops).

- **ADEMA: A Knowledge-State Orchestration Architecture (2026)** — Empirical evidence that removing checkpoint/resume from a multi-agent system produced the only invalid run in a 60-run mechanism matrix. Validates UMAF's design decision to make checkpoint-based recovery a first-class feature.

- **LLM Provider Abstraction (LiteObject, 2024)** — Open-source implementation of the Adapter + Factory pattern for swappable LLM providers. UMAF's `LLMProvider` ABC + `get_llm()` factory follows the same architectural pattern, adapted for the specific needs of multi-agent orchestration (working directory awareness, tool mapping, environment injection).

## Open Questions & Future Directions

1. **Anthropic API direct integration**: Currently, the Claude CLI backend requires the `claude` npm package to be installed and routes through a subprocess. A native Anthropic API integration (bypassing the CLI entirely, using the Anthropic Python SDK with native tool use) would eliminate subprocess overhead, enable parallel agent execution without OOM concerns (currently SkillPipeline limits claude_cli detectors to 1 parallel worker), and provide finer-grained cost tracking and streaming control. The `LLMProvider` ABC is already designed to accommodate this — adding an `AnthropicProvider` would require only implementing `invoke()` and `stream_invoke()`.

2. **Cross-backend load balancing and failover**: UMAF currently requires the user to choose a single backend per pipeline run (`--backend` flag). A more sophisticated approach would allow per-agent backend selection (matching the TopologyPipeline's design suggestion of cheaper models for simple tasks, more capable models for complex reasoning), automatic failover when one backend fails (DeepSeek API error → fall back to Claude CLI), or cost-aware routing (use DeepSeek until complexity threshold exceeded, then escalate to Claude).

3. **Streaming output to user during agent execution**: Neither backend currently provides real-time streaming to the end user. DeepSeek uses polling (invoke → check response → parse tool call), and Claude CLI processes stream-json events internally but only exposes final results. Implementing LangChain's `astream_events` or a callback-based streaming layer would improve UX for long-running research and code generation tasks.

4. **Token counting and cost estimation**: UMAF has no mechanism to track token consumption or estimate pipeline costs. With large research decompositions (8 workers × 600s each × multiple versions), costs can scale rapidly without the user's awareness. Adding token counting (via DeepSeek API's `usage` response field and Claude CLI's `result` event `total_cost_usd`) and budget-based early termination would be valuable for production use.

5. **Formal verification of checkpoint correctness**: The `CheckpointManager` currently serializes and deserializes messages without verifying semantic equivalence (does the deserialized checkpoint produce the same agent behavior?). For production systems, a formal verification mechanism — re-running a random sample of checkpoints and comparing outputs — would provide stronger durability guarantees.

6. **Multi-model orchestration within a single pipeline**: The `tools_config.json` format and `AgentRole` architecture already support per-role backend specification, but the pipeline runner applies a single backend globally. Enabling per-agent backend selection would allow, for example: DeepSeek for the research decomposer (simple classification), Claude CLI for research workers (deep multi-step research), DeepSeek for the reviewer (scoring with well-defined schema).

7. **Pre-fetch layer generalization**: The current pre-fetch mechanism is hardcoded for arxiv.org and a few other academic domains. A generalized pre-fetch registry (domain patterns + download strategies) would allow users to configure pre-fetching for any domain blocked by Claude Code's cc-switch, not just academic sites.

8. **Persistent memory across pipeline invocations**: `CheckpointManager` operates within a single pipeline run. The SelfEvolution pipeline reads agent logs to identify improvement patterns, but there is no persistent knowledge store that accumulates learnings across multiple runs. A vector database of past agent outputs and their review scores would enable the framework to avoid re-doing work it has already successfully completed.

9. **Reducing DeepSeek JSON parse failure rate**: Despite the sophisticated 4-strategy parser and 3-tier write strategy hierarchy, JSON parse failures remain a significant source of agent inefficiency. Exploring alternative approaches — constrained decoding (forcing the model to output valid JSON token-by-token), few-shot examples of successful tool calls in the system prompt, or migrating to DeepSeek's native function calling if and when it becomes available — could substantially improve DeepSeek backend reliability.

10. **Quantitative benchmarking of backend reliability**: While individual pipeline runs provide anecdotal evidence of backend performance, UMAF lacks a systematic benchmark comparing DeepSeek vs. Claude CLI across identical tasks. A benchmark suite with 50+ standardized tasks covering all 7 pipelines, measuring success rate, iteration count, time-to-completion, output quality (via reviewer scoring), and cost would provide the data needed to make informed backend selection decisions.

## Relevance to Main Topic

The LLM backend system is the execution substrate for all seven UMAF pipelines — every agent interaction, from code generation to research synthesis to skill detection, flows through `BaseAgent.run()` → `get_llm(backend).invoke()`. The dual-backend design is not merely an implementation detail but a fundamental architectural commitment that shapes every other layer of the framework:

- **Agent core**: The entire `BaseAgent` class (~830 lines) bifurcates into two execution paths (`_run_deepseek` and `_run_claude_cli`), each with its own prompt format, tool interface, error handling, and checkpointing strategy. The JSON tool-call parser, with its 4 strategies and state-machine-based repair, exists solely because DeepSeek lacks native tool calling — it represents a substantial engineering investment (approximately 200 lines of parsing code) to compensate for an API limitation.

- **Role layer**: Every `AgentRole.build_task(backend, **context)` method must generate two distinct prompt variants, representing a permanent maintenance cost. The backend-aware task generation documented in Section 3 above is a cross-cutting concern affecting all 32 roles. The `BaseDecomposerRole._backend_instructions()` template method exists specifically to inject backend-specific guidance into decomposition prompts.

- **Pipeline layer**: Pipeline parallelism limits (SkillPipeline caps claude_cli detectors at 1) exist because each Claude CLI subprocess is memory-heavy (~2-4GB per subprocess). The pre-fetch layer (`_prefetch_arxiv_sources`) is invoked by research pipelines before worker execution, adding framework-level complexity to work around a Claude Code limitation.

- **Configurability**: `tools_config.json` per-role tool assignments are backend-agnostic by design — the same tool list serves both backends, with `BaseAgent` handling the translation (Python tool names → Claude CLI native names, JSON tool-call instructions → native tool instructions). This design choice means tool configuration is a single-source-of-truth concern, but it also limits per-backend tool customization (e.g., giving Claude CLI workers `Bash` but not DeepSeek workers `run_command` for the same role).

- **Ecosystem positioning**: UMAF's dual-backend design aligns with a broader industry trend toward provider-agnostic agent frameworks (LightAgent, AG2, Agent SDK Core all support OpenAI + DeepSeek + Anthropic backends). However, UMAF is unique in supporting both an API-based and a CLI-subprocess-based backend with equivalent capabilities, making it a valuable reference architecture for frameworks that need to integrate with local AI tools (Claude Code, Codex CLI, Aider) alongside cloud APIs.

The backend system's design decisions cascade through the entire framework: the JSON repair system exists because DeepSeek can't natively call tools; the pre-fetch layer exists because Claude Code can't verify arxiv.org; the checkpoint granularity differs between backends because their invocation models are fundamentally different. Understanding these constraints and their solutions is essential for anyone extending UMAF, adding new pipelines, or porting its architecture to other LLM provider combinations.
