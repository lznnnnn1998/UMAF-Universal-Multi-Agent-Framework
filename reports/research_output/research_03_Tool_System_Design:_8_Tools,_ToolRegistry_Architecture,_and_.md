# Tool System Design: 8 Tools, ToolRegistry Architecture, and tools_config.json as Single Source of Truth

## Overview

UMAF's tool system is a three-layer architecture spanning 1,853 lines of Python across five files, providing autonomous LLM agents with eight sandboxed capabilities through a centralized registry and an externalized JSON configuration. The design follows three architectural principles: **separation of specification from implementation** (ToolSpec dataclasses describe what a tool does; TOOL_MAP callables implement how), **centralized role-based access control** (28 classmethods in ToolRegistry return per-role tool lists, driven entirely by a single JSON file), and **backend-transparent tool translation** (the same `tools_for_backend()` call produces DeepSeek JSON schemas or Claude CLI native tool names through a regex-based translation layer).

The motivation for this architecture arises from a fundamental tension in multi-agent systems: agents need powerful tools to accomplish tasks, but unrestricted tool access creates safety risks and degrades LLM reasoning accuracy (the "tool overload" problem, where too many available tools causes the model to select the wrong one or hallucinate non-existent capabilities). UMAF resolves this tension by externalizing tool assignments into `tools_config.json`, a declarative file that maps each of the 32 AgentRoles across 7 pipelines to a curated subset of the 8 available tools. This design enables three critical workflows: (1) operators can audit agent capabilities without reading Python code; (2) tools can be reconfigured at runtime via `--tools-config` without restarting or modifying any agent logic; and (3) role-based restrictions are enforced at the registry level — a role whose config entry is empty receives zero tools (the default for all 28 methods is `[]`), ensuring no accidental tool leakage.

The tool system is the execution substrate of the broader UMAF architecture described in the Architecture Overview (Section 5, "Tool Assignment Architecture"). Every `AgentRole.execute()` call in every pipeline node flows through this system: the agent role queries `ToolRegistry.*_tools()` → `_apply_override()` checks `tools_config.json` for pipeline-specific and `__global__` overrides → the resulting `ToolSpec` list is converted to backend-specific dicts → `BaseAgent.run()` injects them into the LLM prompt. This tight integration means the tool system is not an optional plugin but a load-bearing architectural component — changing a tool assignment in `tools_config.json` immediately alters agent behavior across all pipeline executions.

## Key Methods & Approaches

### 1. The 8 Tool Implementations: Specifications, Timeouts, and Parameters

Each tool is defined in two places: a `ToolSpec` dataclass in `tools/registry.py` (lines 79-118) providing the canonical metadata, and an implementation function in `tools/functions.py` (lines 26-259) providing the executable logic. The following table provides the complete specification for all 8 tools:

| # | Tool Name | Parameters | Default Timeout | Implementation | Description |
|---|-----------|-----------|-----------------|---------------|-------------|
| 1 | `read_file` | `path: str (required)` | N/A | `tools/functions.py:26-34` | Resolves path relative to `working_dir` via `_resolve_path()`. Returns file contents as UTF-8 text. Handles `FileNotFoundError` and generic exceptions with user-facing error strings. |
| 2 | `write_file` | `path: str (required)`, `content: str (required)` | N/A | `tools/functions.py:37-45` | Resolves path, creates parent directories automatically (`mkdir(parents=True, exist_ok=True)`). Writes single-string content. Returns success/error message. Used by 5 of 7 pipeline coder roles. |
| 3 | `write_lines` | `path: str (required)`, `lines: list[str] (required)` | N/A | `tools/functions.py:48-60` | Joins lines array with newlines (`"\n".join(lines)`). Exists because JSON arrays of strings are easier for DeepSeek LLMs to emit correctly than multi-line escaped strings — the primary workaround for JSON tool-call escaping issues in the DeepSeek backend. |
| 4 | `run_command` | `command: str (required)`, `description: str (optional)` | 30s (configurable) | `tools/functions.py:63-82` | Executes via `subprocess.run(shell=True, capture_output=True, text=True)`. Configurable timeout via `_tool_timeouts` dict. Merges stdout + stderr. Catches `TimeoutExpired` with descriptive error. The most widely-assigned tool — appears in 6 of 7 pipeline configurations. |
| 5 | `call_claude` | `prompt: str (required)` | 120s (configurable to 600s) | `tools/functions.py:85-111` | Spawns `claude -p` subprocess with `--output-format text` and `--permission-mode bypassPermissions`. Injects environment from `claude_env_sample.json` via `merge_claude_env()`. Catches `TimeoutExpired` and `FileNotFoundError` (CLI not installed). |
| 6 | `web_search` | `query: str (required)`, `max_results: int (optional, default 10, max 20)` | 15s (configurable) | `tools/functions.py:114-179` | Uses DuckDuckGo Lite (no API key required). Two-tier regex parsing: primary parser finds `class="result-link"` anchors and `class="result-snippet"` table cells; fallback parser uses simpler anchor regex when primary fails. Returns formatted markdown with titles, URLs, and snippets. |
| 7 | `web_fetch` | `url: str (required)`, `max_chars: int (optional, default 12000, max 20000)` | 20s (configurable to 30s) | `tools/functions.py:182-218` | Fetches via `urllib.request.urlopen()`. Strips `<script>`, `<style>`, `<nav>`, `<footer>`, `<header>` tags for HTML content; removes all remaining HTML tags; collapses whitespace. Returns plain text truncated to `max_chars`. Bypasses Claude Code's network sandbox entirely. |
| 8 | `download_file` | `url: str (required)`, `output_path: str (required)` | 30s (configurable) | `tools/functions.py:227-258` | Downloads via `urllib.request.urlopen()` and writes bytes to local file. Auto-creates parent directories. Returns human-readable string with file size in KB. Primary mechanism for navigating Claude Code's cc-switch domain verification block on arxiv.org — content is pre-fetched at the framework level before agents run. |

**The `_resolve_path()` helper** (`tools/functions.py:17-23`): A critical shared utility that normalizes paths relative to `working_dir`. It detects when the agent has accidentally prepended the working directory name to the path (a common LLM error) and strips it before resolution. This single function prevents the most frequent file-not-found error in agent tool calls.

**Tool timeout configuration**: Default timeouts are hardcoded in each function but overrideable via `ToolRegistry._tool_timeouts`. The `tools_config.json` `__timeouts__` section sets these at load time through `set_tool_config()`. For example, `call_claude` is overridden from its default 120s to 600s to accommodate long-running reasoning subtasks in the Research pipeline. The `run_command` function reads `ToolRegistry._tool_timeouts.get("run_command", 30)` on each invocation, making timeout changes effective immediately without function redefinition.

### 2. ToolRegistry Architecture: 28 Role-Specific Classmethods

The `ToolRegistry` class (`tools/registry.py:17-218`) is the central hub of the tool system, providing 28 classmethods that return per-role tool lists. Its design follows the **Registry Pattern** from enterprise software architecture, adapted for LLM agent frameworks.

**Classmethod taxonomy by pipeline:**

*Coder Pipeline (2 methods)*:
- `coder_tools()` → delegates to `_apply_override("coder", "coder", [])` — Coder gets 6 tools (read_file, write_file, run_command, call_claude, web_search, web_fetch)
- `reviewer_tools()` → delegates to `_apply_override("coder", "reviewer", [])` — Reviewer gets 5 tools (no write_file)

*Research Pipeline (4 methods)*:
- `research_decomposer_tools(backend="deepseek")` → delegates to `_apply_override("research", "decomposer", [])` — Head gets 4 tools (read_file, write_file, run_command, web_search)
- `research_worker_tools()` → delegates to `_apply_override("research", "worker", [])` — Workers get all 7 tools (most permissive role)
- `research_reviewer_tools()` → delegates to `_apply_override("research", "reviewer", [])` — Reviewer gets 4 tools (read_file, write_file, web_search, web_fetch)
- `writer_tools()` → delegates to `_apply_override("research", "writer", [])` — Writer gets 3 tools (read_file, write_file, write_lines)

*CoderPP Pipeline (4 methods)*:
- `coderpp_decomposer_tools(backend="deepseek")` → Head gets 3 tools (read_file, write_file, run_command)
- `coderpp_worker_tools()` → Workers get 4 tools (read_file, write_file, write_lines, run_command)
- `coderpp_reviewer_tools()` → Reviewer gets 4 tools (read_file, write_file, write_lines, run_command)
- `organizer_tools()` → Organizer gets 4 tools (read_file, write_file, write_lines, run_command)

*Topology Pipeline (4 methods)*:
- `topology_analyzer_tools()` → 2 tools (read_file, write_file)
- `topology_designer_tools()` → 2 tools (read_file, write_file)
- `topology_evaluator_tools()` → 2 tools (read_file, write_file)
- `topology_writer_tools()` → 1 tool (write_file only — the most restricted role)

*Skill Pipeline (4 methods)*:
- `skill_scanner_tools()` → 3 tools (read_file, write_file, run_command)
- `skill_detector_tools()` → 3 tools (read_file, write_file, run_command)
- `skill_aggregator_tools()` → 2 tools (read_file, write_file)
- `skill_writer_tools()` → 2 tools (read_file, write_file)

*Feature Pipeline (5 methods, defined in `tools/feature_tools.py` and patched at import)*:
- `feature_scanner_tools()` → 3 tools (read_file, write_file, run_command)
- `feature_planner_tools()` → 2 tools (read_file, write_file)
- `feature_coder_tools()` → 4 tools (read_file, write_file, write_lines, run_command)
- `feature_reviewer_tools()` → 2 tools (read_file, run_command)
- `feature_writer_tools()` → 1 tool (write_file only)

*SelfEvolution Pipeline (5 methods)*:
- `self_evolution_analyzer_tools()` → 3 tools (read_file, write_file, run_command)
- `self_evolution_planner_tools()` → 2 tools (read_file, write_file)
- `self_evolution_coder_tools()` → 4 tools (read_file, write_file, write_lines, run_command)
- `self_evolution_reviewer_tools()` → 3 tools (read_file, write_file, run_command)
- `self_evolution_writer_tools()` → 1 tool (write_file only)

**Utility classmethods** (`registry.py:42-218`):
- `set_tool_config(config)` — Class-level override injection (detailed in Section 3)
- `_apply_override(pipeline, role, defaults)` — Case-insensitive role matching against config
- `to_dicts(specs)` — Converts `list[ToolSpec]` to `list[dict]` for prompt injection
- `get_map()` — Returns a copy of `TOOL_MAP` for `BaseAgent` initialization

**The zero-default guarantee**: Every classmethod passes `[]` as the `defaults` parameter to `_apply_override()`. This means that if `tools_config.json` is not loaded (or a role is undefined in config), the agent receives zero tools. This is a safety-by-design decision: accidental tool leakage is impossible because there is no codepath that provides tools without explicit configuration. Compare this to alternative designs where tools are hardcoded as classmethod defaults (the pre-v1.7 approach), which could allow an agent to access dangerous tools simply because a developer forgot to update a method.

### 3. tools_config.json: Single Source of Truth Architecture

The `tools_config.json` file (`tools_config.json`, 236 lines) serves as the declarative, auditable, and externally modifiable definition of which tools each agent role can access. Its design follows the **Configuration-as-Code** pattern, where operational parameters are externalized from source code into a version-controlled, human-readable format.

#### 3.1 File Structure

The file is a JSON object with top-level keys representing pipeline names and two special metadata keys:

```json
{
  "__about__": "human-readable description",
  "__format__": "documentation of the structure",
  "__tools_reference__": { /* complete tool catalog with parameter specs */ },
  "__timeouts__": { /* per-tool timeout overrides */ },
  "__global__": { /* fallback for all pipelines */ },
  "research": { /* pipeline → role → [tool_names] */ },
  "coder": { /* ... */ },
  "coderpp": { /* ... */ },
  "topology": { /* ... */ },
  "skill": { /* ... */ },
  "feature": { /* ... */ },
  "self_evolution": { /* ... */ }
}
```

Each pipeline section maps role names to arrays of tool name strings. Role names use **case-insensitive substring matching** against `ToolRegistry` method suffixes. For example, the key `"worker"` in the `"research"` section matches both `research_worker_tools()` and — through the `"coderpp"` section with its own `"worker"` key — `coderpp_worker_tools()`. This design avoids duplicating the full method name while preserving pipeline-scoped isolation.

#### 3.2 The `__global__` Fallback Mechanism

The `__global__` key (`tools_config.json:70-72`) provides a universal fallback: if a role is not defined in its pipeline-specific section, `ToolRegistry._apply_override()` checks `__global__` before returning the empty default. The lookup order in `_apply_override()` (`registry.py:64-76`) is:

1. Check the specific pipeline section (e.g., `"coder"`) for a matching role key
2. Check `"__global__"` for a matching role key
3. Return the defaults (always `[]`)

This enables operators to define broad policies (e.g., "all writer roles get only `write_file`") in `__global__` and override them per-pipeline as needed. The lookups iterate `for key in (pipeline, "__global__")`, making the pipeline section strictly higher priority than the global fallback.

#### 3.3 Metadata Key Stripping on Load

`ToolRegistry.set_tool_config()` (`registry.py:43-56`) strips metadata keys from the loaded config using a predicate function:

```python
def _is_meta(k: str) -> bool:
    return k.startswith("_") and k != "__global__"
```

This is applied to both top-level keys (pipeline names) and nested role-level keys within each pipeline section. The design convention is that any key prefixed with `_` (except `__global__`) is documentation-only metadata. This allows the config file to include inline documentation (`_description`, `__about__`, `__format__`, `__tools_reference__`) without those keys polluting the runtime data structures used for tool lookup. The `__timeouts__` key is also popped and processed separately before stripping.

#### 3.4 Per-Tool Timeout Overrides

The `__timeouts__` section (`tools_config.json:65-69`) maps tool names to timeout values in seconds:

```json
"__timeouts__": {
    "_description": "Override per-tool timeouts...",
    "call_claude": 600,
    "web_fetch": 30
}
```

The processing in `set_tool_config()` (`registry.py:44-48`) validates that each value is a positive integer and stores it in `cls._tool_timeouts`. These overrides supersede the hardcoded defaults in `tools/functions.py` (e.g., `run_command` defaults to 30s, `web_search` to 15s). Importantly, the `run_command` and `call_claude` functions read `ToolRegistry._tool_timeouts.get("<name>", <hardcoded_default>)` at invocation time, meaning timeout changes take effect immediately without requiring function redefinition or agent re-initialization.

#### 3.5 The `--tools-config` CLI Flag

In `main.py:73-77`, the `--tools-config` argument accepts an alternative JSON file path, defaulting to `<repo_root>/tools_config.json`. This enables:
- **Per-deployment customization**: Production deployments can use a restrictive config while development uses a permissive one
- **A/B testing**: Different tool configurations can be compared for task completion rates without code changes
- **Security auditing**: Operators can point `--tools-config` at a read-only file in a protected directory, preventing modification by the agent itself

The loading flow (`main.py:102-105`) is: `_load_tools_config(args.tools_config)` → validates JSON and type → `ToolRegistry.set_tool_config(config)` → prints config path for audit trail. The config is loaded before any pipeline is instantiated, ensuring consistent tool assignments across the entire run.

### 4. The Tool Name Translation Layer for Claude CLI

When the backend is `claude_cli`, the framework must translate its internal Python tool names into the names that Claude Code's native tool-calling interface expects. This translation happens at two levels:

#### 4.1 Tool Spec Translation (Prompt Building)

`BaseAgent._CLAUDE_NATIVE_TOOL_SPECS` (`agent.py:255-264`) is a dict mapping each of the 8 Python tool names to a tuple of `(native_name, native_description)`:

| Python Name | Claude CLI Native Name | Translation Strategy |
|-------------|----------------------|---------------------|
| `read_file` | `Read` | Direct 1:1 mapping — `Read` is a standard Claude Code tool |
| `write_file` | `Write` | Direct 1:1 mapping — `Write` is a standard Claude Code tool |
| `write_lines` | `Write` | N:1 mapping — both `write_file` and `write_lines` map to `Write`, with `write_lines` providing additional instructions to "Join lines with newlines when writing" |
| `run_command` | `Bash` | Direct 1:1 mapping — standard shell execution |
| `web_search` | `WebSearch` | Direct 1:1 mapping — Claude Code native web search |
| `call_claude` | `Bash` | Represents nested `claude -p` as a Bash command with explicit instructions: "For deep reasoning or analysis, think through the problem yourself — you ARE the reasoning engine. Do NOT spawn nested claude -p calls." |
| `web_fetch` | `Bash` | Inlines a complete Python one-liner for urllib fetch as the Bash tool description: `python3 -c "import urllib.request; req=urllib.request.Request(URL, headers={...}); print(urllib.request.urlopen(req, timeout=20).read().decode(...)[:8000])"` |
| `download_file` | `Bash` | Inlines a Python one-liner for urllib download + save as the Bash tool description, with the instruction to "replace URL and OUTPUT_PATH with actual values" |

The `_build_claude_cli_prompt()` method (`agent.py:365-392`) iterates through the agent's assigned tools, looks up each in `_CLAUDE_NATIVE_TOOL_SPECS`, and builds a markdown-formatted tool list for the Claude CLI system prompt. Importantly, it de-duplicates the tool list using the `allowed` set (line 842-843), ensuring that tools mapping to the same Claude CLI native name (e.g., `write_file` and `write_lines` both → `Write`) appear only once in `--allowedTools`.

#### 4.2 Task Text Translation (Regex Replacement)

`BaseAgent._translate_task_for_claude()` (`agent.py:394-402`) is a static method that applies regex-based text replacement to the task prompt before it's sent to Claude CLI. It uses `_TOOL_NAMES_TO_TRANSLATE` (`agent.py:266-275`), a list of `(python_name, claude_name)` tuples:

```python
_TOOL_NAMES_TO_TRANSLATE = [
    ("call_claude", 'Bash (run: claude -p "your prompt")'),
    ("run_command", "Bash"),
    ("read_file", "Read"),
    ("write_file", "Write"),
    ("write_lines", "Write"),
    ("web_search", "WebSearch"),
    ("web_fetch", "Bash (python3 urllib fetch)"),
    ("download_file", "Bash (python3 urllib download + save to file)"),
]
```

For each pair, the method runs:
```python
re.sub(rf'`{re.escape(py_name)}`|\b{re.escape(py_name)}\b', cli_name, task)
```

This handles two patterns: backtick-enclosed tool names (`` `read_file` `` → `Read`) and bare word-boundary occurrences (`read_file` → `Read`). The regex escaping via `re.escape()` prevents name collisions (e.g., partial matches of shorter names). The translation is applied before any agent runs, ensuring that Research pipeline task prompts like "Use `read_file` to inspect..." are automatically converted to "Use Read to inspect..." for Claude CLI comprehension.

#### 4.3 Allowed Tools Injection

In `_run_claude_cli()` (`agent.py:842-844`), the translated tool names are deduplicated and passed as `--allowedTools` to the `claude -p` subprocess:

```python
allowed = list(dict.fromkeys(
    spec[0] for t in self.tools if (spec := self._CLAUDE_NATIVE_TOOL_SPECS.get(t["name"]))
))
cmd.extend(["--allowedTools", ",".join(allowed)])
```

The `dict.fromkeys()` pattern preserves insertion order while removing duplicates — critical for cases like `write_file` and `write_lines` both mapping to `Write`.

### 5. The TOOL_MAP Separation Pattern

The `TOOL_MAP` (`tools/functions.py:261-273`) is a flat dictionary mapping tool name strings to their implementation callables:

```python
TOOL_MAP = {
    "read_file": read_file,
    "write_file": write_file,
    "write_lines": write_lines,
    "run_command": run_command,
    "call_claude": call_claude,
    "web_search": web_search,
    "web_fetch": web_fetch,
    "download_file": download_file,
}
```

This dictionary is simultaneously:
1. Stored as a module-level variable in `tools/functions.py`
2. Assigned to `ToolRegistry.TOOL_MAP` at import time (line 273)
3. Retrieved by `ToolRegistry.get_map()` (returns a copy for encapsulation)
4. Passed to `BaseAgent.__init__()` as `tool_map` in `AgentRole.execute()` (`agent.py:1132`)

The separation pattern provides three benefits:
- **Spec/impl decoupling**: `ToolSpec` objects in `registry.py` define the interface; `TOOL_MAP` in `functions.py` provides the implementation. The registry has no import dependency on the functions module (it imports only `typing`), while the functions module imports from the registry to populate `TOOL_MAP`.
- **Testability**: Tests can replace `tool_map` with mock implementations without touching ToolRegistry. The `BaseAgent.__init__()` accepts an explicit `tool_map` parameter.
- **Lazy binding**: The `TOOL_MAP` assignment at import time (`ToolRegistry.TOOL_MAP = TOOL_MAP`) happens after all functions are defined, preventing circular import issues.

### 6. Modular Package Structure

The `tools/` package comprises four files in a clean layered architecture:

```
tools/
├── __init__.py        (18 lines) — Package exports and auto-patching
├── registry.py        (218 lines) — ToolSpec + ToolRegistry + 23 classmethods
├── functions.py       (273 lines) — 8 implementations + TOOL_MAP + _resolve_path
└── feature_tools.py   (70 lines) — 5 feature_*_tools() methods, patched at import
```

**`__init__.py`** (`tools/__init__.py:1-18`): Re-exports `ToolSpec`, `ToolRegistry`, `TOOL_MAP`, and all 8 implementation functions. Auto-applies feature tool methods to `ToolRegistry` via `apply_feature_tools(ToolRegistry)`. This is the single import point — consumers do `from tools import ToolRegistry` rather than importing individual submodules.

**`registry.py`**: Contains the `ToolSpec` dataclass (frozen, 3 fields: `name`, `description`, `parameters`), the `ToolRegistry` class with `_TOOL_NAME_MAP` (Python-to-constant mapping for config resolution), 8 `ToolSpec` class attributes, 23 role-specific classmethods, `set_tool_config()`, `_apply_override()`, `to_dicts()`, and `get_map()`. The class has no constructor — all state is class-level, making it a singleton-like registry pattern.

**`functions.py`**: Contains `_resolve_path()` (shared path normalizer), 8 tool implementation functions (each accepting `**kwargs` with at minimum `working_dir`), and the `TOOL_MAP` dictionary. Imports `merge_claude_env` from `claude_config.py` for `call_claude()` environment injection. Imports `ToolRegistry` for timeout lookups at invocation time.

**`feature_tools.py`**: Uses a **monkey-patching pattern**: the `apply_to_tool_registry()` function (lines 25-70) dynamically creates 5 `@classmethod` functions (`feature_scanner_tools` through `feature_writer_tools`) and attaches them to the `ToolRegistry` class. This is called once at first import (`tools/__init__.py:18`). The pattern avoids modifying `registry.py` when adding new pipeline tool methods — new pipelines can add their own `_tools.py` file with a similar `apply_to_tool_registry()` function.

### 7. Comparison with Related Work

UMAF's tool system design can be situated within the broader landscape of LLM agent tool architectures:

**vs. MCP (Model Context Protocol)**: Anthropic's MCP standard (late 2024) defines a JSON-RPC protocol for tool discovery and invocation between AI applications and tool servers. UMAF predates MCP and takes a simpler approach: instead of a network protocol with server-side tool registration, UMAF uses a single local JSON file as the tool registry. The trade-off is clear — MCP enables dynamic, cross-process tool discovery at the cost of protocol overhead and infrastructure complexity; UMAF's `tools_config.json` is trivially auditable and deployable but requires all tools to be Python functions within the same process.

**vs. Semantic/Tool Retrieval (RAG-based)**: Several frameworks (ABI Swarm, Z-Space) use vector embeddings to retrieve relevant tools at runtime based on semantic similarity to the user query. UMAF takes the opposite approach: role-based static assignment, where the pipeline designer decides which tools each role gets. This sacrifices flexibility (a role cannot dynamically discover a new tool mid-task) for predictability (operators know exactly what each agent can do, essential for safety-critical deployments).

**vs. LangChain Tool Integration**: LangChain's `@tool` decorator and `StructuredTool` class auto-generate JSON schemas from function signatures. UMAF's `ToolSpec` dataclass is more manual but also more explicit — parameters are documented as human-readable strings rather than auto-generated from type hints, which is beneficial when the LLM's understanding of a parameter differs from its Python type (e.g., `"path": "str"` is clearer to an LLM than `path: str` which could be misinterpreted as any string).

**vs. Dynamic Tool Generation (Apollo, AGENTORCHESTRA)**: Some systems allow agents to generate and deploy their own tools at runtime. UMAF's tool system is deliberately static — the 8 tools represent a fixed capability set that covers file I/O, shell execution, web access, and nested LLM calls. The design philosophy is that a small, well-understood tool set with explicit role-based restrictions is safer and more reliable than an unbounded tool generation capability.

### 8. Design Patterns and Architectural Principles

**Safety-by-default**: All 28 role methods default to `[]`. No tool is available unless explicitly configured.

**Single source of truth**: `tools_config.json` is the only place where role→tool mappings are defined. The Python methods are pure pass-throughs.

**Separation of concerns**: ToolSpec (what) ≠ implementation (how) ≠ assignment (who) ≠ translation (how presented).

**Progressive disclosure**: The `__global__` key provides broad defaults; pipeline-specific keys override; individual roles within pipelines provide the finest granularity.

**Graceful degradation**: If config is missing or a role is undefined, the agent runs with zero tools but the pipeline doesn't crash — the agent simply uses internal reasoning.

**Import-time composability**: The feature_tools patching pattern allows new pipelines to extend ToolRegistry without modifying its source file, analogous to the Open/Closed Principle.

## Important Papers & References

- **Anthropic. "Model Context Protocol Specification" (2024-2025)** — The MCP standard provides a JSON-RPC 2.0 protocol for AI-tool integration with tool discovery, streaming responses, and formal OAuth 2.1 authorization. UMAF's `tools_config.json` is an alternative, simpler approach to the same problem: a static, file-based tool registry vs. MCP's dynamic, protocol-based one. URL: https://modelcontextprotocol.io

- **Gamma, E., Helm, R., Johnson, R., and Vlissides, J. "Design Patterns: Elements of Reusable Object-Oriented Software" (Addison-Wesley, 1994)** — UMAF's ToolRegistry implements the Registry pattern (a variant of the Singleton pattern), while the feature_tools.py patching follows the Visitor pattern (adding operations to a class without modifying it). The ToolSpec/TOOL_MAP separation implements the Bridge pattern (decoupling abstraction from implementation).

- **Schmid, P., et al. "Survey of LLM Agent Communication with MCP: A Software Design Pattern Centric Review" (arXiv:2506.05364, May 2025)** — Catalogues design patterns for agent-tool communication, including Mediator, Observer, and Broker patterns. UMAF's ToolRegistry acts as a Mediator between pipeline roles and tool implementations, routing tool requests through a centralized registry rather than direct agent-to-tool coupling.

- **Panchal, D. "Simpliflow: A Lightweight Open-Source Python Library for Generative AI Agent Workflows Defined in a Single JSON Configuration File" (arXiv:2510.10675, 2024)** — Demonstrates the "configuration-over-code" philosophy where entire agent workflows are defined in JSON. UMAF's `tools_config.json` follows this same pattern, treating the JSON file as the single source of truth for tool assignments. URL: https://export.arxiv.org/pdf/2510.10675

- **Yao, S., et al. "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)** — The foundational paper establishing the Thought→Action→Observation loop. UMAF's `BaseAgent._run_deepseek()` implements this loop: the LLM outputs a JSON tool call (Action), `_execute_tool()` runs it (Observation), and the result is fed back for the next reasoning step.

- **Wang, L., et al. "A Survey on Large Language Model based Autonomous Agents" (Frontiers of Computer Science, 2024)** — Comprehensive survey covering tool-augmented LLMs. Identifies the "tool overload" problem: as the number of available tools grows, LLM accuracy in tool selection degrades. UMAF's per-role tool subsetting directly addresses this by limiting each agent to 1-7 relevant tools rather than exposing all 8.

- **AWS Prescriptive Guidance. "Agentic AI Patterns" (July 2025)** — Defines a taxonomy of agent patterns including Tool-based Agents (Function Calling and Server-based). UMAF's architecture combines both: the DeepSeek backend uses Function Calling (JSON tool-call parsing), while the Claude CLI backend uses Server-based tool delegation (subprocess with `--allowedTools`). URL: https://docs.aws.amazon.com/pdfs/prescriptive-guidance/latest/agentic-ai-patterns/agentic-ai-patterns.pdf

- **Zheng, Y., et al. "CaveAgent: Stateful and Object-Oriented LLM Agent Runtime" (arXiv, Jan 2025)** — Challenges the stateless JSON tool-call paradigm by introducing persistent variable spaces. UMAF takes a middle ground: the DeepSeek backend is stateless JSON, but the Claude CLI backend maintains state through incremental checkpointing of the streaming subprocess. URL: https://arxiv.org/pdf/2601.01569.pdf

- **TEA Protocol / AGENTORCHESTRA. "A Unified Tool-Environment-Agent Protocol for Multi-Agent Systems" (OpenReview, 2025)** — Proposes tool context protocol with embedding-based retrieval and agent-as-tool registration. Achieved 83.39% on GAIA benchmark. Relevant to UMAF's future direction: could `call_claude` be extended to register agents as tools? URL: https://openreview.net/pdf?id=YcnKdeI9pp

## Open Questions & Future Directions

1. **Dynamic tool discovery**: UMAF's tool set is statically defined and fixed at startup. As the MCP ecosystem matures, could UMAF adopt MCP for dynamic tool server discovery (e.g., connecting to external MCP servers for database queries, API calls, or specialized computation) while retaining `tools_config.json` for role-based access control?

2. **Tool combination safety**: The current system controls which tools each role can access but not which tool *sequences* are allowed. A malicious or confused agent could chain safe tools into dangerous patterns (e.g., `web_fetch` → `write_file` → `run_command` to download and execute remote code). A future "tool sequence policy" layer could define allowed and forbidden tool chains.

3. **Per-invocation tool budgeting**: `tools_config.json` controls which tools are available but not how many times they can be invoked. A Research worker could theoretically call `web_search` 50 times in a single task. Adding per-tool invocation limits or cost budgets would provide finer-grained control.

4. **Tool observability metrics**: Currently, tool execution results are logged but not aggregated into metrics. Adding per-tool latency histograms, error rate tracking, and invocation counts to the agent logs would enable operators to identify which tools are most frequently used, slowest, or most error-prone.

5. **Tool versioning**: If a tool implementation changes (e.g., `web_search` switches from DuckDuckGo Lite to Brave Search API), there is no mechanism to version the tool or record which version was used in past pipeline runs. Schema versioning (as practiced in MCP servers) could be adopted.

6. **Tool result caching**: Repeated tool calls with identical parameters (e.g., the same `web_search` query across multiple workers) waste API calls and introduce inconsistency. A result cache with TTL-based invalidation could improve efficiency for read-only tools.

7. **Cross-backend tool parity**: The tool name translation layer (`_CLAUDE_NATIVE_TOOL_SPECS`) is manually maintained. As Claude Code's native tool set evolves, this mapping could drift out of sync. Auto-generating the translation layer from Claude Code's tool manifest would improve maintainability.

8. **Tool sandboxing**: `run_command` executes arbitrary shell commands with the agent process's permissions. Docker-based or Firejail-based sandboxing per tool invocation would be a natural security hardening for production deployments, especially given that `run_command` is the most widely assigned tool (used by 6 of 7 pipelines).

9. **Agent-as-tool integration**: `call_claude` spawns a fresh subprocess for each invocation. The TEA Protocol's concept of "agent as tool" (registering an agent as a callable capability) could be implemented by extending `ToolSpec` with an `agent_role` field, allowing pipeline roles to invoke other specialized agents as tools rather than only spawning generic Claude instances.

10. **Config validation and schema enforcement**: `tools_config.json` has no formal JSON Schema. Adding one (with tool name enumeration, timeout type checking, and role name validation) would catch misconfigurations at load time rather than at runtime when an agent fails due to missing tools.

## Relevance to Main Topic

The tool system is the execution fabric connecting all seven UMAF pipelines to the external world. Every agent in every pipeline — all 32 AgentRoles — acquires its capabilities through the three-layer architecture described here: `tools_config.json` → `ToolRegistry` classmethods → `BaseAgent` tool dispatch. Understanding the tool system is essential for:

- **Operators** who need to audit and restrict agent capabilities per deployment environment
- **Pipeline designers** who must assign appropriate tool subsets to each role based on the principle of least privilege
- **Security reviewers** evaluating the attack surface of autonomous LLM agents
- **Framework extenders** adding new pipelines (the feature_tools.py patching pattern is the template) or new tools (requiring additions to ToolSpec, TOOL_MAP, `_CLAUDE_NATIVE_TOOL_SPECS`, and `_TOOL_NAMES_TO_TRANSLATE`)

The architecture demonstrates that a well-designed tool system for multi-agent frameworks should externalize configuration (JSON), centralize registration (ToolRegistry), separate specification from implementation (ToolSpec vs. TOOL_MAP), and provide backend-transparent translation (regex-based name mapping). These principles are generalizable beyond UMAF to any LLM agent framework that faces the tension between agent capability and operator control.
