---
name: backend-architecture
description: "DeepSeek vs Claude CLI backends — tool name translation, env injection, subprocess architecture, and cwd sandboxing"
metadata: 
  node_type: memory
  type: project
  originSessionId: d4200744-181c-4ba7-9d5b-36d64631acd6
---

## Two Backends

### DeepSeek (`--backend deepseek`)

Default. Uses LangChain `ChatOpenAI`:
```python
ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.3,
    max_tokens=4096,
)
```
Agent loop: system prompt describes tools in JSON format → LLM outputs `{"tool": "...", "args": {...}}` → Python parses and executes → loop until TASK_COMPLETE. Requires `DEEPSEEK_API_KEY` in `.env`.

### Claude CLI (`--backend claude_cli`)

`ClaudeCLILLM` class shells out to the `claude` CLI binary:
```python
claude -p "<prompt>" --output-format text --allowedTools Read,Write,Bash,WebSearch
```

Environment injected from `claude_env_sample.json` via `merge_claude_env()`:
- `ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic` (routes through DeepSeek proxy)
- `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_MODEL=deepseek-v4-pro`, etc.
- The `claude -p` process is itself multi-turn with native tool access — the framework makes a single invocation and captures the final text output.

## Tool Name Translation

The framework uses Python tool names internally (`read_file`, `write_file`, `run_command`, `call_claude`, `web_search`). Claude CLI uses native names (`Read`, `Write`, `Bash`, `WebSearch`).

Translation happens in two places:
1. **System prompt** (`_build_claude_cli_prompt`): uses `_CLAUDE_NATIVE_TOOL_SPECS` to list tools by native name with correct parameter specs
2. **Task text** (`_translate_task_for_claude`): word-boundary regex replaces all occurrences of Python names with native names, e.g. `call_claude` → `Bash (run: claude -p "your prompt")`

The translation covers ALL phrasings (not just "Use X to"): `using`, `via`, `call X with`, etc. — via `\b` word-boundary regex.

## Critical: cwd Sandboxing

`ClaudeCLILLM.invoke()` passes `cwd=working_dir` to `subprocess.run()`. Without this, `claude -p` writes files to the current directory (project root) instead of the temp working directory. The `call_claude` tool in `tools.py` also sets `cwd=working_dir`.

## Permission Setup

Subprocess `claude -p` instances need permissions. `~/.claude/settings.json` must contain:
```json
{
  "permissions": {
    "WebSearch": "*",
    "Bash": "*",
    "Read": "*",
    "Write": "*",
    "Edit": "*"
  }
}
```

Without this, `claude -p` will prompt for permission and hang (timeout after 120s).

**Why:** Understanding the two backends is essential for configuration, debugging, and extending the framework.
**How to apply:** Set `CLAUDE_ENV_PATH` env var to use a custom config file. Related: [[project-overview]], [[circuit-breakers]].
