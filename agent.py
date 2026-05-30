import json
import os
import re
import time
from typing import Any, Callable, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from llm import get_llm

# Map our framework tool names to Claude CLI's native tool specs
# Each entry: (native_name, description_with_correct_params)
_CLAUDE_NATIVE_TOOL_SPECS: dict[str, tuple[str, str]] = {
    "read_file": ("Read", "Read a file. Parameters: file_path (str, required)"),
    "write_file": ("Write", "Write content to a file. Parameters: file_path (str), content (str)"),
    "run_command": ("Bash", "Run a shell command. Parameters: command (str, required), description (str, optional)"),
    "web_search": ("WebSearch", "Search the web. Parameters: query (str, required)"),
    "call_claude": ("Bash", "Use Bash for running commands and scripts. For deep reasoning or analysis, think through the problem yourself — you ARE the reasoning engine. Do NOT spawn nested claude -p calls."),
}

# Python tool name → Claude CLI native tool name (for translating task text)
_TOOL_NAME_TRANSLATION = {
    "call_claude": "Bash (run: claude -p \"your prompt\")",
    "read_file": "Read",
    "write_file": "Write",
    "run_command": "Bash",
    "web_search": "WebSearch",
    "web_fetch": "WebFetch",
}

# Tool names that appear in task descriptions and need translation
# Regex word-boundary replacement covers all phrasings: "use X to", "using X", "via X", etc.
_TOOL_NAMES_TO_TRANSLATE = [
    ("call_claude", 'Bash (run: claude -p "your prompt")'),
    ("run_command", "Bash"),
    ("read_file", "Read"),
    ("write_file", "Write"),
    ("web_search", "WebSearch"),
    ("web_fetch", "WebFetch"),
]

# Tool error patterns that indicate a persistent failure (don't keep retrying)
_PERSISTENT_ERRORS = ("timed out", "not found", "no such file", "permission denied")
_FORCE_WRAPUP_THRESHOLD = 3   # steps remaining before forcing wrap-up
_MAX_CONSECUTIVE_ERRORS = 3   # consecutive tool errors before stopping retries


def _build_system_prompt(tools: list[dict[str, Any]], backend: str = "deepseek") -> str:
    if backend == "claude_cli":
        return _build_claude_cli_prompt(tools)
    return _build_deepseek_prompt(tools)


def _build_deepseek_prompt(tools: list[dict[str, Any]]) -> str:
    tool_descriptions = []
    for t in tools:
        tool_descriptions.append(
            f"  - {t['name']}: {t['description']}\n"
            f"    Parameters: {json.dumps(t['parameters'])}"
        )
    tools_text = "\n".join(tool_descriptions)

    return f"""You are an autonomous agent that completes tasks by using tools.

## Available Tools
{tools_text}

## Output Format
To call a tool, output a JSON block exactly like this:
```json
{{"tool": "tool_name", "args": {{"param1": "value1", "param2": "value2"}}}}
```

You may output reasoning or explanations before or after the JSON block. The tool result will be shown to you.

When the task is complete, output: TASK_COMPLETE

## Working Directory
All file paths are relative to the provided working directory."""


def _translate_task_for_claude(task: str) -> str:
    """Replace Python tool names in task text with Claude CLI native tool names.

    Uses word-boundary regex to catch all phrasings: "use X to", "using X",
    "via X", "call X with", etc. — not just a fixed set of sentence templates.
    """
    result = task
    for py_name, cli_name in _TOOL_NAMES_TO_TRANSLATE:
        # Match backtick-wrapped `tool_name` and bare tool_name as a standalone word
        result = re.sub(
            rf'`{re.escape(py_name)}`|\b{re.escape(py_name)}\b',
            cli_name,
            result,
        )
    return result


def _build_claude_cli_prompt(tools: list[dict[str, Any]]) -> str:
    tool_lines = []
    for t in tools:
        spec = _CLAUDE_NATIVE_TOOL_SPECS.get(t["name"])
        if spec:
            native_name, native_desc = spec
            tool_lines.append(f"  - **{native_name}**: {native_desc}")

    tools_text = "\n".join(tool_lines) if tool_lines else "(no external tools available — use internal reasoning)"

    return f"""You are an autonomous agent. Complete the task using your available tools.

## Available Tools
{tools_text}

## Instructions
- Use the tools listed above to accomplish the task.
- All file paths are relative to the working directory specified in the task.
- When the task is fully complete, include the exact text "TASK_COMPLETE" in your final response.

## Important Notes
- Write final outputs to the file path specified in the task.
- If WebSearch is unavailable and you need to fetch a URL, use Bash with curl (only if Bash is listed above).

DO NOT output JSON tool call blocks. Use your native tool calling instead.
Use EXACTLY the parameter names listed for each tool.

IMPORTANT: After using tools, always produce a final text response. Include TASK_COMPLETE when the entire task is finished."""


def _parse_tool_call(text: str) -> Optional[dict[str, Any]]:
    """Extract a JSON tool-call block using brace counting.

    Uses brace counting instead of regex to handle nested objects/arrays in args
    (e.g. ``{"tool": "write_file", "args": {"content": "a {nested} value"}}``).
    """
    start_pattern = r'\{\s*"tool"\s*:\s*"(\w+)"\s*,\s*"args"\s*:\s*\{'
    match = re.search(start_pattern, text)
    if not match:
        return None

    tool_name = match.group(1)
    outer_start = match.start()

    # Brace-count from the opening '{' to find matching '}'
    depth = 0
    end_pos = -1
    for i in range(outer_start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                end_pos = i
                break

    if end_pos == -1:
        return None

    json_str = text[outer_start:end_pos + 1]
    try:
        obj = json.loads(json_str)
        return {"tool": obj["tool"], "args": obj["args"]}
    except (json.JSONDecodeError, KeyError):
        return None


def _is_persistent_error(result: str) -> bool:
    """Check if a tool result indicates a persistent failure that shouldn't be retried."""
    result_lower = result.lower()
    return any(e in result_lower for e in _PERSISTENT_ERRORS)


def _save_agent_log(
    agent_name: str,
    working_dir: str,
    prompt: str,
    response_text: str,
    success: bool,
    elapsed: float,
    extra: Optional[dict[str, Any]] = None,
) -> str:
    """Save agent conversation to a debug log file in the working directory."""
    log_dir = os.path.join(working_dir, "agent_logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", agent_name)[:40]
    log_file = os.path.join(log_dir, f"{safe_name}_{ts}.json")
    log_entry = {
        "agent": agent_name,
        "timestamp": ts,
        "elapsed_seconds": round(elapsed, 1),
        "success": success,
        "prompt": prompt,
        "response": response_text[:5000],
    }
    if extra:
        log_entry["extra"] = extra
    with open(log_file, "w") as f:
        json.dump(log_entry, f, indent=2, default=str)
    return log_file


def run_agent(
    task: str,
    working_dir: str,
    tools: list[dict[str, Any]],
    tool_map: dict[str, Callable],
    max_steps: int = 10,
    backend: str = "deepseek",
    agent_name: str = "agent",
) -> dict[str, Any]:
    if backend == "claude_cli":
        return _run_with_claude_cli(task, working_dir, tools, agent_name)
    return _run_with_deepseek(task, working_dir, tools, tool_map, max_steps, agent_name)


def _run_with_deepseek(
    task: str,
    working_dir: str,
    tools: list[dict[str, Any]],
    tool_map: dict[str, Callable],
    max_steps: int,
    agent_name: str = "agent",
) -> dict[str, Any]:
    llm = get_llm("deepseek")
    system_prompt = _build_system_prompt(tools, "deepseek")
    t0 = time.time()
    messages: list = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Working directory: {working_dir}\n\nTask: {task}"),
    ]

    iterations = 0
    success = False
    consecutive_errors = 0
    known_unavailable: set[str] = set()

    for i in range(max_steps):
        iterations = i + 1
        steps_left = max_steps - iterations

        # --- Pre-call intervention: force wrap-up if running out of steps ---
        if steps_left <= _FORCE_WRAPUP_THRESHOLD:
            if steps_left == 0:
                urgency = "This is your LAST step — you must wrap up NOW."
            else:
                urgency = f"Only {steps_left} step(s) remaining."
            messages.append(HumanMessage(
                content=(
                    f"IMPORTANT: {urgency} "
                    "Stop retrying tools that have been failing. "
                    "Summarize whatever research findings you have collected so far, "
                    "write your best-effort output file, and respond with TASK_COMPLETE immediately."
                )
            ))

        # --- Pre-call intervention: warn about unavailable tools ---
        if known_unavailable:
            messages.append(HumanMessage(
                content=(
                    f"Note: the following tools are NOT available (do not attempt them again): "
                    f"{', '.join(sorted(known_unavailable))}. "
                    "Use only the tools listed in the system prompt."
                )
            ))
            known_unavailable.clear()

        # --- Pre-call intervention: error spiral detected ---
        if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
            messages.append(HumanMessage(
                content=(
                    "CRITICAL: The last several tool calls all failed with persistent errors "
                    "(timeout, not found, etc.). Do NOT retry those same tools. "
                    "Work with whatever information or files you already have. "
                    "Write your best-effort findings to the output file and respond with TASK_COMPLETE."
                )
            ))
            consecutive_errors = 0

        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)
        messages.append(AIMessage(content=response_text))

        if "TASK_COMPLETE" in response_text:
            success = True
            break

        tool_call = _parse_tool_call(response_text)
        if tool_call:
            tool_name = tool_call["tool"]
            tool_args = tool_call["args"]

            # Check for unknown tool
            if tool_name not in tool_map:
                available = list(tool_map.keys())
                result = f"Error: unknown tool '{tool_name}'. Available: {available}"
                known_unavailable.add(tool_name)
            else:
                func = tool_map[tool_name]
                try:
                    result = func(**tool_args, working_dir=working_dir)
                except TypeError:
                    result = func(**tool_args)

            # Track persistent errors
            if _is_persistent_error(result):
                consecutive_errors += 1
            else:
                consecutive_errors = 0

            messages.append(HumanMessage(content=f"Tool result: {result}"))
        else:
            messages.append(HumanMessage(
                content=(
                    "No valid tool call found in your response. "
                    "Either call a tool using the JSON format or output TASK_COMPLETE if done."
                )
            ))

    # --- Post-loop: if we exhausted steps without TASK_COMPLETE, try one forced summary ---
    if not success and max_steps > 0:
        messages.append(HumanMessage(
            content=(
                "You have exhausted all steps. Write a summary of whatever you accomplished, "
                "including any files you already created. Then output TASK_COMPLETE."
            )
        ))
        try:
            response = llm.invoke(messages)
            response_text = response.content if hasattr(response, "content") else str(response)
            messages.append(AIMessage(content=response_text))
            iterations += 1
            if "TASK_COMPLETE" in response_text:
                success = True
        except Exception:
            pass

    # Save conversation log
    elapsed = time.time() - t0
    final_response = ""
    for msg in reversed(messages):
        content = msg.content if hasattr(msg, "content") else str(msg)
        if len(content) > 50:
            final_response = content
            break
    _save_agent_log(agent_name, working_dir, task, final_response, success, elapsed,
                    extra={"iterations": iterations, "backend": "deepseek"})

    return {
        "messages": messages,
        "iterations": iterations,
        "success": success,
    }


def _run_with_claude_cli(
    task: str,
    working_dir: str,
    tools: list[dict[str, Any]],
    agent_name: str = "agent",
) -> dict[str, Any]:
    llm = get_llm("claude_cli")
    system_prompt = _build_system_prompt(tools, "claude_cli")
    t0 = time.time()

    # Translate Python tool names to Claude CLI native names in the task text
    translated_task = _translate_task_for_claude(task)

    prompt = f"""{system_prompt}

Working directory: {working_dir}

## Task
{translated_task}"""

    allowed = list(dict.fromkeys(
        spec[0] for t in tools if (spec := _CLAUDE_NATIVE_TOOL_SPECS.get(t["name"]))
    ))

    # Try primary invocation
    response = llm.invoke(
        [HumanMessage(content=prompt)],
        allowed_tools=allowed,
        cwd=working_dir,
    )
    response_text = response.content if hasattr(response, "content") else str(response)

    success = "TASK_COMPLETE" in response_text

    # If timeout or failure, retry once with a simplified prompt asking for summary
    is_framework_error = (
        "timed out" in response_text.lower()
        or "error:" in response_text.lower()  # our framework's error prefix, not research content
        or "[stderr]" in response_text.lower()
    )
    if not success and is_framework_error:
        retry_prompt = (
            f"{system_prompt}\n\n"
            f"The previous attempt failed due to a timeout or error. "
            f"Please do your best with whatever is available — skip time-consuming steps. "
            f"Write whatever findings you can to the output file and output TASK_COMPLETE.\n\n"
            f"Working directory: {working_dir}\n\n"
            f"## Task\n{translated_task}"
        )
        try:
            response = llm.invoke(
                [HumanMessage(content=retry_prompt)],
                allowed_tools=allowed,
                cwd=working_dir,
            )
            response_text = response.content if hasattr(response, "content") else str(response)
            success = "TASK_COMPLETE" in response_text
        except Exception:
            pass

    elapsed = time.time() - t0
    _save_agent_log(agent_name, working_dir, prompt, response_text, success, elapsed,
                    extra={"backend": "claude_cli", "retried": not success and is_framework_error})

    return {
        "messages": [HumanMessage(content=prompt), response],
        "iterations": 1,
        "success": success,
    }
