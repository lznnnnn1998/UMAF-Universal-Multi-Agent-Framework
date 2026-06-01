import json
import os
import re
import subprocess
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from claude_config import merge_claude_env
from llm import get_llm

# --- Module-level constants ---
_PERSISTENT_ERRORS = ("timed out", "not found", "no such file", "permission denied")
_FORCE_WRAPUP_THRESHOLD = 3
_MAX_CONSECUTIVE_ERRORS = 3
_WRITE_REMINDER_OFFSET = 4


class CheckpointManager:
    """Version-aware checkpoint and log I/O for a single named agent.

    File patterns under ``{working_dir}/agent_log/``:

    * checkpoint  — ``{safe_name}_v{version:02d}_checkpoint.json``
    * merged      — ``{safe_name}_merged.json``
    * log         — ``{safe_name}_{timestamp}.json``
    """

    def __init__(self, working_dir: str, agent_name: str,
                 log_subdir: str = "agent_log") -> None:
        self.working_dir = working_dir
        self.agent_name = agent_name
        self.log_dir = os.path.join(working_dir, log_subdir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", agent_name)[:40]

    # -- Path builders -------------------------------------------------------

    def path_for_version(self, version: int) -> str:
        return os.path.join(
            self.log_dir, f"{self.safe_name}_v{version:02d}_checkpoint.json",
        )

    # -- Save ----------------------------------------------------------------

    def save(self, version: int, messages: list, iterations: int,
             max_steps: int, has_written_output: bool, task: str = "",
             tools: list[dict[str, Any]] | None = None,
             extra: dict[str, Any] | None = None) -> str:
        serialized = CheckpointManager.serialize_messages(messages)
        data: dict[str, Any] = {
            "agent_name": self.agent_name,
            "version": version,
            "iterations": iterations,
            "max_steps": max_steps,
            "has_written_output": has_written_output,
            "task": task,
            "tools": tools or [],
            "messages": serialized,
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        }
        if extra:
            data["extra"] = extra
        path = self.path_for_version(version)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    # -- Load ----------------------------------------------------------------

    def load(self, version: int) -> dict[str, Any] | None:
        path = self.path_for_version(version)
        if not os.path.exists(path):
            # Migration: rename legacy unversioned checkpoint
            legacy = os.path.join(
                self.log_dir, f"{self.safe_name}_checkpoint.json",
            )
            if os.path.exists(legacy):
                os.rename(legacy, path)
        return CheckpointManager._deserialize_file(path)

    def load_previous(self, current_version: int) -> dict[str, Any] | None:
        """Highest version strictly below *current_version*.

        V2 loads V1, V3 loads V2 — the primary resume path.
        """
        versions = self.list_versions()
        candidates = [v for v in versions if v < current_version]
        if not candidates:
            return None
        return self.load(max(candidates))

    def list_versions(self) -> list[int]:
        prefix = f"{self.safe_name}_v"
        suffix = "_checkpoint.json"
        versions: list[int] = []
        if not os.path.isdir(self.log_dir):
            return versions
        for fname in os.listdir(self.log_dir):
            if fname.startswith(prefix) and fname.endswith(suffix):
                m = re.match(r".*_v(\d+)_checkpoint\.json", fname)
                if m:
                    versions.append(int(m.group(1)))
        return sorted(versions)

    # -- Merge ---------------------------------------------------------------

    def merge(self) -> str | None:
        import hashlib

        versions = self.list_versions()
        if not versions:
            return None

        seen: set[str] = set()
        merged_msgs: list[dict[str, str]] = []
        total_iters = 0
        version_summary: list[dict[str, Any]] = []

        for v in versions:
            data = self.load(v)
            if data is None:
                continue
            msgs = data.get("messages", [])
            new_count = 0
            for m in msgs:
                mtype = m.get("type", "") if isinstance(m, dict) else type(m).__name__
                mcontent = m.get("content", "") if isinstance(m, dict) else (m.content if hasattr(m, "content") else str(m))
                h = hashlib.md5(
                    (mtype + mcontent[:200]).encode(),
                ).hexdigest()
                if h not in seen:
                    seen.add(h)
                    merged_msgs.append({"type": mtype, "content": mcontent})
                    new_count += 1
            total_iters += data.get("iterations", 0)
            version_summary.append({
                "version": v,
                "iterations": data.get("iterations", 0),
                "messages": len(msgs),
                "new_messages": new_count,
                "timestamp": data.get("timestamp", ""),
                "has_written_output": data.get("has_written_output", False),
            })

        merged = {
            "agent_name": self.agent_name,
            "versions": len(versions),
            "total_iterations": total_iters,
            "merged_message_count": len(merged_msgs),
            "version_summary": version_summary,
            "messages": merged_msgs,
            "merged_at": time.strftime("%Y%m%d_%H%M%S"),
        }
        merge_path = os.path.join(
            self.log_dir, f"{self.safe_name}_merged.json",
        )
        with open(merge_path, "w") as f:
            json.dump(merged, f, indent=2, default=str)
        return merge_path

    # -- Logging -------------------------------------------------------------

    def save_log(self, task: str, elapsed: float, success: bool,
                 messages: list, extra: dict[str, Any] | None = None) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(
            self.log_dir, f"{self.safe_name}_{ts}.json",
        )
        final_response = ""
        for msg in reversed(messages):
            if type(msg).__name__ != "AIMessage":
                continue
            content = msg.content if hasattr(msg, "content") else str(msg)
            if len(content) > 50:
                final_response = content[:10000]
                break
        if not final_response:
            for msg in reversed(messages):
                content = msg.content if hasattr(msg, "content") else str(msg)
                if len(content) > 50:
                    final_response = content[:10000]
                    break
        entry: dict[str, Any] = {
            "agent": self.agent_name,
            "timestamp": ts,
            "elapsed_seconds": round(elapsed, 1),
            "success": success,
            "prompt": task,
            "response": final_response,
        }
        if extra:
            entry["extra"] = extra
        with open(log_file, "w") as f:
            json.dump(entry, f, indent=2, default=str)
        return log_file

    # -- Static helpers ------------------------------------------------------

    @staticmethod
    def serialize_messages(messages: list) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for m in messages:
            content = m.content if hasattr(m, "content") else str(m)
            out.append({"type": type(m).__name__, "content": content})
        return out

    @staticmethod
    def deserialize_messages(raw: list[dict[str, str]]) -> list:
        type_map = {
            "SystemMessage": SystemMessage,
            "HumanMessage": HumanMessage,
            "AIMessage": AIMessage,
        }
        messages: list = []
        for m in raw:
            cls = type_map.get(m["type"])
            if cls:
                messages.append(cls(content=m["content"]))
        return messages

    @staticmethod
    def _deserialize_file(path: str) -> dict[str, Any] | None:
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        data["messages"] = CheckpointManager.deserialize_messages(
            data.get("messages", []),
        )
        return data


@dataclass
class AgentResult:
    """Structured result from an agent run."""
    messages: list = field(default_factory=list)
    iterations: int = 0
    success: bool = False


class BaseAgent:
    """Autonomous agent that loops on tool calls until task completion.

    Two backends: deepseek (JSON tool-call loop with circuit breakers)
    and claude_cli (single subprocess invocation with retry).
    """

    # Tool name mapping: framework → Claude CLI native
    _CLAUDE_NATIVE_TOOL_SPECS: dict[str, tuple[str, str]] = {
        "read_file": ("Read", "Read a file. Parameters: file_path (str, required)"),
        "write_file": ("Write", "Write content to a file. Parameters: file_path (str), content (str)"),
        "write_lines": ("Write", "Write a list of lines to a file. Parameters: file_path (str), lines (list of str). Join lines with newlines when writing."),
        "run_command": ("Bash", "Run a shell command. Parameters: command (str, required), description (str, optional)"),
        "web_search": ("WebSearch", "Search the web. Parameters: query (str, required)"),
        "call_claude": ("Bash", "Use Bash for running commands and scripts. For deep reasoning or analysis, think through the problem yourself — you ARE the reasoning engine. Do NOT spawn nested claude -p calls."),
        "web_fetch": ("Bash", "Fetch content from a URL and return as plain text. Use for reading papers from arxiv.org, articles, and documentation. Run: python3 -c \"import urllib.request; req=urllib.request.Request(URL, headers={'User-Agent':'Mozilla/5.0 (compatible; UMAF/1.0)'}); print(urllib.request.urlopen(req, timeout=20).read().decode('utf-8', errors='replace')[:8000])\""),
        "download_file": ("Bash", "Download content from a URL and save to a local file. Use BEFORE Read for arxiv.org papers. Run: python3 -c \"import urllib.request; req=urllib.request.Request(URL, headers={'User-Agent':'Mozilla/5.0 (compatible; UMAF/1.0)'}); data=urllib.request.urlopen(req, timeout=30).read(); open('OUTPUT_PATH','wb').write(data); print(f'Downloaded {{len(data)//1024}}KB to OUTPUT_PATH')\" — replace URL and OUTPUT_PATH with actual values."),
    }

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

    def __init__(
        self,
        backend: str = "deepseek",
        working_dir: str = ".",
        tools: list[dict[str, Any]] | None = None,
        tool_map: dict[str, Callable] | None = None,
        max_steps: int = 10,
        agent_name: str = "agent",
        enable_checkpoint: bool = True,
        version: int = 1,
    ):
        self.backend = backend
        self.working_dir = working_dir
        self.tools = tools or []
        self.tool_map = tool_map or {}
        self.max_steps = max_steps
        self.agent_name = agent_name
        self.enable_checkpoint = enable_checkpoint
        self.version = version
        self._ckpt = CheckpointManager(working_dir, agent_name)

        # Runtime state
        self.messages: list = []
        self.iterations = 0
        self.has_written_output = False
        self.success = False
        self.consecutive_errors = 0
        self.known_unavailable: set[str] = set()
        self._t0: float = 0.0
        self._last_parse_error: str | None = None

    # --- Public API ---

    def run(self, task: str, resume_from: str | None = None) -> dict[str, Any]:
        """Execute the agent on a task. Returns {messages, iterations, success}."""
        if self.backend == "claude_cli":
            return self._run_claude_cli(task)
        return self._run_deepseek(task, resume_from)

    # --- Prompt builders ---

    def _build_deepseek_prompt(self) -> str:
        tool_descriptions = []
        for t in self.tools:
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

## Write Strategy
For `write_lines`, break large files into MULTIPLE smaller calls (max ~40 lines each).
Smaller JSON blocks are more likely to parse correctly. Write the file in sections:
- First call writes the first 30-40 lines
- Next call(s) append the remaining sections
Each call to the same path overwrites, so plan your chunks accordingly.

## Working Directory
All file paths are relative to the provided working directory. Do NOT prepend the working directory name to any path or command.
- `write_file` / `write_lines` / `read_file`: the path is resolved relative to the working directory automatically.
- `run_command`: commands ALREADY execute inside the working directory. Do NOT `cd <working_dir>` — you are already there. Use paths like `modules/x/` not `<working_dir>/modules/x/`."""

    def _build_claude_cli_prompt(self) -> str:
        tool_lines = []
        for t in self.tools:
            spec = self._CLAUDE_NATIVE_TOOL_SPECS.get(t["name"])
            if spec:
                native_name, native_desc = spec
                tool_lines.append(f"  - **{native_name}**: {native_desc}")

        tools_text = "\n".join(tool_lines) if tool_lines else "(no external tools available — use internal reasoning)"

        return f"""You are an autonomous agent. Complete the task using your available tools.

## Available Tools
{tools_text}

## Instructions
- Use the tools listed above to accomplish the task.
- All file paths are relative to the working directory specified in the task. Do NOT prepend the working directory name.
- When the task is fully complete, include the exact text "TASK_COMPLETE" in your final response.

## Important Notes
- Write final outputs to the file path specified in the task.
- If WebSearch is unavailable and you need to fetch a URL, use Bash with curl (only if Bash is listed above).

DO NOT output JSON tool call blocks. Use your native tool calling instead.
Use EXACTLY the parameter names listed for each tool.

IMPORTANT: After using tools, always produce a final text response. Include TASK_COMPLETE when the entire task is finished."""

    @staticmethod
    def _translate_task_for_claude(task: str) -> str:
        for py_name, cli_name in BaseAgent._TOOL_NAMES_TO_TRANSLATE:
            task = re.sub(
                rf'`{re.escape(py_name)}`|\b{re.escape(py_name)}\b',
                cli_name,
                task,
            )
        return task

    # --- Tool call parsing ---

    def _parse_tool_call(self, text: str) -> dict[str, Any] | None:
        start_pattern = r'\{\s*"tool"\s*:\s*"(\w+)"\s*,\s*"args"\s*:\s*\{'
        matches = list(re.finditer(start_pattern, text))
        if not matches:
            return None

        errors: list[str] = []

        # Try each tool-call candidate in the response (last one first — it's
        # usually the most recent and least likely to be a code example).
        for match in reversed(matches):
            tool_name = match.group(1)
            outer_start = match.start()

            # Brace-count with JSON string tracking — content values may contain
            # unescaped braces that would throw off naive depth counting.
            depth = 0
            in_string = False
            escape_next = False
            end_pos = -1
            for i in range(outer_start, len(text)):
                ch = text[i]
                if escape_next:
                    escape_next = False
                    continue
                if ch == '\\':
                    escape_next = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end_pos = i
                        break

            if end_pos == -1:
                continue

            json_str = text[outer_start:end_pos + 1]
            try:
                obj = json.loads(json_str)
                return {"tool": obj["tool"], "args": obj["args"]}
            except json.JSONDecodeError as e:
                snippet_start = max(0, e.pos - 40)
                snippet_end = min(len(json_str), e.pos + 40)
                errors.append(
                    f"[{tool_name}] pos {e.pos}: {e.msg} "
                    f"near ...{json_str[snippet_start:snippet_end]}..."
                )
            except KeyError:
                errors.append(f"[{tool_name}] missing 'tool' or 'args' key")

        if errors:
            self._last_parse_error = "; ".join(errors[:3])
        return None

    @staticmethod
    def _is_persistent_error(result: str) -> bool:
        result_lower = result.lower()
        return any(e in result_lower for e in _PERSISTENT_ERRORS)

    # --- DeepSeek agent loop ---

    def _run_deepseek(self, task: str, resume_from: str | None = None) -> dict[str, Any]:
        llm = get_llm("deepseek")
        system_prompt = self._build_deepseek_prompt()
        self._t0 = time.time()

        if resume_from:
            ck = _load_checkpoint(resume_from)
            if ck:
                self.messages = ck["messages"]
                # Cap iterations so resumed agents always have at least 5 steps
                self.iterations = min(ck.get("iterations", 0), max(0, self.max_steps - 5))
                self.has_written_output = ck.get("has_written_output", False)
                if not self.messages or type(self.messages[0]).__name__ != "SystemMessage":
                    self.messages.insert(0, SystemMessage(content=system_prompt))
            else:
                self.messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Working directory: {self.working_dir}\n\nTask: {task}"),
                ]
        elif self.version > 1 and self.enable_checkpoint:
            ck = self._ckpt.load_previous(self.version)
            if ck:
                self.messages = ck["messages"]
                self.iterations = 0  # fresh step budget on version retry
                self.has_written_output = ck.get("has_written_output", False)
                if not self.messages or type(self.messages[0]).__name__ != "SystemMessage":
                    self.messages.insert(0, SystemMessage(content=system_prompt))
                # Inject a context message so the agent knows this is a retry
                self.messages.append(HumanMessage(
                    content=f"[System: This is version {self.version} retry. "
                            f"You are resuming from version {self.version - 1}. "
                            f"Review what went wrong in the previous version and improve. "
                            f"Working directory: {self.working_dir}\n\nTask: {task}]"
                ))
            else:
                self.messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Working directory: {self.working_dir}\n\nTask: {task}"),
                ]
        else:
            self.messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Working directory: {self.working_dir}\n\nTask: {task}"),
            ]

        self.consecutive_errors = 0
        self.known_unavailable = set()
        self.success = False
        write_reminder_step = max(1, self.max_steps - _WRITE_REMINDER_OFFSET)

        for i in range(self.iterations, self.max_steps):
            self.iterations = i + 1
            steps_left = self.max_steps - self.iterations

            self._maybe_nudge_write(write_reminder_step)
            self._maybe_force_wrapup(steps_left)
            self._maybe_warn_unavailable_tools()
            self._maybe_detect_error_spiral()

            response = llm.invoke(self.messages)
            response_text = response.content if hasattr(response, "content") else str(response)
            self.messages.append(AIMessage(content=response_text))

            has_task_complete = "TASK_COMPLETE" in response_text

            tool_call = self._parse_tool_call(response_text)
            if tool_call:
                self._execute_tool(tool_call, task)

            if has_task_complete:
                self.success = True
                break

            if not tool_call:
                detail = ""
                if self._last_parse_error:
                    detail = f"\n\nParse error details: {self._last_parse_error}"
                    self._last_parse_error = None
                self.messages.append(HumanMessage(
                    content="No valid tool call found in your response. "
                            "Either call a tool using the JSON format or output TASK_COMPLETE if done."
                            f"{detail}\n\n"
                            "If your write_file content is very large, ensure all special characters "
                            "are properly escaped: backslashes \\\\, double-quotes \\\", newlines \\n, "
                            "and tabs \\t. Use a JSON-valid string for the content field."
                ))

        # Post-loop forced summary
        if not self.success and self.max_steps > 0:
            self._force_final_write(llm, task)

        # Save final state
        self._save_checkpoint(extra={"success": self.success})
        elapsed = time.time() - self._t0
        self._save_log(task, elapsed)

        return {
            "messages": self.messages,
            "iterations": self.iterations,
            "success": self.success,
        }

    def _execute_tool(self, tool_call: dict, task: str):
        tool_name = tool_call["tool"]
        tool_args = tool_call["args"]

        if tool_name not in self.tool_map:
            available = list(self.tool_map.keys())
            result = f"Error: unknown tool '{tool_name}'. Available: {available}"
            self.known_unavailable.add(tool_name)
        else:
            func = self.tool_map[tool_name]
            result = ""
            try:
                result = func(**tool_args, working_dir=self.working_dir)
            except TypeError as e_working_dir:
                # Retry without working_dir (some tools accept it, some don't)
                try:
                    result = func(**tool_args)
                except TypeError as e_no_wd:
                    result = (
                        f"Error: invalid arguments for '{tool_name}'. "
                        f"Got: {tool_args}. Details: {e_no_wd}"
                    )

        if tool_name in ("write_file", "write_lines") and not result.startswith("Error"):
            self.has_written_output = True

        if self._is_persistent_error(result):
            self.consecutive_errors += 1
        else:
            self.consecutive_errors = 0

        self.messages.append(HumanMessage(content=f"Tool result: {result}"))

        if self.enable_checkpoint:
            self._save_checkpoint(task=task)

    # --- Interventions ---

    def _maybe_nudge_write(self, reminder_step: int):
        if not self.has_written_output and self.iterations >= reminder_step:
            remaining = self.max_steps - self.iterations
            self.messages.append(HumanMessage(content=(
                f"You have used {self.iterations}/{self.max_steps} steps and have NOT yet written "
                "your output file. The primary goal is to produce the output file "
                "— not just to keep searching. Call write_file within the next "
                f"{remaining} step(s) with whatever you have so far. "
                "You can always refine it afterward if steps remain."
            )))

    def _maybe_force_wrapup(self, steps_left: int):
        if steps_left <= _FORCE_WRAPUP_THRESHOLD:
            urgency = "You have exhausted your step budget — this is your FINAL chance." if steps_left == 0 else f"Only {steps_left} step(s) remaining."
            if self.has_written_output:
                # Files already exist — just need to conclude
                self.messages.append(HumanMessage(content=(
                    f"CRITICAL: {urgency}\n"
                    "You have ALREADY written output files — do NOT call write_file again. "
                    "Your ONLY task now is to verify the files exist (with read_file) and "
                    "then output TASK_COMPLETE immediately. Do NOT make any more tool calls "
                    "other than read_file to verify your work exists."
                )))
            else:
                self.messages.append(HumanMessage(content=(
                    f"CRITICAL: {urgency}\n"
                    "Do NOT call web_search, download_file, web_fetch "
                    "run_command, or call_claude anymore. You have already collected "
                    "enough research material. Your ONLY task now is to call write_file "
                    "with your best-effort findings, then output TASK_COMPLETE. "
                    "Write the file FIRST, then say TASK_COMPLETE — do NOT skip writing the file."
                )))

    def _maybe_warn_unavailable_tools(self):
        if self.known_unavailable:
            self.messages.append(HumanMessage(content=(
                f"Note: the following tools are NOT available (do not attempt them again): "
                f"{', '.join(sorted(self.known_unavailable))}. "
                "Use only the tools listed in the system prompt."
            )))
            self.known_unavailable.clear()

    def _maybe_detect_error_spiral(self):
        if self.consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
            self.messages.append(HumanMessage(content=(
                "CRITICAL: The last several tool calls all failed with persistent errors "
                "(timeout, not found, etc.). Do NOT retry those same tools. "
                "Work with whatever information or files you already have. "
                "Write your best-effort findings to the output file and respond with TASK_COMPLETE."
            )))
            self.consecutive_errors = 0

    def _force_final_write(self, llm, task: str):
        if self.has_written_output:
            # Files were written but TASK_COMPLETE wasn't output. Just ask for it.
            self.messages.append(HumanMessage(content=(
                "You have exhausted all your steps but you HAVE already written output files. "
                "Do NOT call write_file or any other tool. Simply output TASK_COMPLETE now "
                "to signal that your work is done. Just say TASK_COMPLETE — nothing else."
            )))
        else:
            self.messages.append(HumanMessage(content=(
                "You have exhausted all your steps. You did NOT write the output file yet, "
                "which was the primary goal of this task. Call write_lines IMMEDIATELY with "
                "your best-effort findings — use whatever notes and information you have gathered. "
                "Write the file now as a single JSON block, then output TASK_COMPLETE. "
                "IMPORTANT: Use the lines array (list of strings) for code — each line is a separate "
                "string, which avoids multi-line escaping problems. "
                "If you skip writing the file again, the task will fail completely."
            )))
        try:
            response = llm.invoke(self.messages)
            response_text = response.content if hasattr(response, "content") else str(response)
            self.messages.append(AIMessage(content=response_text))
            self.iterations += 1

            # Try to execute any tool call in the final response
            tool_call = self._parse_tool_call(response_text)
            if tool_call:
                self._execute_tool(tool_call, task)

            if "TASK_COMPLETE" in response_text:
                self.success = True

            # One more retry if write_file/write_lines still didn't happen
            if not self.has_written_output:
                self.messages.append(HumanMessage(content=(
                    "You STILL haven't written your output. Output ONLY a single JSON block: "
                    '{"tool": "write_lines", "args": {"path": "YOUR_OUTPUT_PATH", "lines": ["line1", "line2", ...]}} '
                    "Each line is a separate string — this avoids escaping issues with multi-line content. "
                    "Then output TASK_COMPLETE."
                )))
                response2 = llm.invoke(self.messages)
                response_text2 = response2.content if hasattr(response2, "content") else str(response2)
                self.messages.append(AIMessage(content=response_text2))
                self.iterations += 1
                tool_call2 = self._parse_tool_call(response_text2)
                if tool_call2:
                    self._execute_tool(tool_call2, task)
                if "TASK_COMPLETE" in response_text2:
                    self.success = True

            if self.enable_checkpoint:
                self._save_checkpoint(task=task, extra={"post_loop_forced": True})
        except Exception:
            pass

    # --- Claude CLI backend ---

    def _run_claude_cli(self, task: str) -> dict[str, Any]:
        llm = get_llm("claude_cli")
        system_prompt = self._build_claude_cli_prompt()
        self._t0 = time.time()
        translated_task = self._translate_task_for_claude(task)
        timeout = getattr(llm, "timeout", 600)

        # Build resume context from previous version checkpoint
        resume_context = ""
        if self.version > 1 and self.enable_checkpoint:
            prev = self._ckpt.load_previous(self.version)
            if prev:
                prev_msgs = prev.get("messages", [])
                # Extract the last few assistant responses as context
                snippets = []
                for m in prev_msgs[-8:]:
                    mtype = m.get("type", "") if isinstance(m, dict) else type(m).__name__
                    mcontent = m.get("content", "") if isinstance(m, dict) else (
                        m.content if hasattr(m, "content") else str(m)
                    )
                    if mtype == "AIMessage":
                        snippets.append(mcontent[:500])
                if snippets:
                    resume_context = (
                        "\n\n[System: This is version " + str(self.version) + " retry. "
                        "The previous attempt ran out of time. Below is a summary of prior "
                        "work. Pick up where it left off — focus on writing the output "
                        "file immediately rather than re-doing all research.]\n\n"
                        "## Prior Work Summary\n" + "\n---\n".join(snippets[-3:]) + "\n"
                    )

        def _build_prompt(prefix: str = "") -> str:
            return f"{prefix}{system_prompt}\n\nWorking directory: {self.working_dir}\n\n## Task\n{translated_task}{resume_context}"

        allowed = list(dict.fromkeys(
            spec[0] for t in self.tools if (spec := self._CLAUDE_NATIVE_TOOL_SPECS.get(t["name"]))
        ))

        def _stream_one(prompt: str) -> tuple[str, bool, list]:
            """Run one claude -p invocation with stream-json output.

            Parses events as they arrive, records messages incrementally, and
            flushes a checkpoint after every assistant/user event so intermediate
            state is visible even if the process is killed mid-run.

            Returns (final_text, success, stream_messages).
            """
            cmd = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose",
                   "--permission-mode", "bypassPermissions"]
            if allowed:
                cmd.extend(["--allowedTools", ",".join(allowed)])

            try:
                stderr_file = tempfile.NamedTemporaryFile(
                    mode="w+", delete=False, suffix=".stderr",
                )
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=stderr_file,
                    text=True,
                    env=merge_claude_env(),
                    cwd=self.working_dir,
                )
            except FileNotFoundError:
                return "Error: claude CLI not found", False, []

            messages: list = []
            final_text = ""
            success = False
            tool_count = 0
            wrote_file = False

            def _flush_ckpt():
                """Snapshot current state to a checkpoint file."""
                if self.enable_checkpoint:
                    self.messages = [HumanMessage(content=prompt)] + list(messages)
                    self._save_checkpoint(task=task)
                    self.messages = []

            # Hard timeout via threading.Timer: proc.stdout.readline() blocks
            # during long-running tool executions (e.g., pytest taking 10+ min),
            # so an in-loop elapsed check can't fire until the next line arrives.
            killed = [False]
            timer = threading.Timer(timeout, lambda: (
                proc.kill(), killed.__setitem__(0, True)
            ))
            timer.start()

            try:
                for line in proc.stdout:
                    elapsed = time.time() - self._t0
                    if elapsed > timeout:
                        proc.kill()
                        final_text = f"Error: claude CLI timed out after {timeout}s"
                        _flush_ckpt()
                        break

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type")

                    if event_type == "assistant":
                        content_blocks = event.get("message", {}).get("content", [])
                        text_parts = []
                        tool_uses = []
                        for block in content_blocks:
                            if block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                            elif block.get("type") == "tool_use":
                                tool_uses.append(block)
                                tool_count += 1

                        text = "".join(text_parts)
                        final_text += text

                        # Store as plain text for backward compatibility with
                        # parse_result (which scans AIMessage content for JSON arrays).
                        if text:
                            messages.append(AIMessage(content=text))

                        # Record tool calls in a companion HumanMessage so they
                        # appear in checkpoints but don't break text-based parsers.
                        for tu in tool_uses:
                            tool_name = tu.get("name", "?")
                            if tool_name in ("Write", "write_file"):
                                wrote_file = True
                            messages.append(HumanMessage(
                                content=f"[tool_call: {tool_name} "
                                        f"{json.dumps(tu.get('input', {}), ensure_ascii=False)}]"
                            ))

                        # Only check the current turn's text for TASK_COMPLETE,
                        # not accumulated text — earlier turns may mention
                        # "TASK_COMPLETE" in paraphrased instructions.
                        if "TASK_COMPLETE" in text:
                            success = True
                            _flush_ckpt()
                            break

                        _flush_ckpt()

                    elif event_type == "user":
                        # Tool result from the CLI
                        content = event.get("message", {}).get("content", [])
                        for block in content:
                            if block.get("type") == "tool_result":
                                result_content = block.get("content", "")
                                if isinstance(result_content, list):
                                    result_content = "\n".join(
                                        r.get("text", "") if isinstance(r, dict) else str(r)
                                        for r in result_content
                                    )
                                messages.append(HumanMessage(
                                    content=f"[tool_result: {result_content[:2000]}]"
                                ))
                        _flush_ckpt()

                    elif event_type == "result":
                        result_text = event.get("result", "")
                        if result_text and result_text not in final_text:
                            final_text = result_text
                        if "TASK_COMPLETE" in result_text:
                            success = True
                        _flush_ckpt()
                        break

            finally:
                timer.cancel()
                if killed[0] and "timed out" not in final_text:
                    final_text = f"Error: claude CLI timed out after {timeout}s"
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except (subprocess.TimeoutExpired, OSError):
                    proc.kill()
                    proc.wait()
                # Read stderr from temp file for error diagnosis
                try:
                    stderr_file.flush()
                    stderr_file.seek(0)
                    stderr_output = stderr_file.read()
                    if stderr_output and stderr_output.strip():
                        if not final_text or "Error:" in final_text:
                            final_text += f"\n[stderr]: {stderr_output.strip()[:2000]}"
                except (IOError, OSError):
                    pass
                finally:
                    try:
                        os.unlink(stderr_file.name)
                    except OSError:
                        pass

            return final_text.strip() or "(no response)", success, messages, wrote_file

        prompt = _build_prompt()
        final_text, self.success, stream_messages, wrote_file = _stream_one(prompt)

        is_error = (
            "timed out" in final_text.lower()
            or "error:" in final_text.lower()
        )
        retried = False
        if not self.success and is_error:
            retried = True
            prompt = _build_prompt(
                "The previous attempt failed due to a timeout or error. "
                "Do your best with whatever is available — skip time-consuming steps. "
                "Write whatever findings you can and output TASK_COMPLETE.\n\n"
            )
            self._t0 = time.time()  # reset clock for retry
            final_text, self.success, stream_messages, wrote_file = _stream_one(prompt)
            # Recompute error flag from the retry result
            is_error = (
                "timed out" in final_text.lower()
                or "error:" in final_text.lower()
            )

        self.messages = [HumanMessage(content=prompt)] + stream_messages
        self.iterations = sum(
            1 for m in self.messages if type(m).__name__ == "AIMessage"
        )

        # claude_cli stream doesn't expose individual tool calls to the
        # framework loop, so we detect writes from the recorded messages.
        if wrote_file:
            self.has_written_output = True

        elapsed = time.time() - self._t0
        self._save_log(translated_task, elapsed, extra={
            "backend": "claude_cli_stream", "retried": retried,
        })
        self._save_checkpoint(task=task, extra={
            "success": self.success, "backend": "claude_cli_stream", "retried": retried,
        })

        return {
            "messages": self.messages,
            "iterations": self.iterations,
            "success": self.success,
        }

    # --- Checkpoint / resume ---

    def _checkpoint_path(self) -> str:
        return self._ckpt.path_for_version(self.version)

    def _save_checkpoint(self, task: str = "", extra: dict[str, Any] | None = None):
        if not self.enable_checkpoint:
            return ""
        return self._ckpt.save(
            version=self.version,
            messages=self.messages,
            iterations=self.iterations,
            max_steps=self.max_steps,
            has_written_output=self.has_written_output,
            task=task,
            tools=self.tools,
            extra=extra,
        )

    # --- Logging ---

    def _save_log(self, task: str, elapsed: float, extra: dict[str, Any] | None = None):
        return self._ckpt.save_log(
            task=task,
            elapsed=elapsed,
            success=self.success,
            messages=self.messages,
            extra=extra,
        )


# --- Module-level helpers (backward-compatible) ---

def _checkpoint_path(working_dir: str, agent_name: str) -> str:
    """Return the path for version 1 checkpoint of the given agent."""
    return CheckpointManager(working_dir, agent_name).path_for_version(1)


def _load_checkpoint(path: str) -> dict[str, Any] | None:
    """Load a saved checkpoint. Returns None if checkpoint is missing or invalid."""
    return CheckpointManager._deserialize_file(path)


class AgentRole(ABC):
    """Abstract base for a specialized agent role.

    Each concrete role encapsulates:
    - Which tools are available (per backend)
    - How to build the task prompt (per backend)
    - How to parse the raw agent result into structured output

    Subclasses implement tools_for_backend(), build_task(), and optionally
    parse_result(). Call execute() to run the full lifecycle.
    """

    agent_name: str = "agent"
    max_steps: int = 10

    @abstractmethod
    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        """Return the tool specs available to this role for the given backend."""

    @abstractmethod
    def build_task(self, backend: str, **context: Any) -> str:
        """Build the full task prompt for the agent."""

    def parse_result(self, result: AgentResult, working_dir: str, **context: Any) -> Any:
        """Parse the raw agent result into structured output. Override in subclasses."""
        return result

    def execute(
        self,
        working_dir: str,
        backend: str = "deepseek",
        resume_from: str | None = None,
        version: int = 1,
        **context: Any,
    ) -> Any:
        """Run the full agent lifecycle: build task → run agent → parse result."""
        from llm import get_llm_provider
        from tools import ToolRegistry

        provider = get_llm_provider(backend)
        agent = BaseAgent(
            backend=backend,
            working_dir=working_dir,
            tools=self.tools_for_backend(backend),
            tool_map=ToolRegistry.get_map(),
            max_steps=self.max_steps,
            agent_name=self.agent_name,
            enable_checkpoint=True,
            version=version,
        )
        raw = agent.run(
            task=self.build_task(backend, working_dir=working_dir, **context),
            resume_from=resume_from,
        )
        result = AgentResult(
            messages=raw["messages"],
            iterations=raw["iterations"],
            success=raw["success"],
        )
        return self.parse_result(result, working_dir, **context)


def run_agent(
    task: str,
    working_dir: str,
    tools: list[dict[str, Any]],
    tool_map: dict[str, Callable],
    max_steps: int = 10,
    backend: str = "deepseek",
    agent_name: str = "agent",
    resume_from: str | None = None,
    version: int = 1,
) -> dict[str, Any]:
    """Convenience function — creates a BaseAgent and runs it.

    Kept for backward compatibility with existing callers.
    """
    agent = BaseAgent(
        backend=backend,
        working_dir=working_dir,
        tools=tools,
        tool_map=tool_map,
        max_steps=max_steps,
        agent_name=agent_name,
        enable_checkpoint=resume_from is not None,
        version=version,
    )
    return agent.run(task=task, resume_from=resume_from)


# ═══════════════════════════════════════════════════════════════════════════
# Base Decomposer — shared by coderpp and research head agents
# ═══════════════════════════════════════════════════════════════════════════

class BaseDecomposerRole(AgentRole):
    """Abstract base for head agents that decompose a complex input into sub-tasks.

    Shared by ``CoderPPDecomposerRole`` and ``ResearchDecomposerRole``.
    Subclasses override template methods to supply pipeline-specific role text,
    sizing guides, JSON schemas, and fallback logic.  The base class handles
    JSON extraction (bracket-counting parser), the parse-result flow (agent
    messages → disk files → fallback), and backend-aware prompt assembly.
    """

    # -- Template methods (override in subclasses) -------------------------

    def _role_prompt(self, input_spec: str, **context: Any) -> str:
        """Opening role / persona paragraph for the decomposer."""
        raise NotImplementedError

    def _sizing_guide(self) -> str:
        """Complexity-based sizing rules (markdown bullet list)."""
        raise NotImplementedError

    def _sub_unit_requirements(self) -> str:
        """Requirements each sub-unit must satisfy (markdown bullet list)."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _json_template() -> str:
        """JSON array template string showing the expected schema."""

    def _extra_phases(self) -> str:
        """Optional extra phases (e.g. environment setup) after decomposition."""
        return ""

    def _disk_fallback_paths(self, working_dir: str) -> list[str]:
        """Files to check when JSON is not found in the agent's response text."""
        _ = working_dir
        return ["decomposition.json"]

    @staticmethod
    @abstractmethod
    def _fallback_decompose(input_spec: str) -> list[dict[str, Any]]:
        """Fallback decomposition when every other extraction path fails."""

    def _backend_instructions(self, backend: str) -> str:
        """Backend-specific suffix appended to the prompt."""
        if backend == "claude_cli":
            return (
                "Use your own knowledge — do NOT search the web. "
                "If a .tex file was provided in the requirement, read it first "
                "to extract decomposition ideas. Output the JSON array, then "
                "complete any additional phases. End with TASK_COMPLETE."
            )
        return (
            "If the requirement references a .tex file, read it first to extract "
            "relevant sections. Output the JSON array, complete any additional "
            "phases, and end with TASK_COMPLETE."
        )

    # -- build_task (concrete) --------------------------------------------

    def build_task(self, backend: str, input_spec: str = "", **context: Any) -> str:
        """Assemble the full decomposition prompt from template methods."""
        common = (
            f"{self._role_prompt(input_spec, **context)}\n\n"
            f"Requirement: {input_spec}\n\n"
            f"## Decomposition Sizing\n"
            f"{self._sizing_guide()}\n\n"
            f"## Sub-Unit Requirements\n"
            f"{self._sub_unit_requirements()}\n\n"
            f"Output the decomposition as a JSON array:\n"
            f"```json\n{self._json_template()}\n```\n"
            f"IMPORTANT: The JSON array MUST appear INLINE in your response text. "
            f"Do NOT replace it with a markdown table — the pipeline parser reads "
            f"the JSON from your response. Also use write_file to save a backup "
            f'copy to "decomposition.json".'
        )
        extra = self._extra_phases()
        if extra:
            common += f"\n\n{extra}"

        return f"{common}\n\n{self._backend_instructions(backend)}"

    # -- parse_result (concrete) ------------------------------------------

    def parse_result(self, result: AgentResult, working_dir: str,
                     input_spec: str = "", **context: Any) -> list[dict[str, Any]]:
        """Extract sub-tasks: agent messages → disk files → fallback decompose."""
        sub_tasks: list[dict[str, Any]] = []

        # 1. Try extracting JSON from agent response messages
        for msg in reversed(result.messages):
            content = msg.content if hasattr(msg, "content") else str(msg)
            json_str = self._extract_json_array(content)
            if json_str:
                try:
                    sub_tasks = json.loads(json_str)
                    if isinstance(sub_tasks, list) and len(sub_tasks) > 0:
                        break
                except json.JSONDecodeError:
                    continue

        # 2. Try reading decomposition files from disk (agent may write to
        #    file without including JSON inline, or use alternate filenames)
        if not sub_tasks:
            for fname in self._disk_fallback_paths(working_dir):
                path = os.path.join(working_dir, fname)
                if os.path.exists(path):
                    try:
                        with open(path) as f:
                            sub_tasks = json.load(f)
                        if isinstance(sub_tasks, list) and len(sub_tasks) > 0:
                            break
                    except (json.JSONDecodeError, OSError):
                        continue

        # 3. Fallback: rule-based decomposition
        if not sub_tasks:
            sub_tasks = self._fallback_decompose(input_spec)
        return sub_tasks

    # -- JSON extraction ---------------------------------------------------

    @staticmethod
    def _extract_json_array(text: str) -> str | None:
        """Extract the first complete JSON array using bracket counting.

        Unlike greedy ``r\"[[\\s\\S]*]\"`` this correctly handles content that
        may itself contain brackets (LaTeX ``\\begin{...}``, math expressions,
        nested objects) by tracking string state and bracket depth.
        """
        start = text.find('[')
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == '\\':
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None
