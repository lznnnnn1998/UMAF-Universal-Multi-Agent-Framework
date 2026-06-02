import json
import os
import subprocess
from abc import ABC, abstractmethod
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI

from claude_config import merge_claude_env

load_dotenv()


# --- LLM Provider ABC ---

class LLMProvider(ABC):
    """Abstract base for LLM backends. Unifies DeepSeek API and Claude CLI."""

    @abstractmethod
    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
        """Send messages to the LLM and return the response."""


# --- DeepSeek backend (default) ---

class DeepSeekProvider(LLMProvider):
    """LLM provider backed by the DeepSeek API via ChatOpenAI."""

    def __init__(
        self,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
        api_key: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ):
        self._llm = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
        return self._llm.invoke(messages, **kwargs)


# Module-level singleton (backward-compatible)
deepseek = DeepSeekProvider()._llm


# --- Claude CLI backend ---

def _messages_to_prompt(messages: list[BaseMessage]) -> str:
    """Convert LangChain messages to a single prompt string for the CLI."""
    parts = []
    for m in messages:
        role = type(m).__name__.replace("Message", "").upper()
        content = m.content if hasattr(m, "content") else str(m)
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


class ClaudeCLILLM(LLMProvider):
    """Drop-in replacement for ChatOpenAI that shells out to `claude` CLI.

    The invoke() method accepts an optional `allowed_tools` keyword argument:
        allowed_tools: list[str] of Claude CLI native tool names to enable
                       (e.g. ["Bash", "Read", "Write", "WebSearch", "WebFetch"]).

    stream_invoke() uses --output-format stream-json and yields parsed JSON
    events as they arrive, enabling incremental checkpointing during long runs.
    """

    def __init__(self, timeout: int = 600):
        self.timeout = timeout

    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
        prompt = _messages_to_prompt(messages)
        allowed_tools = kwargs.get("allowed_tools", [])
        cwd = kwargs.get("cwd")

        cmd = ["claude", "-p", prompt, "--output-format", "text",
              "--permission-mode", "bypassPermissions"]

        if allowed_tools:
            cmd.extend(["--allowedTools", ",".join(allowed_tools)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=merge_claude_env(),
                cwd=cwd,
            )
            response_text = result.stdout.strip() or "(no response)"
            if result.stderr:
                stderr = result.stderr.strip()
                if stderr:
                    response_text += f"\n[stderr]: {stderr}"
        except subprocess.TimeoutExpired:
            response_text = "Error: claude CLI timed out"
        except FileNotFoundError:
            response_text = "Error: claude CLI not found"
        except Exception as e:
            response_text = f"Error: {e}"

        return AIMessage(content=response_text)

    def stream_invoke(self, messages: list[BaseMessage], **kwargs: Any):
        """Run claude -p with --output-format stream-json, yielding events.

        Each yielded item is a dict parsed from one JSON line of stdout.
        The caller is responsible for timeout enforcement and cleanup.

        Yields:
            dict: parsed stream-json event (type, message, etc.)
        """
        prompt = _messages_to_prompt(messages)
        allowed_tools = kwargs.get("allowed_tools", [])
        cwd = kwargs.get("cwd")

        cmd = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose",
              "--permission-mode", "bypassPermissions"]
        if allowed_tools:
            cmd.extend(["--allowedTools", ",".join(allowed_tools)])

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=merge_claude_env(),
            cwd=cwd,
        )

        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        finally:
            # Ensure the process is cleaned up
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                proc.kill()
                proc.wait()


_claude_cli_instance = ClaudeCLILLM()


def get_llm(backend: str = "deepseek"):
    """Return the configured LLM instance (backward-compatible).

    Args:
        backend: 'deepseek' (default) or 'claude_cli'.
    """
    if backend == "claude_cli":
        return _claude_cli_instance
    return deepseek


def get_llm_provider(backend: str = "deepseek") -> LLMProvider:
    """Return an LLMProvider instance for the given backend.

    Args:
        backend: 'deepseek' (default) or 'claude_cli'.
    """
    if backend == "claude_cli":
        return ClaudeCLILLM()
    return DeepSeekProvider()
