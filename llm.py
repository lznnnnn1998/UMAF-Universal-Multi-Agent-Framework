import os
import subprocess
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI

from claude_config import merge_claude_env

load_dotenv()

# --- DeepSeek backend (default) ---

deepseek = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.3,
    max_tokens=4096,
)


# --- Claude CLI backend ---

def _messages_to_prompt(messages: list[BaseMessage]) -> str:
    """Convert LangChain messages to a single prompt string for the CLI."""
    parts = []
    for m in messages:
        role = type(m).__name__.replace("Message", "").upper()
        content = m.content if hasattr(m, "content") else str(m)
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


class ClaudeCLILLM:
    """Drop-in replacement for ChatOpenAI that shells out to `claude` CLI.

    The invoke() method accepts an optional `allowed_tools` keyword argument:
        allowed_tools: list[str] of Claude CLI native tool names to enable
                       (e.g. ["Bash", "Read", "Write", "WebSearch", "WebFetch"]).
    """

    def __init__(self, timeout: int = 300):
        self.timeout = timeout

    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
        prompt = _messages_to_prompt(messages)
        allowed_tools = kwargs.get("allowed_tools", [])
        cwd = kwargs.get("cwd")

        cmd = ["claude", "-p", prompt, "--output-format", "text"]

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


_claude_cli_instance = ClaudeCLILLM()


def get_llm(backend: str = "deepseek"):
    """Return the configured LLM instance.

    Args:
        backend: 'deepseek' (default) or 'claude_cli'.
    """
    if backend == "claude_cli":
        return _claude_cli_instance
    return deepseek
