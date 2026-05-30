import json
import os
from pathlib import Path
from typing import Optional

# Look for claude_env_sample.json in project root, fall back to
# CLAUDE_ENV_PATH env var or the default location
_CONFIG_PATH = Path(__file__).parent / "claude_env_sample.json"
_EXAMPLE_PATH = Path(__file__).parent / "claude_env_sample.example.json"


def _load_config() -> dict[str, str]:
    """Load environment variables from the Claude Code config file."""
    path = os.getenv("CLAUDE_ENV_PATH", str(_CONFIG_PATH))
    if not Path(path).exists():
        # Fall back to example template
        if _EXAMPLE_PATH.exists():
            path = str(_EXAMPLE_PATH)
        else:
            return {}

    with open(path) as f:
        config = json.load(f)

    env_vars = config.get("claudeCode.environmentVariables", [])
    return {item["name"]: item["value"] for item in env_vars}


_claude_env = _load_config()


def get_claude_env() -> dict[str, str]:
    """Return the env vars to inject when calling the claude CLI.

    These are merged into the subprocess environment so every agent's
    call to `claude` uses the configured backend (DeepSeek proxy, etc.).
    """
    return dict(_claude_env)


def merge_claude_env(base_env: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Return a full env dict: current process env + claude overrides."""
    merged = dict(os.environ)
    if base_env:
        merged.update(base_env)
    merged.update(_claude_env)
    return merged
