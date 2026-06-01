from __future__ import annotations

import json
import os
from pathlib import Path

# Look for claude_env_sample.json in project root, fall back to
# CLAUDE_ENV_PATH env var or the default location
_CONFIG_PATH = Path(__file__).parent / "claude_env_sample.json"
_EXAMPLE_PATH = Path(__file__).parent / "claude_env_sample.example.json"


class ClaudeConfig:
    """Manages Claude Code environment configuration with lazy loading.

    Replaces the module-level _claude_env singleton. Loads the config
    file on first access rather than at import time.
    """

    def __init__(self):
        self._env: dict[str, str] | None = None

    def _load(self) -> dict[str, str]:
        path = os.getenv("CLAUDE_ENV_PATH", str(_CONFIG_PATH))
        if not Path(path).exists():
            if _EXAMPLE_PATH.exists():
                path = str(_EXAMPLE_PATH)
            else:
                return {}

        with open(path) as f:
            config = json.load(f)

        env_vars = config.get("claudeCode.environmentVariables", [])
        return {item["name"]: item["value"] for item in env_vars}

    def get_env(self) -> dict[str, str]:
        """Return the env vars (cached after first load)."""
        if self._env is None:
            self._env = self._load()
        return dict(self._env)

    def merge_env(self, base_env: dict[str, str] | None = None) -> dict[str, str]:
        """Return os.environ + base_env + claude overrides."""
        merged = dict(os.environ)
        if base_env:
            merged.update(base_env)
        merged.update(self.get_env())
        return merged


# Singleton instance (backward-compatible with old module-level _claude_env)
_config = ClaudeConfig()


def _load_config() -> dict[str, str]:
    """Load environment variables from the Claude Code config file."""
    return _config._load()


_claude_env = _config.get_env()


def get_claude_env() -> dict[str, str]:
    """Return the env vars to inject when calling the claude CLI.

    These are merged into the subprocess environment so every agent's
    call to `claude` uses the configured backend (DeepSeek proxy, etc.).
    """
    return _config.get_env()


def merge_claude_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return a full env dict: current process env + claude overrides."""
    return _config.merge_env(base_env)
