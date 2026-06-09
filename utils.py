"""Shared utility functions used across the framework."""

from __future__ import annotations


def find_matching_delimiter(text: str, start: int,
                            open_char: str, close_char: str) -> str | None:
    """Return the substring from *start* to the matching closing delimiter.

    Tracks string state and escape sequences so delimiters inside JSON strings
    are not counted.  Returns ``None`` if the closing delimiter is never found.
    """
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
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def extract_json_object(text: str) -> str | None:
    """Extract the outermost JSON object from mixed text."""
    start = text.find("{")
    if start == -1:
        return None
    return find_matching_delimiter(text, start, "{", "}")


def extract_json_array(text: str) -> str | None:
    """Extract the first complete JSON array from text using bracket counting."""
    start = text.find("[")
    if start == -1:
        return None
    return find_matching_delimiter(text, start, "[", "]")


_PROFICIENCY_SCORES: dict[str, int] = {
    "expert": 4,
    "advanced": 3,
    "intermediate": 2,
    "beginner": 1,
}


def safe_read(path: str, default: str = "") -> str:
    """Read a file, returning *default* on any error."""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return default


def scan_review_verdict(messages: list) -> bool | None:
    """Scan agent messages for REVIEW_PASSED/REVIEW_FAILED tokens.

    Returns ``True`` if the last verdict is REVIEW_PASSED, ``False`` if
    REVIEW_FAILED, or ``None`` if no verdict token is found.
    """
    for msg in reversed(messages):
        if type(msg).__name__ != "AIMessage":
            continue
        content = msg.content if hasattr(msg, "content") else str(msg)
        if "REVIEW_PASSED" in content and "REVIEW_FAILED" not in content:
            return True
        if "REVIEW_FAILED" in content:
            return False
    return None


def serialize_messages(messages: list, key: str = "type") -> list[dict[str, str]]:
    """Serialize LangChain messages to plain dicts.

    *key* controls the role field name: ``"type"`` for checkpoints,
    ``"role"`` for LangGraph state.
    """
    out: list[dict[str, str]] = []
    for m in messages:
        content = m.content if hasattr(m, "content") else str(m)
        out.append({key: type(m).__name__, "content": content})
    return out
