"""Shared utility functions used across the framework."""

from __future__ import annotations


def extract_json_object(text: str) -> str | None:
    """Extract the outermost JSON object from mixed text."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def extract_json_array(text: str) -> str | None:
    """Extract the first complete JSON array from text using bracket counting."""
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
