from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from agent import run_agent
from tools import TOOL_MAP

CODER_TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file at the given path.",
        "parameters": {"path": "str"},
    },
    {
        "name": "write_file",
        "description": "Write content to a file at the given path. Creates parent directories if needed.",
        "parameters": {"path": "str", "content": "str"},
    },
    {
        "name": "run_command",
        "description": "Run a shell command and return its output. Use for testing code or exploration.",
        "parameters": {"command": "str"},
    },
    {
        "name": "call_claude",
        "description": "Call the Claude Code CLI to handle a complex subtask. Provide a clear prompt describing what you need.",
        "parameters": {"prompt": "str"},
    },
    {
        "name": "web_search",
        "description": "Search the web for information. Returns titles, URLs, and snippets.",
        "parameters": {"query": "str", "max_results": "int (optional, default 10)"},
    },
    {
        "name": "web_fetch",
        "description": "Fetch content from a URL as plain text. Use for reading papers, articles, and documentation from trusted sources like arxiv.org.",
        "parameters": {"url": "str", "max_chars": "int (optional, default 12000, max 20000)"},
    },
]

REVIEWER_TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file at the given path.",
        "parameters": {"path": "str"},
    },
    {
        "name": "run_command",
        "description": "Run a shell command and return its output.",
        "parameters": {"command": "str"},
    },
    {
        "name": "call_claude",
        "description": "Call the Claude Code CLI to analyze code or review logic. Provide a clear prompt describing what to check.",
        "parameters": {"prompt": "str"},
    },
    {
        "name": "web_search",
        "description": "Search the web for information. Returns titles, URLs, and snippets.",
        "parameters": {"query": "str", "max_results": "int (optional, default 10)"},
    },
    {
        "name": "web_fetch",
        "description": "Fetch content from a URL as plain text. Use for reading papers and documentation from trusted sources like arxiv.org.",
        "parameters": {"url": "str", "max_chars": "int (optional, default 12000, max 20000)"},
    },
]


class MultiAgentState(TypedDict):
    messages: list[dict[str, Any]]
    current_agent: str
    requirement: str
    working_dir: str
    review_passed: bool
    iteration: int
    backend: str


def _serialize_messages(messages: list) -> list[dict[str, Any]]:
    result = []
    for m in messages:
        result.append({
            "role": type(m).__name__,
            "content": m.content if hasattr(m, "content") else str(m),
        })
    return result


def _make_coder(backend: str):
    def coder_node(state: MultiAgentState) -> dict:
        task = (
            f"Implement the following requirement by writing code files:\n\n"
            f"{state['requirement']}\n\n"
            f"Write the code, test it, and mark TASK_COMPLETE when done."
        )

        result = run_agent(
            task=task,
            working_dir=state["working_dir"],
            tools=CODER_TOOLS,
            tool_map=TOOL_MAP,
            max_steps=15,
            backend=state.get("backend", backend),
        )

        return {
            "messages": _serialize_messages(result["messages"]),
            "current_agent": "reviewer",
            "review_passed": False,  # reset — new code always needs fresh review
            "iteration": state["iteration"] + 1,
        }
    return coder_node


def _make_reviewer(backend: str):
    def reviewer_node(state: MultiAgentState) -> dict:
        task = (
            f"Review the code written to satisfy this requirement:\n\n"
            f"{state['requirement']}\n\n"
            f"Read the files, check for bugs, test if possible. "
            f"If the code is correct, output REVIEW_PASSED and then TASK_COMPLETE. "
            f"If there are issues, describe them clearly and output REVIEW_FAILED, then TASK_COMPLETE."
        )

        result = run_agent(
            task=task,
            working_dir=state["working_dir"],
            tools=REVIEWER_TOOLS,
            tool_map=TOOL_MAP,
            max_steps=10,
            backend=state.get("backend", backend),
        )

        final_text = ""
        for m in reversed(result["messages"]):
            content = m.content if hasattr(m, "content") else str(m)
            final_text += content

        review_passed = "REVIEW_PASSED" in final_text and "REVIEW_FAILED" not in final_text

        return {
            "messages": _serialize_messages(result["messages"]),
            "review_passed": review_passed,
            "current_agent": "coder" if not review_passed else "reviewer",
            "iteration": state["iteration"] + 1,
        }
    return reviewer_node


def _make_router():
    def router(state: MultiAgentState) -> Literal["coder", "reviewer", "__end__"]:
        if state["review_passed"]:
            return END
        if state["iteration"] >= 5:
            return END
        if state["current_agent"] == "coder":
            return "coder"
        return "reviewer"
    return router


def build_graph(backend: str = "deepseek") -> StateGraph:
    """Build and compile the multi-agent LangGraph.

    Args:
        backend: 'deepseek' (default) or 'claude_cli'.
    """
    workflow = StateGraph(MultiAgentState)

    workflow.add_node("coder", _make_coder(backend))
    workflow.add_node("reviewer", _make_reviewer(backend))

    workflow.set_entry_point("coder")

    router = _make_router()
    workflow.add_conditional_edges(
        "coder",
        router,
        {"reviewer": "reviewer", "coder": "coder", END: END},
    )
    workflow.add_conditional_edges(
        "reviewer",
        router,
        {"reviewer": "reviewer", "coder": "coder", END: END},
    )

    return workflow.compile()
