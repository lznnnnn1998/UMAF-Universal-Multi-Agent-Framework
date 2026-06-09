"""Coder Pipeline — coder ↔ reviewer loop (max 5 cycles)."""

from __future__ import annotations

import os
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from agent import AgentRole
from utils import scan_review_verdict, serialize_messages
from tools import ToolRegistry
from .base import BasePipeline


class MultiAgentState(TypedDict):
    messages: list[dict[str, Any]]
    current_agent: str
    requirement: str
    working_dir: str
    review_passed: bool
    iteration: int
    backend: str
    coder_files: list[str]


class CoderRole(AgentRole):
    """Generates code to fulfill a requirement."""

    agent_name = "coder"
    max_steps = 15

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        return ToolRegistry.to_dicts(ToolRegistry.coder_tools())

    def build_task(self, backend: str, requirement: str = "", **context: Any) -> str:
        return (
            f"Implement the following requirement by writing code files:\n\n"
            f"{requirement}\n\n"
            f"## Design Guidelines\n"
            f"- Separate core logic into testable functions with parameters — do NOT hardcode values inside functions.\n"
            f"- main() must accept CLI arguments (sys.argv or argparse) so the script is reusable, not a one-shot demo.\n"
            f"- A user running `python script.py` with no arguments should get sensible defaults AND a help/usage path.\n"
            f"- Write tests that cover both default behavior and custom inputs.\n\n"
            f"Write the code, run the tests, and mark TASK_COMPLETE when done."
        )


class ReviewerRole(AgentRole):
    """Reviews generated code for bugs and correctness."""

    agent_name = "reviewer"
    max_steps = 10

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        return ToolRegistry.to_dicts(ToolRegistry.reviewer_tools())

    def build_task(self, backend: str, requirement: str = "",
                   coder_files: list[str] | None = None, **context: Any) -> str:
        files_section = ""
        if coder_files:
            file_list = "\n".join(f"  - `{f}`" for f in sorted(coder_files)[:50])
            trunc = ""
            if len(coder_files) > 50:
                trunc = f"\n  ... and {len(coder_files) - 50} more files"
            files_section = (
                f"\n## Files Produced by Coder\n"
                f"The following files were written by the coder. "
                f"Read and review each one:\n\n{file_list}{trunc}\n"
            )
        return (
            f"Review the code written for this requirement:\n\n"
            f"{requirement}\n"
            f"{files_section}\n"
            f"Read the files, check for bugs, test if possible. "
            f"If correct, output REVIEW_PASSED then TASK_COMPLETE. "
            f"If issues found, describe them and output REVIEW_FAILED then TASK_COMPLETE."
        )


class CoderPipeline(BasePipeline):
    """Coder generates code, Reviewer reviews it. Max 5 cycles."""

    name = "coder"
    default_output_dir = "coder_output"

    def _decompose(self, input_spec: str) -> list[dict[str, Any]]:
        # Coder pipeline has no decomposition — return requirement as-is
        return [{"id": 1, "title": "Requirement", "description": input_spec}]

    def _display_decomposition(self, sub_tasks: list[dict]):
        print(f"\nRequirement: {sub_tasks[0]['description'][:200]}")

    def _build_initial_state(self, input_spec: str, sub_tasks: list[dict]) -> dict:
        return {
            "messages": [],
            "current_agent": "coder",
            "requirement": input_spec,
            "working_dir": self.working_dir,
            "review_passed": False,
            "iteration": 0,
            "backend": self.backend,
            "coder_files": [],
        }

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(MultiAgentState)
        backend = self.backend
        working_dir = self.working_dir

        coder_role = CoderRole()
        reviewer_role = ReviewerRole()

        def _coder_node(state: MultiAgentState) -> dict:
            result = coder_role.execute(
                working_dir=state["working_dir"],
                backend=state.get("backend", backend),
                requirement=state["requirement"],
            )
            # result is AgentResult — serialize messages
            serialized = serialize_messages(result.messages, key="role")
            # Collect files the coder produced so the reviewer knows what to review
            wd = state["working_dir"]
            coder_files: list[str] = []
            if os.path.isdir(wd):
                for root, dirs, files in os.walk(wd):
                    dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules", ".git")]
                    for f in files:
                        if not f.startswith("."):
                            coder_files.append(os.path.relpath(os.path.join(root, f), wd))
            return {
                "messages": serialized,
                "current_agent": "reviewer",
                "review_passed": False,
                "iteration": state["iteration"] + 1,
                "coder_files": coder_files,
            }

        def _reviewer_node(state: MultiAgentState) -> dict:
            result = reviewer_role.execute(
                working_dir=state["working_dir"],
                backend=state.get("backend", backend),
                requirement=state["requirement"],
                coder_files=state.get("coder_files", []),
            )
            review_passed = scan_review_verdict(result.messages) or False
            serialized = serialize_messages(result.messages, key="role")
            return {
                "messages": serialized,
                "review_passed": review_passed,
                "current_agent": "coder" if not review_passed else "reviewer",
                "iteration": state["iteration"] + 1,
            }

        def _router(state: MultiAgentState) -> Literal["coder", "reviewer", "__end__"]:
            if state["review_passed"]:
                return END
            if state["iteration"] >= 5:
                return END
            return state["current_agent"]

        workflow.add_node("coder", _coder_node)
        workflow.add_node("reviewer", _reviewer_node)
        workflow.set_entry_point("coder")
        workflow.add_conditional_edges("coder", _router, {"reviewer": "reviewer", "coder": "coder", END: END})
        workflow.add_conditional_edges("reviewer", _router, {"reviewer": "reviewer", "coder": "coder", END: END})
        return workflow.compile()

    def _print_results(self, final_state: dict):
        print("-" * 50)
        if final_state["review_passed"]:
            print("SUCCESS: Review passed!")
        else:
            print("FINISHED: Max iterations reached or review not passed.")
        print(f"Iterations: {final_state['iteration']}")
        print(f"Working directory: {self.working_dir}")
