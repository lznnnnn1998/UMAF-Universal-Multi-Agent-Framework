import argparse
import os
import sys
from pathlib import Path

from graph import MultiAgentState, build_graph
from research.graph import ResearchState, build_research_graph


def _run_coder_pipeline(requirement: str, working_dir: str, backend: str):
    """Run the original coder + reviewer pipeline."""
    graph = build_graph(backend=backend)

    initial_state: MultiAgentState = {
        "messages": [],
        "current_agent": "coder",
        "requirement": requirement,
        "working_dir": working_dir,
        "review_passed": False,
        "iteration": 0,
        "backend": backend,
    }

    final_state = graph.invoke(initial_state)

    print("-" * 50)
    if final_state["review_passed"]:
        print("SUCCESS: Review passed!")
    else:
        print("FINISHED: Max iterations reached or review not passed.")

    print(f"Iterations: {final_state['iteration']}")
    print(f"Working directory: {working_dir}")


def _run_research_pipeline(topic: str, working_dir: str, backend: str):
    """Run the research pipeline: head → workers → reviewer → writer."""
    graph = build_research_graph(backend=backend)

    initial_state: ResearchState = {
        "topic": topic,
        "working_dir": working_dir,
        "backend": backend,
        "sub_tasks": [],
        "worker_outputs": [],
        "top_3": [],
        "latex_file": "",
        "status": "initialized",
    }

    print("Phase 1/4: Decomposing research topic into sub-tasks...")
    final_state = graph.invoke(initial_state)

    print("-" * 50)
    print(f"Status: {final_state['status']}")
    print(f"Sub-tasks generated: {len(final_state.get('sub_tasks', []))}")
    print(f"Worker outputs: {len(final_state.get('worker_outputs', []))}")
    print(f"Top 3 proposals: {len(final_state.get('top_3', []))}")

    if final_state.get("top_3"):
        print("\n--- TOP 3 RESEARCH PROPOSALS ---")
        for i, item in enumerate(final_state["top_3"]):
            print(f"  {i+1}. [{item.get('total_score', '?')}/50] {item.get('title', 'Untitled')}")

    latex_file = final_state.get("latex_file", "")
    if latex_file and os.path.exists(latex_file):
        print(f"\nLaTeX proposal saved to: {latex_file}")
    else:
        print("\nWarning: LaTeX file was not generated.")

    print(f"\nAll outputs in: {working_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Universal Multi-Agent Framework"
    )
    parser.add_argument(
        "requirement",
        nargs="?",
        help="The task/requirement or research topic for the agents",
    )
    parser.add_argument(
        "--mode",
        "-m",
        default="coder",
        choices=["coder", "research"],
        help="Pipeline mode: 'coder' (code gen) or 'research' (multi-agent research with LaTeX output)",
    )
    parser.add_argument(
        "--working-dir",
        "-d",
        default=None,
        help="Working directory for file operations (default: research_output/ inside the repo)",
    )
    parser.add_argument(
        "--backend",
        "-b",
        default="deepseek",
        choices=["deepseek", "claude_cli"],
        help="LLM backend: 'deepseek' (API) or 'claude_cli' (local claude CLI)",
    )
    args = parser.parse_args()

    requirement = args.requirement
    if not requirement:
        if sys.stdin.isatty():
            prompt = "Enter research topic" if args.mode == "research" else "Enter requirement"
            requirement = input(f"{prompt}: ").strip()
        else:
            requirement = sys.stdin.read().strip()

    if not requirement:
        print("Error: no requirement/topic provided.")
        sys.exit(1)

    working_dir = args.working_dir or str(Path(__file__).resolve().parent / "research_output")
    Path(working_dir).mkdir(parents=True, exist_ok=True)

    print(f"Mode: {args.mode}")
    print(f"Working directory: {working_dir}")
    print(f"Input: {requirement}")
    print(f"Backend: {args.backend}")
    print("-" * 50)

    if args.mode == "research":
        _run_research_pipeline(requirement, working_dir, args.backend)
    else:
        _run_coder_pipeline(requirement, working_dir, args.backend)


if __name__ == "__main__":
    main()
