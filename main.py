import argparse
import json
import sys
from pathlib import Path

from pipeline import (CoderPipeline, ResearchPipeline, CoderPPPipeline,
                       TopologyPipeline, SkillPipeline, FeaturePipeline,
                       SelfEvolutionPipeline)
from tools import ToolRegistry

PIPELINES = {
    "coder": CoderPipeline,
    "research": ResearchPipeline,
    "coderpp": CoderPPPipeline,
    "topology": TopologyPipeline,
    "skill": SkillPipeline,
    "feature": FeaturePipeline,
    "self_evolution": SelfEvolutionPipeline,
}


def _load_tools_config(path: str) -> dict[str, dict[str, list[str]]]:
    """Load a tools configuration JSON file.

    Returns the parsed dict suitable for ToolRegistry.set_tool_config().
    Exits with an error message if the file is missing or invalid.
    """
    try:
        with open(path) as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: tools config file not found: {path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in tools config: {e}")
        sys.exit(1)
    if not isinstance(config, dict):
        print("Error: tools config must be a JSON object mapping pipeline names to role→tool-list dicts.")
        sys.exit(1)
    return config


def main():
    parser = argparse.ArgumentParser(description="Universal Multi-Agent Framework")
    parser.add_argument(
        "requirement", nargs="?",
        help="The task/requirement or research topic for the agents",
    )
    parser.add_argument(
        "--mode", "-m", default="coder", choices=list(PIPELINES.keys()),
        help="Pipeline mode",
    )
    parser.add_argument(
        "--working-dir", "-d", default=None,
        help="Working directory (default: <mode>_output/ inside the repo)",
    )
    parser.add_argument(
        "--backend", "-b", default="deepseek", choices=["deepseek", "claude_cli"],
        help="LLM backend",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Remove and recreate the output directory before running",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Enable checkpoint loading (resume from prior run)",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip the decomposition confirmation prompt",
    )
    parser.add_argument(
        "--tools-config",
        default=str(Path(__file__).resolve().parent / "tools_config.json"),
        help="Path to a JSON file specifying tool overrides per pipeline/role.",
    )
    parser.add_argument(
        "--target", "-t", default=None,
        help="Directory or file to analyze (skill/feature/topology/self_evolution modes). "
             "This is WHAT you want to analyze, distinct from --working-dir "
             "which is WHERE outputs go.",
    )
    args = parser.parse_args()

    requirement = args.requirement
    if not requirement:
        # For skill/topology/feature/self_evolution modes, --target makes the text
        # requirement optional.
        if args.target and args.mode in ("skill", "topology", "feature", "self_evolution"):
            requirement = args.target
        elif sys.stdin.isatty():
            prompt_map = {"research": "Enter research topic", "coderpp": "Enter coding requirement", "coder": "Enter requirement", "feature": "Enter feature description", "self_evolution": "Enter self-evolution goal"}
            requirement = input(f"{prompt_map.get(args.mode, 'Enter requirement')}: ").strip()
        else:
            requirement = sys.stdin.read().strip()

    if not requirement:
        print("Error: no requirement/topic provided.")
        sys.exit(1)

    # Load tools config (defaults to tools_config.json in repo root)
    config = _load_tools_config(args.tools_config)
    ToolRegistry.set_tool_config(config)
    print(f"Tools config: {args.tools_config}")

    pipeline_cls = PIPELINES[args.mode]
    pipeline = pipeline_cls(
        working_dir=args.working_dir,
        backend=args.backend,
        clean=args.clean,
        resume=args.resume,
        yes=args.yes,
    )

    print(f"Mode: {args.mode}")
    print(f"Working directory: {pipeline.working_dir}")
    print(f"Input: {requirement}")
    print(f"Backend: {args.backend}")
    if args.clean:
        print("Clean: yes")
    if args.resume:
        print("Resume: yes")
    print("-" * 50)

    pipeline.run(requirement, target_dir=args.target)


if __name__ == "__main__":
    main()
