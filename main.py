import argparse
import sys

from pipeline import CoderPipeline, ResearchPipeline, CoderPPPipeline, TopologyPipeline, SkillPipeline

PIPELINES = {
    "coder": CoderPipeline,
    "research": ResearchPipeline,
    "coderpp": CoderPPPipeline,
    "topology": TopologyPipeline,
    "skill": SkillPipeline,
}


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
    args = parser.parse_args()

    requirement = args.requirement
    if not requirement:
        if sys.stdin.isatty():
            prompt_map = {"research": "Enter research topic", "coderpp": "Enter coding requirement", "coder": "Enter requirement"}
            requirement = input(f"{prompt_map.get(args.mode, 'Enter requirement')}: ").strip()
        else:
            requirement = sys.stdin.read().strip()

    if not requirement:
        print("Error: no requirement/topic provided.")
        sys.exit(1)

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

    pipeline.run(requirement)


if __name__ == "__main__":
    main()
