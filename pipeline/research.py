"""Research Pipeline — head → workers → reviewer → writer."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agent import CheckpointManager
from .base import BasePipeline
from research.head_agent import decompose_topic, _fallback_decompose
from research.worker_agent import research_subtask
from research.reviewer_agent import review_and_score
from research.writer import write_proposal


HEAD_TIMEOUT = 300
WORKER_TIMEOUT = 900
RESEARCH_MAX_VERSIONS = 6
RESEARCH_MAX_WORKER_RETRIES = 5


class ResearchState(TypedDict):
    topic: str
    working_dir: str
    backend: str
    sub_tasks: list[dict[str, Any]]
    worker_outputs: list[dict[str, Any]]
    scored_works: list[dict[str, Any]]
    latex_file: str
    status: str
    worker_stats: dict[str, int]
    version: int


class ResearchPipeline(BasePipeline):
    """Head decomposes topic → Workers research in parallel → Reviewer scores → Writer generates LaTeX."""

    name = "research"
    default_output_dir = "research_output"

    def _decompose(self, input_spec: str) -> list[dict[str, Any]]:
        return decompose_topic(input_spec, self.working_dir, self.backend)

    def _display_decomposition(self, sub_tasks: list[dict]):
        print(f"\nResearch sub-topics ({len(sub_tasks)}):")
        print("-" * 50)
        for t in sub_tasks:
            print(f"  [{t['id']}] {t['title']}")
            print(f"      {t['description'][:120]}")

    def _build_initial_state(self, input_spec: str, sub_tasks: list[dict]) -> dict:
        return {
            "topic": input_spec,
            "working_dir": self.working_dir,
            "backend": self.backend,
            "sub_tasks": sub_tasks,
            "worker_outputs": [],
            "scored_works": [],
            "latex_file": "",
            "status": "decomposed",  # skip head node
            "worker_stats": {"total": len(sub_tasks), "succeeded": 0, "failed": 0, "duplicates_skipped": 0},
            "version": 1,
        }

    def _try_load_resume_state(self, input_spec: str) -> dict | None:
        """Reconstruct Research state from decomposition.json and checkpoints."""
        wd = self.working_dir
        decomp_path = os.path.join(wd, "decomposition.json")
        if not os.path.exists(decomp_path):
            return None

        try:
            with open(decomp_path) as f:
                sub_tasks = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        if not sub_tasks or not isinstance(sub_tasks, list):
            return None

        # Validate dependency graph on resume — warn about cycles early
        dep_issues = BasePipeline._validate_dependencies(sub_tasks)
        if dep_issues:
            print(f"\n[resume] WARNING: Dependency issues in decomposition.json ({len(dep_issues)}):")
            for issue in dep_issues:
                print(f"  - {issue}")
            print("  The pipeline will attempt to break cycles at runtime, but results may be suboptimal.")
            print("  Consider fixing decomposition.json and re-running.")

        log_dir = os.path.join(wd, "agent_log")

        # Scan worker checkpoints: determine max version AND actual success per worker
        max_version = 1
        worker_success: dict[str, bool] = {}  # worker_id → actually succeeded
        if os.path.isdir(log_dir):
            for fname in os.listdir(log_dir):
                m = re.match(r"worker_(\d+)_v(\d+)_checkpoint\.json", fname)
                if m:
                    wid = int(m.group(1))
                    ver = int(m.group(2))
                    if ver > max_version:
                        max_version = ver
                    # Check if this checkpoint declares success
                    try:
                        ck = json.load(open(os.path.join(log_dir, fname)))
                        if ck.get("success") or ck.get("has_written_output"):
                            worker_success[str(wid)] = True
                    except (json.JSONDecodeError, OSError):
                        pass

        # Build worker_outputs from .md files on disk, gated by checkpoint success
        worker_outputs: list[dict] = []
        for st in sub_tasks:
            sid = st.get("id", 0)
            title = st.get("title", "")
            sid_str = str(sid)
            has_checkpoint = any(
                fname.startswith(f"worker_{sid:02d}_v") and
                fname.endswith("_checkpoint.json")
                for fname in (os.listdir(log_dir) if os.path.isdir(log_dir) else [])
            )
            ok = worker_success.get(sid_str, False)
            output_file = ""
            # Worker succeeded if: checkpoint confirms it, OR no checkpoint but file exists
            if ok or not has_checkpoint:
                for fname in sorted(os.listdir(wd)) if os.path.isdir(wd) else []:
                    if fname.startswith(f"research_{sid:02d}_") and fname.endswith(".md"):
                        output_file = fname
                        break
                if not output_file:
                    prefix = f"research_{sid:02d}"
                    for fname in sorted(os.listdir(wd)) if os.path.isdir(wd) else []:
                        if fname.startswith(prefix) and fname.endswith(".md"):
                            output_file = fname
                            break

            worker_outputs.append({
                "sub_task_id": sid,
                "title": title,
                "output_file": output_file,
                "summary": "Resumed from disk" if output_file else "Pending",
            })

        # Scan for scoring report and latex
        scored_works: list[dict] = []
        scoring_path = os.path.join(wd, "scoring_report.json")
        if os.path.exists(scoring_path):
            try:
                with open(scoring_path) as f:
                    scored_works = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        latex_file = ""
        latex_path = os.path.join(wd, "research_proposal.tex")
        if os.path.exists(latex_path):
            latex_file = "research_proposal.tex"

        # Determine status
        succeeded = sum(1 for wo in worker_outputs if wo.get("output_file"))
        total = len(sub_tasks)

        if latex_file:
            status = "written"  # pipeline completed
        elif scored_works:
            status = "reviewed"
        elif succeeded == total:
            status = "researched"
        elif succeeded > 0:
            status = "worker_retry"
        else:
            status = "decomposed"

        return {
            "topic": input_spec,
            "working_dir": wd,
            "backend": self.backend,
            "sub_tasks": sub_tasks,
            "worker_outputs": worker_outputs,
            "scored_works": scored_works,
            "latex_file": latex_file,
            "status": status,
            "version": max(max_version, 1),
            "worker_stats": {
                "total": total,
                "succeeded": succeeded,
                "failed": total - succeeded,
                "duplicates_skipped": 0,
                "retries": max(max_version - 1, 0),
            },
        }

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(ResearchState)
        backend = self.backend
        working_dir = self.working_dir

        def _head_node(state: ResearchState) -> dict:
            # Skip if already decomposed (from double-check or resume)
            if state.get("sub_tasks"):
                # Preserve incoming status (e.g. "worker_retry" on resume)
                return {"status": state.get("status") or "decomposed",
                        "worker_stats": state.get("worker_stats", {})}

            topic = state["topic"]
            sub_tasks: list[dict] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(decompose_topic, topic=topic, working_dir=working_dir, backend=backend)
                try:
                    sub_tasks = future.result(timeout=HEAD_TIMEOUT)
                except (concurrent.futures.TimeoutError, Exception):
                    sub_tasks = []
            if not sub_tasks or len(sub_tasks) < 2:
                sub_tasks = _fallback_decompose(topic)
            return {
                "sub_tasks": sub_tasks,
                "status": "decomposed",
                "worker_stats": {"total": len(sub_tasks), "succeeded": 0, "failed": 0, "duplicates_skipped": 0},
            }

        def _workers_node(state: ResearchState) -> dict:
            sub_tasks = state.get("sub_tasks", [])
            if not sub_tasks:
                return {"status": "error_no_subtasks"}

            version = state.get("version", 1)
            current_status = state.get("status", "")
            prev_outputs = state.get("worker_outputs", [])
            worker_retry_count = state.get("worker_stats", {}).get("retries", 0)

            # Map existing outputs by sub_task_id for retry preservation
            id_to_output: dict[int, dict] = {
                wo["sub_task_id"]: wo for wo in prev_outputs if wo.get("output_file")
            }

            # ── Determine which tasks to run ──
            if current_status == "worker_retry":
                # Retry only failed workers (no output file)
                failed = [
                    st for st in sub_tasks
                    if st["id"] not in id_to_output
                ]
                if not failed:
                    return {
                        "worker_outputs": list(id_to_output.values()),
                        "status": "researched",
                        "version": version,
                        "worker_stats": {"total": len(sub_tasks),
                                         "succeeded": len(id_to_output),
                                         "failed": 0, "duplicates_skipped": 0,
                                         "retries": version - 1},
                    }

                if worker_retry_count >= RESEARCH_MAX_WORKER_RETRIES:
                    print(f"\n[research worker retry] Max retries ({RESEARCH_MAX_WORKER_RETRIES}) reached. "
                          f"{len(id_to_output)}/{len(sub_tasks)} workers have output.")

                    # Try one last pass with all remaining workers — downstream
                    # workers that were deferred by an early break may still
                    # produce useful output via independent research.
                    still_pending = [
                        st for st in sub_tasks
                        if st["id"] not in id_to_output
                    ]
                    if still_pending:
                        print(f"  Final attempt for {len(still_pending)} remaining worker(s): "
                              f"{[t['id'] for t in still_pending]}")
                        final_version = version + 1
                        outputs, succeeded, final_failed = BasePipeline._run_workers_with_deps(
                            still_pending,
                            lambda item, wd, be: research_subtask(item, wd, be, version=final_version),
                            working_dir, backend, WORKER_TIMEOUT,
                        )
                        for out in outputs:
                            if out.get("output_file"):
                                # Verify file actually exists on disk
                                fp = os.path.join(working_dir, out["output_file"])
                                if os.path.exists(fp) and os.path.getsize(fp) > 0:
                                    id_to_output[out["sub_task_id"]] = out

                    # Placeholders only for workers that still have no output
                    for st in sub_tasks:
                        if st["id"] not in id_to_output:
                            id_to_output[st["id"]] = {
                                "sub_task_id": st["id"],
                                "title": st["title"],
                                "output_file": "",
                                "summary": "Worker did not produce output after all retries.",
                            }
                    return {
                        "worker_outputs": list(id_to_output.values()),
                        "status": "researched_partial",
                        "version": version,
                        "worker_stats": {"total": len(sub_tasks),
                                         "succeeded": sum(1 for wo in id_to_output.values() if wo.get("output_file")),
                                         "failed": sum(1 for wo in id_to_output.values() if not wo.get("output_file")),
                                         "duplicates_skipped": 0,
                                         "retries": version - 1},
                    }

                print(f"\n[research version {version}] Retrying {len(failed)} failed worker(s): "
                      f"{[t['id'] for t in failed]}")
                new_version = version + 1
                tasks_to_run = failed
                # Preserve successful outputs
                kept_outputs = list(id_to_output.values())
                worker_retry_count += 1
            else:
                # Fresh start: run all workers
                new_version = version
                tasks_to_run = list(sub_tasks)
                kept_outputs = []
                worker_retry_count = 0

            # ── Execute tasks with dependency ordering ──
            outputs, succeeded, failed = BasePipeline._run_workers_with_deps(
                tasks_to_run,
                lambda item, wd, be: research_subtask(item, wd, be, version=new_version),
                working_dir, backend, WORKER_TIMEOUT,
            )

            # Verify output files exist
            for out in outputs:
                of = out.get("output_file", "")
                if of:
                    fp = os.path.join(working_dir, of)
                    if not os.path.exists(fp) or os.path.getsize(fp) == 0:
                        out["output_file"] = ""

            # Merge with preserved outputs
            all_outputs = kept_outputs + outputs

            # Dedup
            def _fp(text: str) -> str:
                return hashlib.md5(" ".join(text.lower().split())[:200].encode()).hexdigest()

            seen: set[str] = set()
            dupes = 0
            for out in all_outputs:
                s = out.get("summary", "")
                if not s:
                    continue
                f = _fp(s)
                if f in seen:
                    out["summary"] = "Skipped: duplicate output."
                    dupes += 1
                else:
                    seen.add(f)

            total_succeeded = sum(1 for wo in all_outputs if wo.get("output_file"))
            total_failed = len(sub_tasks) - total_succeeded

            # Retry if any workers failed and we haven't exceeded limits
            if total_failed > 0 and new_version < RESEARCH_MAX_VERSIONS and worker_retry_count < RESEARCH_MAX_WORKER_RETRIES:
                print(f"\n[research version {new_version}] {total_failed}/{len(sub_tasks)} workers failed — "
                      f"will retry failed workers.")
                return {
                    "worker_outputs": all_outputs,
                    "status": "worker_retry",
                    "version": new_version,
                    "worker_stats": {"total": len(sub_tasks), "succeeded": total_succeeded,
                                     "failed": total_failed, "duplicates_skipped": dupes,
                                     "retries": worker_retry_count},
                }

            return {
                "worker_outputs": all_outputs,
                "status": "researched" if total_succeeded > 0 else "researched_partial",
                "version": new_version,
                "worker_stats": {"total": len(sub_tasks), "succeeded": total_succeeded,
                                 "failed": total_failed, "duplicates_skipped": dupes,
                                 "retries": worker_retry_count},
            }

        def _reviewer_node(state: ResearchState) -> dict:
            outputs = state.get("worker_outputs", [])
            version = state.get("version", 1)
            reviewable = [wo for wo in outputs if wo.get("output_file") or (
                wo.get("summary") and "timed out" not in wo["summary"].lower()
                and "skipped" not in wo["summary"].lower()
            )]
            if not reviewable:
                return {"status": "error_no_reviewable"}

            try:
                scored = review_and_score(reviewable, state["topic"], working_dir, backend, version=version)
            except Exception:
                scored = []

            if not scored:
                # Fallback: rank by file existence and summary length
                scored = sorted(reviewable, key=lambda w: (
                    1 if w.get("output_file") and os.path.exists(os.path.join(working_dir, w["output_file"])) else 0,
                    len(w.get("summary", "")),
                ), reverse=True)
                for i, item in enumerate(scored):
                    item.setdefault("scores", {
                        "depth": 5, "accuracy": 5, "relevance": 5, "clarity": 5, "originality": 5,
                        "justification": "Auto-ranked (reviewer was unable to score).",
                    })
                    item["total_score"] = 25
                    item["rank"] = i + 1

            return {"scored_works": scored, "status": "reviewed"}

        def _writer_node(state: ResearchState) -> dict:
            scored = state.get("scored_works", [])
            version = state.get("version", 1)
            if not scored:
                return {"status": "error_no_scored_works"}
            try:
                latex_file = write_proposal(scored, state["topic"], working_dir, backend, version=version)
            except Exception:
                latex_file = ""

            # Merge all agent checkpoints on completion
            sub_tasks = state.get("sub_tasks", [])
            for st in sub_tasks:
                ckm = CheckpointManager(working_dir, f"worker_{st['id']:02d}")
                ckm.merge()
            # Also merge head and reviewer
            for name in ("head_decompose", "reviewer"):
                ckm = CheckpointManager(working_dir, name)
                ckm.merge()

            return {"latex_file": latex_file, "status": "written" if latex_file else "error_latex_failed"}

        flow = {"decomposed": "workers", "worker_retry": "workers", "researched": "reviewer", "researched_partial": "reviewer", "reviewed": "writer", "written": END}
        terminal = {"error_no_subtasks", "error_no_reviewable", "error_no_scored_works"}
        router = BasePipeline._status_router(flow, terminal)

        workflow.add_node("head", _head_node)
        workflow.add_node("workers", _workers_node)
        workflow.add_node("reviewer", _reviewer_node)
        workflow.add_node("writer", _writer_node)
        workflow.set_entry_point("head")
        for node in ("head", "workers", "reviewer", "writer"):
            workflow.add_conditional_edges(node, router, {"workers": "workers", "reviewer": "reviewer", "writer": "writer", END: END})

        return workflow.compile()

    def _print_results(self, final_state: dict):
        print("-" * 50)
        print(f"Status: {final_state['status']}")
        print(f"Sub-tasks: {len(final_state.get('sub_tasks', []))}")
        print(f"Worker outputs: {len(final_state.get('worker_outputs', []))}")
        scored = final_state.get("scored_works", [])
        print(f"Scored works: {len(scored)}")
        if scored:
            print("\n--- SCORED RESEARCH PROPOSALS ---")
            for i, item in enumerate(scored):
                print(f"  {i+1}. [{item.get('total_score', '?')}/50] {item.get('title', 'Untitled')}")
        latex = final_state.get("latex_file", "")
        if latex and os.path.exists(latex):
            print(f"\nLaTeX: {latex}")
        else:
            print("\nWarning: LaTeX file not generated.")
        print(f"\nOutputs in: {self.working_dir}")
