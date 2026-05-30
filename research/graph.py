import concurrent.futures
import hashlib
import os
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from research.head_agent import decompose_topic, _fallback_decompose
from research.worker_agent import research_subtask
from research.reviewer_agent import review_and_score
from research.writer import write_proposal

# --- Circuit breaker constants ---
WORKER_TIMEOUT_SECONDS = 300   # max wall-clock time per worker
HEAD_TIMEOUT_SECONDS = 120     # max time for decomposition
MAX_DUPLICATE_WORKERS = 2      # if this many workers produce identical summaries, skip remaining similar ones


class ResearchState(TypedDict):
    topic: str
    working_dir: str
    backend: str
    sub_tasks: list[dict[str, Any]]
    worker_outputs: list[dict[str, Any]]
    top_3: list[dict[str, Any]]
    latex_file: str
    status: str
    worker_stats: dict[str, int]  # {"total": N, "succeeded": M, "failed": F, "duplicates_skipped": D}


def _content_fingerprint(text: str) -> str:
    """Short fingerprint of content for deduplication."""
    # Normalize whitespace and take a hash of first 200 chars
    normalized = " ".join(text.lower().split())[:200]
    return hashlib.md5(normalized.encode()).hexdigest()


def _make_decompose_node():
    def decompose_node(state: ResearchState) -> dict:
        """Head agent: decomposes the research topic into sub-tasks, with timeout."""
        topic = state["topic"]
        working_dir = state["working_dir"]
        backend = state.get("backend", "deepseek")

        sub_tasks: list[dict[str, Any]] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                decompose_topic,
                topic=topic,
                working_dir=working_dir,
                backend=backend,
            )
            try:
                sub_tasks = future.result(timeout=HEAD_TIMEOUT_SECONDS)
            except (concurrent.futures.TimeoutError, Exception):
                sub_tasks = []

        if not sub_tasks or len(sub_tasks) < 2:
            sub_tasks = _fallback_decompose(topic)

        return {
            "sub_tasks": sub_tasks,
            "status": "decomposed",
            "worker_stats": {"total": len(sub_tasks), "succeeded": 0, "failed": 0, "duplicates_skipped": 0},
        }
    return decompose_node


def _make_workers_node():
    def workers_node(state: ResearchState) -> dict:
        """Worker agents: each researches one sub-topic with timeout + dedup detection.

        Workers run in parallel via ThreadPoolExecutor. Each worker has an individual
        timeout; slow workers don't block others from completing.
        """
        sub_tasks = state.get("sub_tasks", [])
        if not sub_tasks:
            return {"status": "error_no_subtasks"}

        backend = state.get("backend", "deepseek")
        working_dir = state["working_dir"]

        outputs: list[dict[str, Any]] = []
        succeeded = 0
        failed = 0

        max_workers = min(len(sub_tasks), 2)  # limit concurrent claude -p subprocesses

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sub = {}
            for sub in sub_tasks:
                future = executor.submit(
                    research_subtask,
                    sub_task=sub,
                    working_dir=working_dir,
                    backend=backend,
                )
                future_to_sub[future] = sub

            for future, sub in future_to_sub.items():
                try:
                    result = future.result(timeout=WORKER_TIMEOUT_SECONDS)
                except concurrent.futures.TimeoutError:
                    outputs.append({
                        "sub_task_id": sub["id"],
                        "title": sub["title"],
                        "output_file": "",
                        "summary": f"Worker timed out after {WORKER_TIMEOUT_SECONDS}s.",
                    })
                    failed += 1
                except Exception as e:
                    outputs.append({
                        "sub_task_id": sub["id"],
                        "title": sub["title"],
                        "output_file": "",
                        "summary": f"Worker exception: {e}",
                    })
                    failed += 1
                else:
                    output_file = result.get("output_file", "")
                    if output_file:
                        full_path = os.path.join(working_dir, output_file)
                        if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
                            succeeded += 1
                        else:
                            failed += 1
                    else:
                        failed += 1
                    outputs.append(result)

        # --- Post-hoc deduplication (applied after all workers complete) ---
        seen_fingerprints: set[str] = set()
        duplicates_skipped = 0
        for out in outputs:
            summary = out.get("summary", "")
            if not summary or "timed out" in summary.lower():
                continue
            fp = _content_fingerprint(summary)
            if fp in seen_fingerprints:
                if duplicates_skipped < MAX_DUPLICATE_WORKERS:
                    duplicates_skipped += 1
                out["summary"] = "Skipped: duplicate output (already covered by earlier worker)."
            else:
                seen_fingerprints.add(fp)

        return {
            "worker_outputs": outputs,
            "status": "researched" if succeeded > 0 else "researched_partial",
            "worker_stats": {
                "total": len(sub_tasks),
                "succeeded": succeeded,
                "failed": failed,
                "duplicates_skipped": duplicates_skipped,
            },
        }
    return workers_node


def _make_reviewer_node():
    def reviewer_node(state: ResearchState) -> dict:
        """Reviewer agent: scores and ranks all worker outputs, even partial ones."""
        worker_outputs = state.get("worker_outputs", [])

        # Filter to only outputs that have files or meaningful summaries
        reviewable = [
            wo for wo in worker_outputs
            if wo.get("output_file") or (
                wo.get("summary") and "timed out" not in wo["summary"].lower()
                and "skipped" not in wo["summary"].lower()
                and "no output" not in wo["summary"].lower()
            )
        ]

        if not reviewable:
            return {"status": "error_no_reviewable_outputs"}

        backend = state.get("backend", "deepseek")
        working_dir = state["working_dir"]
        stats = state.get("worker_stats", {})

        try:
            scored = review_and_score(
                worker_outputs=reviewable,
                topic=state["topic"],
                working_dir=working_dir,
                backend=backend,
            )
        except Exception:
            scored = []

        top_3 = scored[:3] if scored else []

        # If scoring produced fewer than 3 results but we have reviewable outputs,
        # create a best-effort ranking
        if len(top_3) < min(3, len(reviewable)):
            # Sort reviewable outputs by a basic heuristic: has file > has summary
            fallback = sorted(
                reviewable,
                key=lambda w: (
                    1 if w.get("output_file") and os.path.exists(
                        os.path.join(working_dir, w["output_file"])
                    ) else 0,
                    len(w.get("summary", "")),
                ),
                reverse=True,
            )
            for i, item in enumerate(fallback[:3]):
                if i >= len(top_3):
                    top_3.append({
                        "sub_task_id": item.get("sub_task_id", i),
                        "title": item.get("title", "Untitled"),
                        "output_file": item.get("output_file", ""),
                        "scores": {
                            "depth": 5, "accuracy": 5, "relevance": 5,
                            "clarity": 5, "originality": 5,
                            "justification": "Auto-ranked (reviewer was unable to score).",
                        },
                        "total_score": 25,
                        "rank": i + 1,
                    })

        return {
            "top_3": top_3,
            "status": "reviewed",
        }
    return reviewer_node


def _make_writer_node():
    def writer_node(state: ResearchState) -> dict:
        """Writer: generates LaTeX from the top 3 proposals."""
        top_3 = state.get("top_3", [])
        if not top_3:
            return {"status": "error_no_top3"}

        backend = state.get("backend", "deepseek")
        working_dir = state["working_dir"]

        try:
            latex_file = write_proposal(
                top_3=top_3,
                topic=state["topic"],
                working_dir=working_dir,
                backend=backend,
            )
        except Exception:
            latex_file = ""

        return {
            "latex_file": latex_file,
            "status": "written" if latex_file else "error_latex_failed",
        }
    return writer_node


def _router(state: ResearchState) -> Literal["workers", "reviewer", "writer", "__end__"]:
    status = state.get("status", "")

    # Terminal error states — no way forward
    if status in ("error_no_subtasks", "error_no_reviewable", "error_no_top3"):
        return END

    # Normal flow map
    flow = {
        "decomposed": "workers",
        "researched": "reviewer",
        "researched_partial": "reviewer",
        "reviewed": "writer",
        "written": END,
    }
    if status in flow:
        return flow[status]

    return END


def build_research_graph(backend: str = "deepseek") -> StateGraph:
    """Build and compile the research pipeline LangGraph.

    Flow: head (decompose) → workers (research) → reviewer (score) → writer (LaTeX)

    Circuit breakers:
    - Head agent: 120s timeout, fallback decomposition
    - Workers: 300s timeout each, dedup detection (skip after 2 duplicates)
    - Reviewer: always proceeds even with partial worker results
    - Writer: always produces best-effort LaTeX
    """
    workflow = StateGraph(ResearchState)

    workflow.add_node("head", _make_decompose_node())
    workflow.add_node("workers", _make_workers_node())
    workflow.add_node("reviewer", _make_reviewer_node())
    workflow.add_node("writer", _make_writer_node())

    workflow.set_entry_point("head")

    workflow.add_conditional_edges(
        "head",
        _router,
        {"workers": "workers", END: END},
    )
    workflow.add_conditional_edges(
        "workers",
        _router,
        {"reviewer": "reviewer", END: END},
    )
    workflow.add_conditional_edges(
        "reviewer",
        _router,
        {"writer": "writer", END: END},
    )
    workflow.add_conditional_edges(
        "writer",
        _router,
        {END: END},
    )

    return workflow.compile()
