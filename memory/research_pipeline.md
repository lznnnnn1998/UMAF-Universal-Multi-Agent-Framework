---
name: research-pipeline
description: Phase 2 research pipeline — head/workers/reviewer/writer topology with circuit breakers and verified results
metadata: 
  node_type: memory
  type: project
  originSessionId: d4200744-181c-4ba7-9d5b-36d64631acd6
---

Phase 2 implemented a `research/` package with a 4-stage multi-agent pipeline:

## Pipeline Flow

```
Head Agent (decompose) → Workers (parallel research) → Reviewer (score & rank) → Writer (LaTeX) → END
```

1. **Head Agent** ([head_agent.py](research/head_agent.py)) — Decomposes a broad research topic into 5+ specific sub-topics. Uses `run_agent` with DECOMPOSE_TOOLS. Fallback `_fallback_decompose()` extracts keywords from topic (splits on commas/`and`/`vs`/`;`), generates keyword-specific sub-tasks, pads to ≥5 with generic fillers. ThreadPoolExecutor with 120s timeout.

2. **Worker Agents** ([worker_agent.py](research/worker_agent.py)) — Backend-aware (v1.2): `claude_cli` workers use `_build_worker_task_claude_cli()` — WebSearch + Write + Read directly, no nested `claude -p`. `deepseek` workers use `_build_worker_task_deepseek()` with `call_claude` for reasoning. Task instructs agents to research, write `research_NN_Title.md`, verify with Read, signal TASK_COMPLETE. Returns `{sub_task_id, title, output_file, summary}`.

3. **Reviewer Agent** ([reviewer_agent.py](research/reviewer_agent.py)) — Reads all worker outputs, scores on 5 dimensions (1-10 each, max 50): depth, accuracy, relevance, clarity, originality. Writes `scoring_report.json`. `_extract_scores()` reads JSON file first, then falls back to message parsing with non-greedy regex.

4. **Writer** ([writer.py](research/writer.py)) — LLM generates complete LaTeX with preamble, sections, abstract, scoring table, bibliography. Falls back to Python template `_fallback_latex()` with `_latex_escape()` covering all 10 LaTeX special characters.

## Circuit Breakers ([research/graph.py](research/graph.py))

- Head: 120s timeout via ThreadPoolExecutor, fallback on timeout
- Workers: parallel via ThreadPoolExecutor (max 2 concurrent, v1.2), 300s timeout per worker
- File verification: checks `os.path.exists()` and `os.path.getsize() > 0` after each worker
- Deduplication: MD5 fingerprint of summaries, applied post-hoc after all workers complete
- Status flow always moves forward; `researched_partial` is accepted by reviewer
- Reviewer auto-ranks (25/50) if LLM scoring fails

## Verified Results (May 2026)

v1.0 test topic: "model quantization: QAT, PTQ, stochastic rounding"
- Sub-tasks generated: 5 (LLM-based, specific titles)
- Worker completion: 5/5 (100%)
- Top scores: 46/50 (comparative analysis), 44/50 (QAT), 44/50 (PTQ)

v1.2 test topic: "Flash Attention, Multi-Query Attention, Grouped Query Attention, Paged Attention, Ring Attention"
- Sub-tasks generated: 6-7 (LLM-based)
- Worker output files: 4/6 (67%, 21-26KB each, real technical depth)
- Top score: 43/50 (Ring Attention, Flash Attention)
- 2 workers hit 300s timeout (produced files but TASK_COMPLETE not detected)
- LaTeX: 41KB with equations, tables, citations
- Pipeline time: ~35 minutes with 2 concurrent workers

**Why:** User wanted to test the framework with real research tasks and generate LaTeX proposals.
**How to apply:** `python3 main.py -m research -b claude_cli --working-dir research_output "your topic"`. Related: [[project-overview]], [[backend-architecture]], [[circuit-breakers]], [[v1.2-changes]].
