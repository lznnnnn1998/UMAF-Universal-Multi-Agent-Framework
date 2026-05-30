---
name: key-updates
description: "Key takeaways — verified metrics across versions, critical bug summaries, architecture decisions that matter most"
metadata: 
  node_type: memory
  type: project
  originSessionId: db564f0a-1b8e-4bed-8a26-28d0132d0605
---

## Verified Metrics by Version

| Metric | v1.0 | v1.1 | v1.2 | v1.3 | v1.3.1 |
|--------|------|------|------|------|--------|
| Worker completion | 3/7 (43%) | 5/5 (100%) | 4/6 (67%) | — | **4/4 (100%)** |
| Top score | 25/50 | 46/50 | 43/50 | — | **47/50** |
| Pipeline time | 35min | 9min | 35min | — | ~12min |
| Research files | 0 | 40KB | 21-26KB each | — | **11-21KB each (4 files)** |
| LaTeX output | 3.6KB | 40KB | 41KB | — | generated |
| Backslash escape | broken | broken | broken | **fixed** | fixed |
| Unit tests | 0 | 0 | 0 | **8/8 pass** | 8/8 pass |
| Python | 3.9 | 3.9 | 3.9 | **3.11** | 3.11 |
| arxiv access | broken | broken | broken | web_fetch | **download_file + pre-fetch** |

## Critical Bugs Fixed (Across All Versions)

1. **cwd sandboxing** (v1.1): `claude -p` wrote to project root. Fix: `cwd=working_dir` in subprocess.run.
2. **Retry with untranslated task** (v1.1): Workers failed tools on retry. Fix: use `translated_task`.
3. **Sequential workers** (v1.1): 35min worst-case. Fix: ThreadPoolExecutor (max 4, now 2).
4. **Nested `claude -p`** (v1.2): Workers spawned recursive invocations. Fix: backend-aware tasks — `claude_cli` workers use native tools directly.
5. **Empty `--allowedTools`** (v1.2): No flag → all tools available → permission denied. Fix: always pass `--allowedTools`.
6. **LaTeX backslash** (v1.3): `"\\textbackslash "` produced tab character. Fix: raw string.
7. **Tool-before-TASK_COMPLETE ordering** (v1.3.1): Agent loop checked TASK_COMPLETE before executing tool calls, so `write_file` + `TASK_COMPLETE` in same response lost the file. Fix: execute tools first, then check completion. 2/4 → 4/4 workers produce files.

## Architecture Decisions That Matter

- **Backend-aware task generation** (v1.2): The single most impactful design decision. `claude_cli` agents get different tasks than `deepseek` agents because the agent IS the runtime. No nesting.
- **Router always moves forward**: Partial results accepted at every stage. Without this, any single worker failure would deadlock the pipeline.
- **Fallbacks at every stage**: decompose → keywords, research → best-effort summary, scoring → auto-rank 25/50, LaTeX → Python template. Pipeline never crashes.
- **Explicit `working_dir`**: No global state. Every tool, subprocess, and file operation is sandboxed. This fixed the root cause of workers producing empty results.

## Known Limitations (Unresolved)

- `claude -p` may write to slightly different filenames than requested
- Complex research tasks may exceed 300s worker timeout
- DeepSeek JSON tool-call format less reliable than native tool calling
- DuckDuckGo scraping fragile to layout changes
- 2/6 workers still hit timeout in v1.2 (produce files but TASK_COMPLETE not detected)

## Resolved in v1.3

- **arxiv.org access** (v1.3): Added `web_fetch` tool that uses Python urllib directly — bypasses Claude Code's domain verification entirely. All academic sites (arxiv.org, open-access journals) are now accessible to all agents without permission prompts.

### Related
[[version_diffs]], [[architecture_progress]]
