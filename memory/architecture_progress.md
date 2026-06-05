---
name: architecture-progress
description: "Architecture evolution — 6 pipelines, 2 backends, modular packages, OOP class hierarchy, 23 roles, 97 tests"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9c942f95-5fb5-4276-8e53-c16059ca5e31
---

## Architecture Overview

UMAF is a LangChain + LangGraph multi-agent framework with 2 LLM backends and 5 pipelines.

```
main.py → pipeline/      → agent.py → llm.py              (all pipelines)
               │               │          ├── ChatOpenAI (deepseek-chat)
               ▼               ▼          └── ClaudeCLILLM (subprocess)
        BasePipeline    AgentRole ABC
        ├── CoderPipeline       ├── CoderRole
        ├── ResearchPipeline    ├── ReviewerRole
        ├── CoderPPPipeline     ├── ResearchWorkerRole
        ├── TopologyPipeline    ├── ResearchDecomposerRole
        ├── SkillPipeline       ├── ResearchReviewerRole
        └── FeaturePipeline     ├── WriterRole
                                ├── FeatureScannerRole
                                ├── FeaturePlannerRole
                                ├── FeatureCoderRole
                                ├── FeatureReviewerRole
                                ├── FeatureReportWriterRole
                                ├── TopologyAnalyzerRole
                                ├── TopologyDesignerRole
                                ├── TopologyEvaluatorRole
                                ├── TopologyWriterRole
                                ├── SkillScannerRole
                                ├── PythonDetectorRole
                                ├── JSDetectorRole
                                ├── InfraDetectorRole
                                ├── ConfigDocsDetectorRole
                                ├── SkillAggregatorRole
                                └── SkillReportWriterRole
                                (23 roles total)
```

### Directories
```
pipeline/           topology/           research/           coderpp/
├── base.py         ├── analyzer.py     ├── head_agent.py   ├── head_agent.py
├── coder.py        ├── designer.py     ├── worker_agent.py ├── worker_agent.py
├── research.py     ├── evaluator.py    ├── reviewer_agent.py├── reviewer_agent.py
├── coderpp.py      └── writer.py       └── writer.py       └── organizer.py
├── topology.py
├── skill.py        skill/              feature/            tools/
├── feature.py      ├── scanner.py      ├── scanner.py      ├── registry.py
└── __init__.py     ├── detectors.py    ├── planner.py      ├── functions.py
                    ├── aggregator.py   ├── coder.py        └── feature_tools.py
                    └── writer.py       ├── reviewer.py
                                        └── writer.py
```

## Two Backends

**DeepSeek** (default): `ChatOpenAI` with `deepseek-chat`, temp=0.3. JSON tool-call loop → parse → execute → loop.

**Claude CLI**: `ClaudeCLILLM` subprocess `claude -p`. Single invocation (CLI is multi-turn). Tool names translated: Python → native names. Env from `claude_env_sample.json`.

## Six Pipelines

### CoderPipeline
Coder (all tools) → Reviewer (no write_file). Max 5 cycles. Coder resets `review_passed=False` each run.

### ResearchPipeline
```
head (decompose) → workers (dependency-ordered) → reviewer (score) → writer (LaTeX) → END
```
- Head: Backend-aware, dynamic 2-8 sub-topics. 120s timeout.
- Workers: Parallel (max 2), 600s timeout. Stop-on-failure + version-bump retry.
- Reviewer: 5-dimension scoring (depth/accuracy/relevance/clarity/originality, each 1-10).
- Writer: LaTeX generation with `_latex_escape()` for 10 special chars.

### CoderPPPipeline
Multi-file code generation: organizer → workers → reviewer. Reads `.md` and `.tex` spec files.

### TopologyPipeline (v1.5)
```
analyzer → designer → evaluator → writer → END
```
- Analyzer: Assesses 6 complexity factors
- Designer: Proposes 2-4 candidate topologies (4 patterns)
- Evaluator: Scores on 5 dimensions, ranks by total_score
- Writer: Produces `topology_spec.json` + `topology_report.md`

### SkillPipeline (v1.5)
```
scanner → 4 parallel detectors → aggregator → writer → END
```
- Scanner: Scans project directory → `project_scan.json`
- Detectors: Python, JS, Infra, ConfigDocs — parallel domain-specific detection
- Aggregator: Deduplicates and categorizes skills across domains
- Writer: Produces `skills.json` + `skills_report.md`

### FeaturePipeline (v1.6)
```
scanner → planner → coder ↔ reviewer (max 5 cycles) → writer → END
```
- Scanner: Analyzes project directory → `project_context.json`
- Planner: Creates implementation plan with `files_to_create` + `files_to_modify`
- Coder: Implements changes — creates new files, modifies existing files, writes and runs tests
- Reviewer: Validates via REVIEW_PASSED/REVIEW_FAILED token scanning (same pattern as CoderPipeline)
- Writer: Produces `feature_report.md`

## Seven Tools + ToolRegistry (`tools/`)
`read_file`, `write_file`, `run_command` (30s), `call_claude` (120s), `web_search` (DuckDuckGo), `web_fetch` (urllib, 20s), `download_file` (urllib, 30s). Modular package: `registry.py` + `functions.py` + `feature_tools.py`. `ToolRegistry` with 18+ role-specific classmethods — no duplicated definitions.

## Circuit Breakers

**Agent-level** (`agent.py`):
- Force wrap-up at ≤3 steps (forbid all tools except write_file)
- Error spiral: 2 consecutive errors → forced best-effort summary
- Mid-loop write reminder at ~2/3 of max steps
- Post-loop forced summary if all steps exhausted
- Claude CLI retry on timeout/error

**Pipeline-level** (`pipeline.py`):
- Head agent: 120s timeout → fallback decomposition
- Workers: 600s timeout, max 2 concurrent, stop-on-failure blocks downstream
- Version-bump retry: max 3 retries, max 4 versions, context reuse via checkpoints
- MD5 dedup; router always moves forward (`researched_partial` accepted)
- All stages have deterministic fallbacks

## Key Design Decisions
- OOP 5-layer hierarchy: Data types → Infrastructure → Agent core → Concrete roles → Pipeline classes
- `AgentRole` ABC template method + `ToolRegistry` centralization
- Explicit `working_dir` — no global state
- Python >= 3.11: `X | None` syntax
- Fallbacks at every pipeline stage
- Backend-aware task generation (no nested `claude -p`)
- Dependency-aware execution with stop-on-failure

## Evolution

| Version | Date | Focus |
|---------|------|-------|
| v1.0 | May 2026 | Initial: 2 pipelines, 2 backends, 5 tools |
| v1.1 | May 2026 | 12 bug fixes: cwd sandboxing, translation, timeouts, parallel workers |
| v1.2 | May 2026 | Backend-aware agents, scoped permissions, conversation logger |
| v1.3 | May 2026 | Python 3.11, dead code removal, dynamic decomposition, web_fetch |
| v1.3.1 | May 2026 | Tool-before-TASK_COMPLETE fix, download_file, pre-fetch layer |
| v1.4 | Jun 2026 | OOP refactoring, pipeline robustness, CoderPP pipeline |
| v1.4.1 | Jun 2026 | 8 bug fixes: agent loop edge cases, checkpointing, smoke tests |
| v1.5 | Jun 2026 | Topology Optimizer + Skill Summarizer pipelines, 5 pipelines total |
| v1.6 | Jun 2026 | Feature Pipeline + modular package structure (pipeline/, tools/, test/), 6 pipelines, 23 roles, 97 tests |

### Related
[[version_diffs]], [[key_updates]], [[oop_refactoring]]
