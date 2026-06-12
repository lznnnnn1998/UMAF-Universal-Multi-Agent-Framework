---
name: architecture-progress
description: "Architecture evolution — 8 pipelines, 2 backends, modular packages, OOP class hierarchy, 39 roles, 480 tests"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9c942f95-5fb5-4276-8e53-c16059ca5e31
---

## Architecture Overview

UMAF is a LangChain + LangGraph multi-agent framework with 2 LLM backends and 8 pipelines.

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
        ├── FeaturePipeline     ├── WriterRole
        ├── PlanPipeline        ├── SelfEvolutionAnalyzerRole
        └── SelfEvolutionPipeline ├── SelfEvolutionPlannerRole
                                 ├── SelfEvolutionCoderRole
                                 ├── SelfEvolutionReviewerRole
                                 ├── SelfEvolutionWriterRole
                                 ├── FeatureScannerRole
                                 ├── FeaturePlannerRole
                                 ├── FeatureCoderRole
                                 ├── FeatureReviewerRole
                                 ├── FeatureReportWriterRole
                                 ├── PlanScannerRole
                                 ├── PlanDecomposerRole
                                 ├── PlanDependencyAnalyzerRole
                                 ├── PlanRiskAssessorRole
                                 ├── PlanResourceEstimatorRole
                                 ├── PlanCrossCuttingAnalyzerRole
                                 ├── PlanWriterRole
                                 ├── TopologyAnalyzerRole
                                 ├── TopologyDesignerRole
                                 ├── TopologyEvaluatorRole
                                 ├── TopologyWriterRole
                                 ├── SkillScannerRole
                                 ├── DomainExpertiseDetectorRole
                                 ├── TechnicalCraftDetectorRole
                                 ├── MethodologyDetectorRole
                                 ├── RigorDetectorRole
                                 ├── SkillAggregatorRole
                                 └── SkillReportWriterRole
                                 (39 roles total)
```

### Directories
```
pipeline/           topology/           research/           coderpp/
├── base.py         ├── analyzer.py     ├── head_agent.py   ├── head_agent.py
├── coder.py        ├── designer.py     ├── worker_agent.py ├── worker_agent.py
├── research.py     ├── evaluator.py    ├── reviewer_agent.py├── reviewer_agent.py
├── coderpp.py      └── writer.py       └── writer.py       └── organizer.py
├── topology.py
├── skill.py        skill/              feature/            plan/
├── feature.py      ├── scanner.py      ├── scanner.py      ├── scanner.py
├── self_evolution.py ├── detectors.py  ├── planner.py      ├── decomposer.py
├── plan.py         ├── aggregator.py   ├── coder.py        ├── dependency.py
└── __init__.py     └── writer.py       ├── reviewer.py     ├── risk.py
                                         └── writer.py       ├── resource.py
                    self_evolution/                          ├── cross_cutting.py
                    ├── analyzer.py                          └── writer.py
                    ├── planner.py
                    ├── coder.py
                    ├── reviewer.py
                    └── writer.py
```

## Two Backends

**DeepSeek** (default): `ChatOpenAI` with `deepseek-chat`, temp=0.3. JSON tool-call loop → parse → execute → loop.

**Claude CLI**: `ClaudeCLILLM` subprocess `claude -p`. Single invocation (CLI is multi-turn). Tool names translated: Python → native names. Env from `claude_env_sample.json`.

## Eight Pipelines

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

### TopologyPipeline (v1.5, v1.9 retry loop)
```
analyzer → designer → evaluator → writer → END
                ↑          │
                └──────────┘ (retry: max 3, score < 35/50)
```
- Analyzer: Assesses 6 complexity factors
- Designer: Proposes 2-4 candidate topologies (4 patterns), accepts `evaluation_feedback` on retries
- Evaluator: Scores on 5 dimensions, ranks by total_score, routes back to designer if best < 35/50
- Writer: Produces `topology_spec.json` + `topology_report.md`

### SkillPipeline (v1.5, v2 detectors)
```
scanner → 4 parallel detectors → aggregator → writer → END
```
- Scanner: Classifies artifact type, deep-reads content → `artifact_analysis.json`
- Detectors: DomainExpertise, TechnicalCraft, Methodology, Rigor — artifact-agnostic, evidence-based
- Aggregator: Deduplicates and categorizes skills across domains
- Writer: Produces `skills.json` + `skills_report.md`

### FeaturePipeline (v1.6, v1.9 version-aware)
```
scanner → planner → coder ↔ reviewer (version-aware, max 5 versions) → writer → END
```
- Scanner: Analyzes project directory → `project_context.json`
- Planner: Creates implementation plan with `files_to_create` + `files_to_modify`
- Coder: Implements changes — creates new files, modifies existing files, writes and runs tests
- Reviewer: Validates via REVIEW_PASSED/REVIEW_FAILED token scanning (same pattern as CoderPipeline)
- Writer: Produces `feature_report.md`

### SelfEvolutionPipeline (v1.8)
```
analyzer → planner → coder ↔ reviewer (max 3 iterations) → writer → END
```
- Analyzer: Scans UMAF codebase and agent logs → `analysis_report.json`
- Planner: Creates improvement plan → `implementation_plan.json`
- Coder: Implements changes, detects via git diff or mtime
- Reviewer: Verifies with test suite, REVIEW_PASSED/REVIEW_FAILED token scanning
- Writer: Produces `evolution_report.md`
- Safety: Operates in current git branch; changes revertible with `git checkout -- .`

### PlanPipeline (v1.9)
```
scanner → decomposer → [dependency ‖ risk ‖ resource ‖ cross-cutting] → writer → END
```
- Scanner: Analyzes project directory → `project_context.json`. Skips when `file_manifest` exists (resume).
- Decomposer: Builds hierarchical task tree → `task_tree.json`. Skips when `task_tree` exists. Falls back to template.
- 4 parallel analyzers: Dependency graph, Risk matrix, Resource estimates, Cross-cutting concerns
- Writer: Synthesizes → `plan_spec.json` + `plan_report.md`

## Eight Tools + ToolRegistry (`tools/`)
`read_file`, `write_file`, `write_lines` (preferred for code), `run_command` (30s), `call_claude` (120s), `web_search` (DuckDuckGo), `web_fetch` (urllib, 20s), `download_file` (urllib, 30s). Modular package: `registry.py` + `functions.py` + `feature_tools.py`. `ToolRegistry` with 35 role-specific classmethods — no duplicated definitions. Tools assigned via `tools_config.json` (single source of truth).

## Circuit Breakers

**Agent-level** (`agent.py`):
- **Built-in retry**: `AgentRole._MAX_RETRIES=3` — auto version-bump retry with checkpoint context reuse. All agents get this for free.
- Force wrap-up at ≤3 steps (forbid all tools except write_file)
- Error spiral: 2 consecutive errors → forced best-effort summary
- Mid-loop write reminder at ~2/3 of max steps
- Post-loop forced summary if all steps exhausted
- Claude CLI: if files were written before timeout, treat as success

**Pipeline-level** (`pipeline/`):
- Head agent: 120s timeout → fallback decomposition
- Workers: 600s timeout, max 2 concurrent, stop-on-failure blocks downstream
- Version-bump retry: max 3 retries, max 4 versions, context reuse via checkpoints
- Topology designer↔evaluator retry: max 3 iterations, score threshold 35/50
- Feature coder↔reviewer version-aware retry: max 5 versions
- MD5 dedup; router always moves forward (`researched_partial` accepted)
- Guard clauses for resume: scanner/decomposer skip when state already populated
- All stages have deterministic fallbacks

## Key Design Decisions
- OOP 5-layer hierarchy: Data types → Infrastructure → Agent core → Concrete roles → Pipeline classes
- `AgentRole` ABC template method + `ToolRegistry` centralization
- Explicit `working_dir` — no global state
- Python >= 3.11: `X | None` syntax
- Fallbacks at every pipeline stage
- Backend-aware task generation (no nested `claude -p`)
- Dependency-aware execution with stop-on-failure
- Tool assignment driven by `tools_config.json` (v1.7)

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
| v1.6 | Jun 2026 | Feature Pipeline + modular package structure, 6 pipelines, 23 roles |
| v1.6.1 | Jun 2026 | Dependency injection fixes: Coder, Skill, CoderPP pipelines |
| v1.7 | Jun 2026 | tools_config.json, code dedup (~200 lines), dead code removal, backend-agnostic defaults |
| v1.8 | Jun 2026 | Self-Evolution Pipeline, 175 behavioral tests, 32 roles, 7 pipelines, 379 tests |
| **v1.9** | Jun 2026 | Plan Pipeline, AgentRole built-in retry, Topology+Feature retry loops, 39 roles, 8 pipelines, 403 tests, default parallel testing |

### Related
[[version_diffs]], [[key_updates]], [[oop_refactoring]]
