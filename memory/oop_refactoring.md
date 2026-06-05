---
name: oop-refactoring
description: OOP reorganization — 5-layer class hierarchy with 23 concrete roles and 6 pipeline classes. v1.4+.
metadata: 
  node_type: memory
  type: project
  originSessionId: 9c942f95-5fb5-4276-8e53-c16059ca5e31
---

## OOP Architecture (v1.4+, updated v1.6)

Reorganized UMAF from procedural/functional to object-oriented with a 5-layer class hierarchy.

### Layer 0 — Data Types
- `AgentResult` dataclass (agent.py) — messages, iterations, success
- `ToolSpec` dataclass (tools.py) — name, description, parameters

### Layer 1 — Infrastructure
- `LLMProvider` ABC + `DeepSeekProvider` + `ClaudeCLIProvider` (llm.py) — unified backend interface
- `ToolRegistry` class (tools.py) — centralized tool specs, 16+ role-specific classmethods
- `ClaudeConfig` class (claude_config.py) — lazy-loading config

### Layer 2 — Agent Core
- `BaseAgent` — autonomous agent loop with circuit breakers
- `AgentRole` ABC — template method: `tools_for_backend()`, `build_task()`, `parse_result()`, `execute()`

### Layer 3 — 18 Concrete Roles

**Coder pipeline** (pipeline/coder.py):
- `CoderRole`, `ReviewerRole`

**Research pipeline** (research/):
- `ResearchDecomposerRole` (head_agent.py)
- `ResearchWorkerRole` (worker_agent.py)
- `ResearchReviewerRole` (reviewer_agent.py)
- `WriterRole` (writer.py)

**CoderPP pipeline** (coderpp/):
- `CoderPPDecomposerRole` (head_agent.py)
- `CoderPPWorkerRole` (worker_agent.py)
- `CoderPPReviewerRole` (reviewer_agent.py)
- `OrganizerRole` (organizer.py)

**Topology pipeline** (topology/):
- `TopologyAnalyzerRole` (analyzer.py)
- `TopologyDesignerRole` (designer.py)
- `TopologyEvaluatorRole` (evaluator.py)
- `TopologyWriterRole` (writer.py)

**Skill pipeline** (skill/):
- `SkillScannerRole` (scanner.py)
- `PythonDetectorRole` (detectors.py)
- `JSDetectorRole` (detectors.py)
- `InfraDetectorRole` (detectors.py)
- `ConfigDocsDetectorRole` (detectors.py)
- `SkillAggregatorRole` (aggregator.py)
- `SkillReportWriterRole` (writer.py)

**Feature pipeline** (feature/):
- `FeatureScannerRole` (scanner.py)
- `FeaturePlannerRole` (planner.py)
- `FeatureCoderRole` (coder.py)
- `FeatureReviewerRole` (reviewer.py)
- `FeatureReportWriterRole` (writer.py)

### Layer 4 — 6 Pipeline Classes (pipeline/)
- `BasePipeline` → `CoderPipeline`, `ResearchPipeline`, `CoderPPPipeline`, `TopologyPipeline`, `SkillPipeline`, `FeaturePipeline`

### Layer 5 — State Types (pipeline/*.py)
- `MultiAgentState`, `ResearchState`, `CoderPPState`, `TopologyState`, `SkillState`, `FeatureState`

### Files Removed
- `pipeline.py` — replaced by `pipeline/` package (7 modules)
- `tools.py`, `tools_integration.py` — replaced by `tools/` package (3 modules)
- `feature_pipeline.py` — replaced by `feature/` package (5 role files)
- `graph.py`, `research/graph.py`, `coderpp/graph.py` — dead code

### Key Wins
- Tool specs defined once in `ToolRegistry` (was duplicated in 8+ files)
- `AgentRole.execute()` replaces the copy-pasted `run_agent()` pattern
- Backend branching centralized in Role classes, not at every call site
- All existing behavior preserved (circuit breakers, checkpoints, dedup, fallbacks)
- Backward-compatible: `run_agent()`, `decompose_topic()`, etc. still work

### Anti-pattern: double-parse
`AgentRole.execute()` internally calls `parse_result()` and returns the parsed dict. Calling `role.parse_result()` on the return value causes `'dict' object has no attribute 'messages'`. Pipeline nodes should use the return value of `execute()` directly.

**Why:** Eliminate code duplication, improve extensibility, enable type-safe agent composition.
**How to apply:** New agent roles subclass `AgentRole`. New pipelines subclass `BasePipeline`. Use `ToolRegistry` for tool definitions, never module-level lists.
