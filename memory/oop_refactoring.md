---
name: oop-refactoring
description: OOP reorganization — 5-layer class hierarchy with 32 concrete roles and 7 pipeline classes. v1.4+, updated v1.8.
metadata: 
  node_type: memory
  type: project
  originSessionId: 9c942f95-5fb5-4276-8e53-c16059ca5e31
---

## OOP Architecture (v1.4+, updated v1.8)

Reorganized UMAF from procedural/functional to object-oriented with a 5-layer class hierarchy.

### Layer 0 — Data Types
- `AgentResult` dataclass (agent.py) — messages, iterations, success
- `ToolSpec` dataclass (tools/registry.py) — name, description, parameters

### Layer 1 — Infrastructure
- `LLMProvider` ABC + `DeepSeekProvider` + `ClaudeCLIProvider` (llm.py) — unified backend interface
- `ToolRegistry` class (tools/registry.py) — centralized tool specs, 23 role-specific classmethods
- `ClaudeConfig` class (claude_config.py) — lazy-loading config

### Layer 2 — Agent Core
- `BaseAgent` — autonomous agent loop with circuit breakers
- `AgentRole` ABC — template method: `tools_for_backend()`, `build_task()`, `parse_result()`, `execute()`
- `BaseDecomposerRole` — shared decomposition logic with `_extract_json_array()`

### Layer 3 — 32 Concrete Roles

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
- `ObserverRole` (head_agent.py)

**Topology pipeline** (topology/):
- `TopologyAnalyzerRole` (analyzer.py)
- `TopologyDesignerRole` (designer.py)
- `TopologyEvaluatorRole` (evaluator.py)
- `TopologyWriterRole` (writer.py)

**Skill pipeline** (skill/):
- `SkillScannerRole` (scanner.py)
- `DomainExpertiseDetectorRole` (detectors.py)
- `TechnicalCraftDetectorRole` (detectors.py)
- `MethodologyDetectorRole` (detectors.py)
- `RigorDetectorRole` (detectors.py)
- `SkillAggregatorRole` (aggregator.py)
- `SkillReportWriterRole` (writer.py)

**Feature pipeline** (feature/):
- `FeatureScannerRole` (scanner.py)
- `FeaturePlannerRole` (planner.py)
- `FeatureCoderRole` (coder.py)
- `FeatureReviewerRole` (reviewer.py)
- `FeatureReportWriterRole` (writer.py)

**Self-Evolution pipeline** (self_evolution/):
- `SelfEvolutionAnalyzerRole` (analyzer.py)
- `SelfEvolutionPlannerRole` (planner.py)
- `SelfEvolutionCoderRole` (coder.py)
- `SelfEvolutionReviewerRole` (reviewer.py)
- `SelfEvolutionWriterRole` (writer.py)

### Layer 4 — 7 Pipeline Classes (pipeline/)
- `BasePipeline` → `CoderPipeline`, `ResearchPipeline`, `CoderPPPipeline`, `TopologyPipeline`, `SkillPipeline`, `FeaturePipeline`, `SelfEvolutionPipeline`

### Layer 5 — State Types (pipeline/*.py)
- `MultiAgentState`, `ResearchState`, `CoderPPState`, `TopologyState`, `SkillState`, `FeatureState`, `SelfEvolutionState`

### Abstract Base Classes
- `AgentRole` (ABC) — base for all agent roles
- `BaseDecomposerRole` (AgentRole) — shared decomposition logic
- `_BaseDetectorRole` (AgentRole) — base for skill detectors

### Key Wins
- Tool specs defined once in `ToolRegistry` (was duplicated in 8+ files)
- `AgentRole.execute()` replaces the copy-pasted `run_agent()` pattern
- Backend branching centralized in Role classes, not at every call site
- All existing behavior preserved (circuit breakers, checkpoints, dedup, fallbacks)
- Tool assignment driven by `tools_config.json` — no hardcoded tool lists in code

### Anti-pattern: double-parse
`AgentRole.execute()` internally calls `parse_result()` and returns the parsed dict. Calling `role.parse_result()` on the return value causes `'dict' object has no attribute 'messages'`. Pipeline nodes should use the return value of `execute()` directly.

**Why:** Eliminate code duplication, improve extensibility, enable type-safe agent composition.
**How to apply:** New agent roles subclass `AgentRole`. New pipelines subclass `BasePipeline`. Use `ToolRegistry` for tool definitions, never module-level lists. Define tool assignments in `tools_config.json`.
