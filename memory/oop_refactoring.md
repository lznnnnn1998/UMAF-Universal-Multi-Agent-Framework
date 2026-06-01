---
name: oop-refactoring
description: OOP reorganization — 5-layer class hierarchy replacing procedural code. May 2026.
metadata: 
  node_type: memory
  type: project
  originSessionId: c3d0976f-73e9-4392-9e4d-e8fbd0bcb43c
---

## OOP Refactoring (May 2026)

Reorganized UMAF from procedural/functional to object-oriented with a 5-layer class hierarchy.

### New Class Hierarchy

**Layer 0 — Data Types:**
- `AgentResult` dataclass (agent.py) — messages, iterations, success
- `ToolSpec` dataclass (tools.py) — name, description, parameters

**Layer 1 — Infrastructure:**
- `LLMProvider` ABC + `DeepSeekProvider` + `ClaudeCLIProvider` (llm.py) — unified backend interface
- `ToolRegistry` class (tools.py) — centralized tool specs, 12 role-specific methods
- `ClaudeConfig` class (claude_config.py) — lazy-loading config, replaces import-time singleton

**Layer 2 — Agent Core:**
- `BaseAgent` — unchanged logic, constructor/return type modernized
- `AgentRole` ABC — template method: `tools_for_backend()`, `build_task()`, `parse_result()`, `execute()`

**Layer 3 — 10 Concrete Roles:**
- `CoderRole`, `ReviewerRole` (pipeline.py)
- `ResearchDecomposerRole` (research/head_agent.py)
- `ResearchWorkerRole` (research/worker_agent.py)
- `ResearchReviewerRole` (research/reviewer_agent.py)
- `WriterRole` (research/writer.py)
- `CoderPPDecomposerRole` (coderpp/head_agent.py)
- `CoderPPWorkerRole` (coderpp/worker_agent.py)
- `CoderPPReviewerRole` (coderpp/reviewer_agent.py)
- `OrganizerRole` (coderpp/organizer.py)

**Layer 4 — 3 Pipeline Classes:**
- `BasePipeline` → `CoderPipeline`, `ResearchPipeline`, `CoderPPPipeline`

### Files Removed
- `graph.py`, `research/graph.py`, `coderpp/graph.py` — dead code, duplicated in pipeline.py

### Key Wins
- Tool specs defined once in `ToolRegistry` (was duplicated in 8+ files)
- `AgentRole.execute()` replaces the copy-pasted `run_agent()` pattern
- Backend branching centralized in Role classes, not at every call site
- All existing behavior preserved (circuit breakers, checkpoints, dedup, fallbacks)
- Backward-compatible: `run_agent()`, `decompose_topic()`, etc. still work

**Why:** Eliminate code duplication, improve extensibility, enable type-safe agent composition.
**How to apply:** New agent roles should subclass `AgentRole`. New pipelines should subclass `BasePipeline`. Use `ToolRegistry` for tool definitions, never module-level lists.
