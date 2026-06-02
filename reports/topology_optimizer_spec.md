# Topology Optimizer Pipeline — Requirement Specification

## Overview

Build a **TopologyPipeline** that, given a task description, determines the optimal agent topology: how many agents, what roles, what tools each needs, how they connect, and what parallelism strategy to use.

The pipeline itself follows UMAF's standard pattern: `BasePipeline` subclass with a LangGraph state machine, using `AgentRole` subclasses for each stage.

## Pipeline Flow

```
analyzer (assess task complexity)
  → designer (propose 2-4 candidate topologies)
  → evaluator (score each topology on 5 dimensions)
  → writer (select best, produce final topology_spec.json + report.md)
```

No retry loops needed — this is a linear analysis→design→evaluate→write pipeline.

## State Definition

The `TopologyState` TypedDict should have these keys:
- `input_spec: str` — the task description
- `working_dir: str`
- `backend: str`
- `complexity_factors: dict[str, Any]` — from analyzer (data_dependencies, parallelism_opportunities, tool_requirements, error_domains, latency_sensitivity, scale)
- `candidate_topologies: list[dict[str, Any]]` — from designer (each is a topology proposal)
- `evaluated_topologies: list[dict[str, Any]]` — from evaluator (topologies with scores)
- `topology_spec: dict[str, Any]` — from writer (the final selected topology)
- `status: str`

## Agent Role 1: TopologyAnalyzerRole

**File**: `topology/analyzer.py`
**Class**: `TopologyAnalyzerRole(AgentRole)`
**agent_name**: `"topology_analyzer"`
**max_steps**: 8

### Responsibilities
Analyze a task description and identify complexity factors that influence topology design:
1. **Data dependencies** — does the task have sequential data flow? Independent sub-tasks? Shared state?
2. **Parallelism opportunities** — which sub-tasks can run in parallel?
3. **Tool requirements** — what tools does each sub-task need (read, write, search, compute, etc.)?
4. **Error domains** — where can failures occur? Should failures in one part block others?
5. **Latency sensitivity** — is the task latency-sensitive (favor parallelism) or throughput-sensitive (favor depth)?
6. **Scale** — how many distinct sub-problems are there? (2-4, 4-8, 8+)

### Output
A JSON object with the six complexity factors above. Each factor should have a `"level"` (low/medium/high) and `"reasoning"` (1-2 sentence explanation).

### Tools
- For deepseek: `read_file`, `write_file`, `call_claude`
- For claude_cli: `Read`, `Write` (no web search needed — this is analytical reasoning)

### Task Prompt
The agent receives the user's task description and outputs structured JSON analysis. Backend-aware: claude_cli version uses native tool names.

## Agent Role 2: TopologyDesignerRole

**File**: `topology/designer.py`
**Class**: `TopologyDesignerRole(AgentRole)`
**agent_name**: `"topology_designer"`
**max_steps**: 12

### Responsibilities
Given the complexity analysis, propose 2-4 candidate topologies. Each topology is a complete agent configuration:

```json
{
  "topology_id": "topology_a",
  "name": "Pipeline with 3 sequential agents",
  "description": "A linear pipeline where each agent feeds into the next...",
  "agents": [
    {
      "id": 1,
      "role_name": "analyzer",
      "description": "Analyzes input and extracts key information",
      "tools": ["read_file", "web_search"],
      "max_steps": 8,
      "timeout_seconds": 300
    }
  ],
  "connections": [
    {"from": 1, "to": 2, "type": "sequential"},
    {"from": 2, "to": 3, "type": "sequential"}
  ],
  "parallelism_strategy": "sequential",
  "max_parallel_agents": 1,
  "estimated_total_steps": 24,
  "strengths": ["Simple to reason about", "Low resource usage"],
  "weaknesses": ["Higher latency", "No parallelism"]
}
```

### Design Principles to Follow
- **Sequential pipeline**: agents in a chain, each feeds into next. Best for tasks with clear data dependencies.
- **Fan-out/fan-in**: one decomposer, N parallel workers, one aggregator. Best for independent sub-tasks.
- **Debate/consensus**: multiple agents work on same problem, reviewer picks best. Best for creative/ambiguous tasks.
- **Hierarchical**: agents at multiple levels, higher-level agents coordinate lower-level ones. Best for very complex tasks.
- Vary the number of agents: simpler topologies have 2-3 agents, complex ones have 4-7.
- Vary parallelism: some sequential, some parallel with fan-out.
- Each topology should have different strengths/weaknesses to give the evaluator real trade-offs.

### Tools
- For deepseek: `read_file`, `write_file`, `call_claude`
- For claude_cli: `Read`, `Write`

## Agent Role 3: TopologyEvaluatorRole

**File**: `topology/evaluator.py`
**Class**: `TopologyEvaluatorRole(AgentRole)`
**agent_name**: `"topology_evaluator"`
**max_steps**: 10

### Responsibilities
Score each candidate topology on 5 dimensions (each 1-10):
1. **Latency** — how fast can this topology complete? (higher parallelism = higher score)
2. **Reliability** — how well does it handle failures? (fewer single points of failure = higher score)
3. **Cost efficiency** — how many total agent steps? (fewer steps = higher score)
4. **Simplicity** — how easy to understand and debug? (fewer agents/connections = higher score)
5. **Scalability** — how well does it handle growth in task complexity? (modular design = higher score)

### Output
A JSON array of evaluations, each with the topology_id, scores dict, total_score, rank, and reasoning.

### Tools
- For deepseek: `read_file`, `write_file`, `call_claude`
- For claude_cli: `Read`, `Write`

## Agent Role 4: TopologyWriterRole

**File**: `topology/writer.py`
**Class**: `TopologyWriterRole(AgentRole)`
**agent_name**: `"topology_writer"`
**max_steps**: 8

### Responsibilities
1. Select the best topology (highest total score)
2. Write `topology_spec.json` — the complete topology specification in a format that can be consumed by CoderPP or another pipeline generator
3. Write `topology_report.md` — a human-readable report explaining the recommendation with comparison table
4. The topology_spec.json should include a section `"pipeline_implementation_guide"` that describes how to implement this topology as a UMAF pipeline (what classes to create, what flow dict to use, what state keys are needed)

### Tools
- For deepseek: `write_file`
- For claude_cli: `Write`

## Pipeline Class: TopologyPipeline

**File**: `pipeline.py` (add to existing file, after CoderPPPipeline)
**Class**: `TopologyPipeline(BasePipeline)`
**name**: `"topology"`
**default_output_dir**: `"topology_output"`

### Methods
- `_decompose(input_spec)` → runs analyzer, returns empty list (no traditional decomposition needed — the pipeline's own stages handle everything)
- `_display_decomposition()` → prints the complexity factors
- `_build_initial_state()` → creates TopologyState
- `_build_graph()` → builds the 4-node LangGraph:
  - `analyzer_node` → reads state, runs `analyze_complexity()`, sets `complexity_factors`, transitions to `designer`
  - `designer_node` → reads complexity_factors, runs `design_topologies()`, sets `candidate_topologies`, transitions to `evaluator`
  - `evaluator_node` → reads candidate_topologies, runs `evaluate_topologies()`, sets `evaluated_topologies`, transitions to `writer`
  - `writer_node` → reads evaluated_topologies, runs `write_topology_spec()`, sets `topology_spec`, transitions to END
- `_print_results()` → prints the recommended topology name, agent count, key metrics, and output file paths

## ToolRegistry Extensions

Add the following class methods to `ToolRegistry` in `tools.py`:
- `topology_analyzer_tools()` — `read_file`, `write_file`
- `topology_designer_tools()` — `read_file`, `write_file`
- `topology_evaluator_tools()` — `read_file`, `write_file`
- `topology_writer_tools()` — `write_file` only

## Tests

Create `test_topology.py` with the following tests:
1. **test_imports** — all topology modules import cleanly
2. **test_analyzer_instantiation** — TopologyAnalyzerRole can be instantiated with correct agent_name and max_steps
3. **test_designer_instantiation** — TopologyDesignerRole can be instantiated
4. **test_evaluator_instantiation** — TopologyEvaluatorRole can be instantiated
5. **test_writer_instantiation** — TopologyWriterRole can be instantiated
6. **test_tools_for_backend** — all roles return non-empty tool lists for both backends
7. **test_topology_state** — TopologyState TypedDict has all required keys
8. **test_pipeline_instantiation** — TopologyPipeline can be instantiated
9. **test_pipeline_flow** — flow dict has correct transitions
10. **test_fallback_analyzer** — analyzer has fallback complexity analysis (when LLM is unavailable, use keyword-based analysis)
11. **test_e2e_with_mock** — mock the LLM, run pipeline end-to-end with a simple task like "build a pipeline that summarizes text"

## Integration

Register in `main.py`:
```python
from pipeline import CoderPipeline, ResearchPipeline, CoderPPPipeline, TopologyPipeline

PIPELINES = {
    "coder": CoderPipeline,
    "research": ResearchPipeline,
    "coderpp": CoderPPPipeline,
    "topology": TopologyPipeline,
}
```

## Fallback Behavior

Every agent role should have fallback logic:
- **Analyzer fallback**: Keyword-based complexity analysis — count keywords related to parallelism, dependencies, error handling in the task description
- **Designer fallback**: Generate 2 standard topologies (sequential 3-agent pipeline, fan-out with 2 workers)
- **Evaluator fallback**: Simple heuristic scoring based on agent count and parallelism
- **Writer fallback**: Template-based topology_spec.json using default values

## Key Design Constraints

- Follow UMAF patterns: `AgentRole` ABC, `ToolRegistry` centralized tool specs, `BasePipeline` with LangGraph
- Python >= 3.11 syntax (`X | None`, not `Optional[X]`)
- Backend-aware task prompts (different instructions for deepseek vs claude_cli)
- No duplicated tool definitions — use ToolRegistry
- Checkpoint support via CheckpointManager for resumability
- All agent logs written to `agent_log/` subdirectory
