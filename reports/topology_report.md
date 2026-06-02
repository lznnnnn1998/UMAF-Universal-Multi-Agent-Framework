# Topology Evaluation Report: Technology Skill Detection Pipeline

**Date:** 2026-06-02  
**Pipeline:** `TopologyPipeline`  
**Recommended Topology:** Domain-Parallel Detection  
**Design Pattern:** Fan-Out / Fan-In  
**Overall Score:** 36 / 50

---

## 1. Requirement Summary

> Analyze a software project directory to extract and catalog skills: identify programming languages, frameworks, libraries, design patterns, domain knowledge areas, and tooling. The pipeline should scan the project structure, read source files, detect technology stacks, and produce a structured JSON skill inventory with proficiency levels (`detected` / `used` / `extensively-used`) plus a human-readable markdown summary report organized by skill category.

**Key characteristics of this problem:**
- **Embarrassingly parallel** — per-file and per-category technology detection can proceed independently with zero cross-worker dependencies.
- **I/O-bound** — agents spend most of their time reading source files, not computing.
- **Domain-specialized** — Python patterns differ fundamentally from JavaScript patterns; infrastructure-as-code differs from application code. Domain expertise improves detection accuracy.
- **Batch workload** — no real-time latency requirements; acceptable runtime ranges from seconds (small projects) to minutes (large monorepos).
- **Moderate scale** — projects range from tens to thousands of files; skill categories are bounded (5–15 languages, 10–50 frameworks/libraries, 5–15 design patterns).

---

## 2. Candidate Topology Comparison

Four topologies were designed, evaluated, and scored across five dimensions (each scored 1–10, max total = 50).

| # | Topology | Pattern | Latency | Reliability | Cost | Simplicity | Scalability | **Total** |
|---|----------|---------|:-------:|:-----------:|:----:|:----------:|:-----------:|:---------:|
| 1 | **Domain-Parallel Detection** | fan-out/fan-in | **8** | 7 | 6 | 7 | **8** | **36** |
| 2 | Multi-Strategy Consensus | debate/consensus | 7 | **9** | 3 | 6 | 6 | **31** |
| 3 | Domain-Lead Orchestration | hierarchical | 6 | 6 | 3 | 4 | **8** | **27** |
| 4 | Linear Pipeline | sequential | 3 | 2 | **9** | **9** | 3 | **26** |

### Score Visualization

```
Latency        Domain-Parallel  ████████░░ 8   Consensus       ███████░░░ 7
               Lead-Orch        ██████░░░░ 6   Linear          ███░░░░░░░ 3

Reliability    Consensus        █████████░ 9   Domain-Parallel ███████░░░ 7
               Lead-Orch        ██████░░░░ 6   Linear          ██░░░░░░░░ 2

Cost           Linear           █████████░ 9   Domain-Parallel ██████░░░░ 6
               Consensus        ███░░░░░░░ 3   Lead-Orch       ███░░░░░░░ 3

Simplicity     Linear           █████████░ 9   Domain-Parallel ███████░░░ 7
               Consensus        ██████░░░░ 6   Lead-Orch       ████░░░░░░ 4

Scalability    Domain-Parallel  ████████░░ 8   Lead-Orch       ████████░░ 8
               Consensus        ██████░░░░ 6   Linear          ███░░░░░░░ 3
```

### Topology Details

#### #1 Domain-Parallel Detection (Score: 36/50) ⭐ RECOMMENDED

| Dimension | Score | Assessment |
|-----------|:-----:|-------------|
| Latency | 8 | Four domain detectors run in parallel after the scanner. Critical path is only 4 stages deep (scan → max_detector → aggregate → write). Domain-filtered reads keep per-detector work low. Near-optimal wall-clock time for I/O-bound detection. |
| Reliability | 7 | Independent parallel detectors provide strong fault isolation — a js_detector crash does not block python_detector or infra_detector from completing. The project_scanner is a single point of failure with no retry loop, capping the score. |
| Cost Efficiency | 6 | Seven agents and nine edges incur moderate LLM call overhead. Domain-filtered file subsets keep per-detector token usage lower than full-project scans, but the aggregator adds a dedicated merge agent that a simpler topology could fold into the writer. |
| Simplicity | 7 | Classic fan-out/fan-in is well-understood and straightforward to implement. Domain boundaries provide natural interface contracts. The aggregator's dedup and cross-file frequency logic adds moderate complexity, but the overall DAG is easy to trace and test. |
| Scalability | 8 | Adding a new technology domain (e.g., Rust, Go, Mobile) requires only one new parallel detector agent — the fan-out pattern scales linearly with domain count without increasing critical path depth. File volume growth is absorbed by domain splitting. |

**Strengths:**
- Maximizes parallelism in the I/O-bound detection phase — 4 domain specialists scan files concurrently, achieving up to 4× wall-clock reduction.
- Domain specialization improves detection accuracy — each detector focuses on idioms, dependency formats, and pattern conventions specific to its domain.
- Strong fault isolation — a failure in one detector does not block the other three from completing.
- Naturally extensible — new domains added as parallel agents without restructuring the pipeline.

**Weaknesses:**
- The project_scanner is a single point of failure — mitigated with retry + fallback strategy.
- Proficiency classification requires a sequential aggregator bottleneck — inherent to cross-file analysis.
- Domain boundary assignment can be imprecise for polyglot files — handled by the aggregator's dedup logic.

#### #2 Multi-Strategy Consensus (Score: 31/50)

**Best for:** High-stakes audits where accuracy trumps cost. The method-diversity approach (heuristic + semantic + structural) catches detections that any single strategy would miss. The 3× cost premium is justified when missing a critical dependency has significant consequences. Overkill for routine technology inventory tasks.

**Key trade-off:** 3× the file reads and agent compute for built-in confidence scoring and majority-vote accuracy guarantees.

#### #3 Domain-Lead Orchestration (Score: 27/50)

**Best for:** Very large multi-language monorepos (5+ distinct technology domains) where intermediate aggregation by domain leads provides genuine value. The two-tier hierarchy scales well but adds significant coordination overhead. Over-engineered for typical projects.

**Key trade-off:** 10 agents with 18 bidirectional edges make this the most expensive and complex topology. Bidirectional communication doubles the coordination token cost.

#### #4 Linear Pipeline (Score: 26/50)

**Best for:** Quick prototyping and very small projects (<100 files) where simplicity and low cost are paramount. The simplest topology to implement, test, and debug. Unsuitable for production-grade detection on medium-to-large projects due to zero parallelism and poor fault tolerance.

**Key trade-off:** Simplicity at the cost of speed, reliability, and scalability. Total wall-clock time equals the sum of all agent runtimes with no concurrent execution.

---

## 3. Recommendation: Domain-Parallel Detection

### Why This Topology Won

**Domain-Parallel Detection** achieves the highest total score (36/50) by striking the best balance across all five evaluation dimensions. It outperforms the alternatives as follows:

| vs. Alternative | Winning Dimension | Why |
|-----------------|-------------------|-----|
| vs. Linear Pipeline | Latency (+5), Reliability (+5), Scalability (+5) | Parallelism and fault isolation make it viable for real projects; Linear is only suitable for prototyping |
| vs. Domain-Lead Orchestration | Latency (+2), Cost (+3), Simplicity (+3) | Flat fan-out avoids the "middle manager" overhead of domain leads; same scalability at lower cost |
| vs. Multi-Strategy Consensus | Cost (+3), Scalability (+2) | Domain-split reads use ~1/3 the tokens of full-project scans; better cost scaling as project size grows |

**The decisive factors:**

1. **Right level of parallelism** — Four domain detectors match the natural decomposition of the problem (languages, frameworks, infrastructure, tooling). More parallelism (Domain-Lead) adds coordination overhead without proportional benefit. Less parallelism (Linear) leaves I/O bandwidth idle.

2. **Domain specialization without redundancy** — Each detector only processes its domain's files, unlike Multi-Strategy Consensus where all three detectors re-read the entire project. This is the key cost advantage: token usage scales with project size, not project size × strategy count.

3. **Fault isolation matches error domains** — If the JavaScript detector crashes on a malformed `package.json`, the Python, Infrastructure, and Config detectors complete normally. The pipeline degrades gracefully rather than failing entirely.

4. **Natural extensibility** — Adding support for Rust, Go, or mobile development means adding one new parallel agent. The critical path depth stays constant. The aggregator's merge logic handles the new domain transparently.

### When to Choose Another Topology

- **Choose Multi-Strategy Consensus** when detection accuracy is critical and cost is no object (e.g., security audits, compliance checks, due diligence for acquisitions). The 3× redundancy and confidence scoring provide verifiable accuracy guarantees.
- **Choose Domain-Lead Orchestration** for very large monorepos (50,000+ files, 5+ distinct technology domains) where intermediate domain-lead aggregation reduces the chief orchestrator's cognitive load.
- **Choose Linear Pipeline** for quick prototyping, demos, or projects small enough to fit in a single agent context (<100 files). It's the fastest to implement and easiest to debug.

---

## 4. Implementation Notes

### Architecture Overview

```
project_scanner → [python_detector || js_detector || infra_detector || config_docs_detector] → aggregator → report_writer → END
```

- **7 agents**, 5 distinct role classes extending `AgentRole` ABC
- **9 directed edges** forming a star-topology DAG
- **1 synchronization barrier** at the aggregator
- **Concurrency:** 4 parallel detectors, capped at 4 for filesystem I/O saturation management

### Agent Roles

| Agent | Role Class | Tools | Max Steps | Output |
|-------|-----------|-------|:---------:|--------|
| project_scanner | `TopologyScannerRole` | Bash, Read | 15 | `file_inventory.json` |
| python_detector | `TopologyPythonDetectorRole` | Read, Bash | 20 | `python_skills.json` |
| js_detector | `TopologyJavascriptDetectorRole` | Read, Bash | 20 | `js_skills.json` |
| infra_detector | `TopologyInfrastructureDetectorRole` | Read, Bash | 15 | `infra_skills.json` |
| config_docs_detector | `TopologyConfigDocsDetectorRole` | Read, Bash | 15 | `config_docs_skills.json` |
| aggregator | `TopologyAggregatorRole` | Read, Write | 20 | `aggregated_skills.json` |
| report_writer | `TopologyWriterRole` | Write | 15 | `skills_inventory.json` + `skills_report.md` |

### Data Flow Contracts

All intermediate files follow consistent JSON schemas:

- **`file_inventory.json`** — Domain-grouped file catalog with per-file path, extension, and size metadata. Serves as the API contract between scanner and detectors.
- **`{domain}_skills.json`** — Per-domain skill detection results with canonical skill names, occurrence counts, and detection methods. Each detector writes independently.
- **`aggregated_skills.json`** — Merged and deduplicated inventory with proficiency levels assigned by cross-file frequency analysis.
- **`skills_inventory.json`** + **`skills_report.md`** — Final outputs: structured JSON and human-readable markdown.

### Error Handling Strategy

| Failure Point | Strategy |
|---------------|----------|
| Scanner crash | Retry once with simplified prompt; fall back to raw `find` output if retry fails |
| Detector crash | Failed domain produces empty skills array with error field; pipeline continues with partial results |
| Aggregator crash | Detector outputs checkpointed to disk; aggregator can be re-run independently |
| Writer crash | Aggregator output preserved on disk; writer can be re-run as standalone step |

### Proficiency Level Thresholds

| Level | Occurrence Count | Meaning |
|-------|:----------------:|---------|
| `detected` | 1–2 | Technology is present but minimally used (e.g., a single import, a dev dependency) |
| `used` | 3–10 | Technology is actively used across multiple files or modules |
| `extensively-used` | 11+ | Technology is a core part of the project, used pervasively |

Thresholds are configurable via `PROFICIENCY_THRESHOLDS` in the pipeline configuration.

### Key Design Decisions

1. **Domain-split parallelism** — File inventory split into 4 mutually-exclusive groups. Each detector reads only its domain's files, avoiding redundant full-project scans.
2. **File-inventory-as-contract** — The scanner's `file_inventory.json` is the canonical API between stages. Detectors can be developed and tested independently against mock inventories.
3. **Checkpointed outputs** — Every agent writes results to disk before signaling completion. Enables independent re-run of downstream agents without re-executing upstream work.
4. **Centralized proficiency computation** — The aggregator computes all proficiency levels, ensuring consistent thresholds across domains.
5. **Zero inter-detector communication** — Detectors operate in complete isolation. No coordination overhead, no cascading failures, trivially correct parallelism.

---

## 5. Next Steps

### Phase 1: Scaffold the Pipeline Class

1. Create `TopologyPipeline` extending `BasePipeline` in `pipeline.py`.
2. Register the new pipeline mode (`--mode topology`) in `main.py`.
3. Define the flow dict: `scanned → detectors`, `detected → aggregator`, `aggregated → writer`, `written → END`.

### Phase 2: Implement Agent Roles

1. **`TopologyScannerRole`** — `agent.py` subclass. Bash-first file cataloging with fallback. Writes `file_inventory.json`.
2. **`TopologyPythonDetectorRole`** — Python-aware detection: `import` parsing, `requirements.txt` analysis, `pyproject.toml` parsing.
3. **`TopologyJavascriptDetectorRole`** — JS/TS-aware detection: `package.json` parsing, import pattern analysis, framework convention detection.
4. **`TopologyInfrastructureDetectorRole`** — Infra-aware detection: Dockerfile parsing, CI config analysis, IaC manifest scanning.
5. **`TopologyConfigDocsDetectorRole`** — Config/docs detection: YAML/JSON/TOML parsing, markdown analysis, build system identification.
6. **`TopologyAggregatorRole`** — Merge, dedup, proficiency computation, consistency validation.
7. **`TopologyWriterRole`** — JSON inventory generation + markdown report rendering.

### Phase 3: Add Tools (if needed)

- The existing `ToolRegistry` (read_file, write_file, run_command) covers all requirements.
- Consider adding a `jq`-based JSON merge helper in the aggregator for performance on large inventories.

### Phase 4: Testing

- Unit tests for each role class with mock file inventories.
- Integration test with a small polyglot project (Python + JS + Docker + CI config).
- End-to-end test verifying output schema compliance and proficiency level correctness.
- Stress test with a large monorepo (10,000+ files) to validate parallelism and token usage.

### Phase 5: Hardening

- Scanner retry + fallback implementation.
- Detector timeout handling (default 600s per detector).
- Aggregator checkpoint/re-run support.
- Add topology pipeline to the smoke test suite.

---

## Appendix: Score Definitions

| Score | Rating | Description |
|:-----:|--------|-------------|
| 9–10 | Excellent | Near-optimal; best-in-class for this dimension |
| 7–8 | Good | Strong performance; solid choice for most use cases |
| 5–6 | Adequate | Acceptable; functional but with notable trade-offs |
| 3–4 | Weak | Significant shortcomings; only viable in constrained scenarios |
| 1–2 | Poor | Fundamentally unsuitable; avoid for production use |

---

*Report generated by TopologyPipeline evaluation framework. All four candidate topologies were scored by independent evaluator agents against the same five-dimension rubric. The recommended topology (Domain-Parallel Detection) was selected by total score ranking with tie-breaking on latency and scalability.*
