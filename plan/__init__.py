"""Public API for the Plan Pipeline.

Seven AgentRole subclasses implementing task-to-plan transformation:

Scanner:
- PlanScannerRole: Project context gathering for planning (language, framework, conventions, architecture)

Decomposer:
- PlanDecomposerRole: Hierarchical task decomposition with adaptive depth (goals → epics → stories → tasks)

Parallel Analyzers (4 dimensions):
- PlanDependencyAnalyzerRole: Dependency graph with edge types (blocks, informs, enables, is_subtask_of)
- PlanRiskAssessorRole: Risk assessment across 5 dimensions with mitigation recommendations
- PlanResourceEstimatorRole: Effort estimation with bottom-up roll-up and resource contention detection
- PlanCrossCuttingAnalyzerRole: Cross-cutting concern identification (security, testing, docs, deployment, compliance)

Writer:
- PlanWriterRole: Collects all 6 upstream outputs, performs consistency checks, produces plan_spec.json + plan_report.md
"""

from plan.scanner import PlanScannerRole
from plan.decomposer import PlanDecomposerRole
from plan.dependency import PlanDependencyAnalyzerRole
from plan.risk import PlanRiskAssessorRole
from plan.resource import PlanResourceEstimatorRole
from plan.cross_cutting import PlanCrossCuttingAnalyzerRole
from plan.writer import PlanWriterRole

__all__ = [
    "PlanScannerRole",
    "PlanDecomposerRole",
    "PlanDependencyAnalyzerRole",
    "PlanRiskAssessorRole",
    "PlanResourceEstimatorRole",
    "PlanCrossCuttingAnalyzerRole",
    "PlanWriterRole",
]
