"""Topology optimization agent roles.

Four AgentRole subclasses for the UMAF topology optimization pipeline:
- TopologyAnalyzerRole: Assesses task complexity across 6 factors
- TopologyDesignerRole: Proposes 2-4 candidate topologies
- TopologyEvaluatorRole: Scores each topology on 5 dimensions
- TopologyWriterRole: Selects best topology, writes spec and report
"""

from topology.analyzer import TopologyAnalyzerRole
from topology.designer import TopologyDesignerRole
from topology.evaluator import TopologyEvaluatorRole
from topology.writer import TopologyWriterRole

__all__ = [
    "TopologyAnalyzerRole",
    "TopologyDesignerRole",
    "TopologyEvaluatorRole",
    "TopologyWriterRole",
]
