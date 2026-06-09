"""Self-Evolution module — agent roles that analyze and improve UMAF itself."""

from __future__ import annotations

from self_evolution.analyzer import SelfEvolutionAnalyzerRole
from self_evolution.planner import SelfEvolutionPlannerRole
from self_evolution.coder import SelfEvolutionCoderRole
from self_evolution.reviewer import SelfEvolutionReviewerRole
from self_evolution.writer import SelfEvolutionWriterRole

__all__ = [
    "SelfEvolutionAnalyzerRole",
    "SelfEvolutionPlannerRole",
    "SelfEvolutionCoderRole",
    "SelfEvolutionReviewerRole",
    "SelfEvolutionWriterRole",
]
