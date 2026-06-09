"""UMAF Pipeline — abstract base and concrete pipeline implementations.

Provides BasePipeline and all six pipeline classes:
- CoderPipeline (coder ↔ reviewer loop)
- ResearchPipeline (head → workers → reviewer → writer)
- CoderPPPipeline (head → workers → reviewer → organizer)
- TopologyPipeline (analyzer → designer → evaluator → writer)
- SkillPipeline (scanner → 4 detectors → aggregator → writer)
- SelfEvolutionPipeline (analyzer → planner → coder ↔ reviewer → writer)
"""

from .base import BasePipeline
from .coder import CoderPipeline
from .research import ResearchPipeline, ResearchState
from .coderpp import CoderPPPipeline, CoderPPState
from .topology import TopologyPipeline, TopologyState
from .skill import SkillPipeline, SkillState
from .feature import FeaturePipeline, FeatureState
from .self_evolution import SelfEvolutionPipeline, SelfEvolutionState

__all__ = [
    "BasePipeline",
    "CoderPipeline",
    "ResearchPipeline", "ResearchState",
    "CoderPPPipeline", "CoderPPState",
    "TopologyPipeline", "TopologyState",
    "SkillPipeline", "SkillState",
    "FeaturePipeline", "FeatureState",
    "SelfEvolutionPipeline", "SelfEvolutionState",
]
