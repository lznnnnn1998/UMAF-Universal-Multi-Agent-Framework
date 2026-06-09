"""UMAF test suite — covering all 7 pipelines.

Structure
---------
- ``conftest.py``: shared fixtures, config loading, mock helpers, ``tmpdir`` fixture
- ``test_smoke.py``: core agent, pipeline, ToolRegistry, checkpoint
- ``test_pipeline.py``: BasePipeline, topological levels, dependency validation
- ``test_coder.py``: CoderPipeline, CoderRole, ReviewerRole, state, graph nodes
- ``test_research.py``: ResearchPipeline, decomposer, reviewer, writer, resume
- ``test_coderpp.py``: CoderPPPipeline, decomposer, observer, organizer, workers
- ``test_topology.py``: Topology Pipeline roles, state, fallbacks, E2E
- ``test_skill.py``: Skill Pipeline v2 roles, detectors, fallback chain
- ``test_feature.py``: Feature Pipeline roles, state, mock E2E
- ``test_self_evolution.py``: SelfEvolutionPipeline, 5 roles, graph nodes
"""

from __future__ import annotations

from test.conftest import make_agent_result

__all__ = ["make_agent_result"]
