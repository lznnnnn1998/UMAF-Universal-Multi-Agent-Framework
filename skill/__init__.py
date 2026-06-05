"""Public API for the Skill Summarizer Pipeline v2.

Seven AgentRole subclasses implementing artifact-agnostic skill detection:

Scanner:
- SkillScannerRole: Artifact classification + deep content reading

Detectors (4 skill dimensions):
- DomainExpertiseDetectorRole: What specialized knowledge is demonstrated?
- TechnicalCraftDetectorRole: How skilled is the creator at the medium?
- MethodologyDetectorRole: What tools, workflows, and processes are evident?
- RigorDetectorRole: How thorough, careful, and complete is the work?

Aggregator:
- SkillAggregatorRole: Merges tools + inferred skills, deduplicates

Writer:
- SkillReportWriterRole: Produces skills.json + skills_report.md
"""

from skill.scanner import SkillScannerRole
from skill.detectors import (
    DomainExpertiseDetectorRole,
    MethodologyDetectorRole,
    RigorDetectorRole,
    TechnicalCraftDetectorRole,
)
from skill.aggregator import SkillAggregatorRole
from skill.writer import SkillReportWriterRole

__all__ = [
    "SkillScannerRole",
    "DomainExpertiseDetectorRole",
    "TechnicalCraftDetectorRole",
    "MethodologyDetectorRole",
    "RigorDetectorRole",
    "SkillAggregatorRole",
    "SkillReportWriterRole",
]
