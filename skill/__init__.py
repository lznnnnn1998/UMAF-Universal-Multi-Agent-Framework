"""Public API for the skill summarizer agent roles.

Seven AgentRole subclasses implementing the Skill Summarizer Pipeline:
- SkillScannerRole: Scans project directory structure
- PythonDetectorRole: Detects Python ecosystem skills
- JSDetectorRole: Detects JavaScript/Node.js ecosystem skills
- InfraDetectorRole: Detects infrastructure & DevOps skills
- ConfigDocsDetectorRole: Detects config, docs, and API spec skills
- SkillAggregatorRole: Aggregates and deduplicates domain reports
- SkillReportWriterRole: Produces structured output and markdown report
"""

from skill.scanner import SkillScannerRole
from skill.detectors import (
    ConfigDocsDetectorRole,
    InfraDetectorRole,
    JSDetectorRole,
    PythonDetectorRole,
)
from skill.aggregator import SkillAggregatorRole
from skill.writer import SkillReportWriterRole

__all__ = [
    "SkillScannerRole",
    "PythonDetectorRole",
    "JSDetectorRole",
    "InfraDetectorRole",
    "ConfigDocsDetectorRole",
    "SkillAggregatorRole",
    "SkillReportWriterRole",
]
