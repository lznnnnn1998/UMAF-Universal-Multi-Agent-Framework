"""Feature Pipeline — agent roles for adding/editing code in existing projects."""

from .scanner import FeatureScannerRole
from .planner import FeaturePlannerRole
from .coder import FeatureCoderRole
from .reviewer import FeatureReviewerRole
from .writer import FeatureReportWriterRole

__all__ = [
    "FeatureScannerRole",
    "FeaturePlannerRole",
    "FeatureCoderRole",
    "FeatureReviewerRole",
    "FeatureReportWriterRole",
]
