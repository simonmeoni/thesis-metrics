"""Downstream task evaluation for utility assessment of synthetic medical data"""

from thesis_metrics.core.downstream.baseline import SemanticSimilarityEvaluator
from thesis_metrics.core.downstream.icd_classification import ICDClassifier
from thesis_metrics.core.downstream.ner_classification import NERClassifier

__all__ = [
    "SemanticSimilarityEvaluator",
    "ICDClassifier",
    "NERClassifier",
]
