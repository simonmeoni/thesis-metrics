"""Core functionality for privacy evaluation and rephrasing"""

from thesis_metrics.core.evaluation import PrivacyEvaluator
from thesis_metrics.core.privacy_attacks import (
    linkage_attack_tfidf,
    proximity_attack_random,
    proximity_attack_tfidf,
    random_leakage_attack,
)
from thesis_metrics.core.rephrasing import Rephraser, run_rephrasing_sync
from thesis_metrics.core.downstream import (
    SemanticSimilarityEvaluator,
    ICDClassifier,
    NERClassifier,
)

__all__ = [
    "PrivacyEvaluator",
    "linkage_attack_tfidf",
    "proximity_attack_random",
    "proximity_attack_tfidf",
    "random_leakage_attack",
    "Rephraser",
    "run_rephrasing_sync",
    "SemanticSimilarityEvaluator",
    "ICDClassifier",
    "NERClassifier",
]
