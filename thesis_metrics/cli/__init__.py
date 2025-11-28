"""Command-line interfaces for thesis metrics"""

from thesis_metrics.cli.alpacare_eval import main as alpacare_eval_main
from thesis_metrics.cli.privacy_metrics import main as privacy_metrics_main
from thesis_metrics.cli.rephrase import main as rephrase_main
from thesis_metrics.cli.downstream_eval import main as downstream_eval_main

__all__ = [
    "privacy_metrics_main",
    "rephrase_main",
    "alpacare_eval_main",
    "downstream_eval_main",
]
