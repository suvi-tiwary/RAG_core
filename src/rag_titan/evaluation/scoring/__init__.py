"""Scoring: turn stage metrics into 0 to 100 scores, ratings and quality gates."""

from rag_titan.evaluation.scoring.normalize import normalize
from rag_titan.evaluation.scoring.policy import rating
from rag_titan.evaluation.scoring.scorecard import ScoreAggregator

__all__ = [
    "ScoreAggregator",
    "normalize",
    "rating",
]
