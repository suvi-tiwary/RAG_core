"""
The scoring policy: which metrics count, in which direction, and how much.

These weights are sensible defaults, not an industry standard. Change them
through ScoreAggregator(stage_weights=..., metric_weights=..., quality_gates=...).

Each metric rule is (direction, weight). The direction says how a raw value
becomes a 0 to 1 score (see normalize.py):
    "high" / "low"            a rate from 0 to 1 where higher / lower is better
    "ratio"                   a penalty of 0 or more, smaller is better
    "fragmentation"           chunks per gold span, 1 is best
    "cos_high" / "cos_low"    a cosine from -1 to 1
    "cos_distance"            1 - cosine, from 0 to 2, higher is better
    "margin"                  positive minus negative cosine, 0 is neutral
    "gain"                    after - before, from -1 to 1, 0 is neutral
"""

# how much each stage counts in the overall score
STAGE_WEIGHTS = {"chunking": 10, "embedding": 10, "retrieval": 20, "reranking": 10,
                 "augmentation": 15, "generation": 20, "end_to_end": 15}

# stage -> (name shown in reports, what the score means)
STAGE_NAMES = {
    "chunking": ("Document preparation",
                 "Whether documents were split without losing important facts."),
    "embedding": ("Semantic indexing",
                  "Whether content is represented and indexed so it can be found."),
    "retrieval": ("Information search",
                  "How reliably the search finds the required information."),
    "reranking": ("Search ordering", "Whether the important results are near the top."),
    "augmentation": ("Answer context",
                     "Whether the LLM receives complete, relevant supporting material."),
    "generation": ("Answer quality", "Answers compared with validated references and labels."),
    "end_to_end": ("User experience", "Overall correctness, task success and performance."),
}

METRICS = {
    "chunking": {
        "coverage": ("high", 4), "size_compliance": ("high", 1),
        "overlap_ratio": ("ratio", 0.5), "size_cv": ("ratio", 0.2),
        "evidence_completeness": ("high", 4), "evidence_fragmentation": ("fragmentation", 1),
        "structural_integrity": ("high", 2), "metadata_completeness": ("high", 1),
        "metadata_accuracy": ("high", 1), "boundary_f1": ("high", 1.5),
        "lexical_cohesion": ("high", 0.3), "lexical_boundary_clarity": ("high", 0.3),
        "lexical_chunk_stickiness": ("low", 0.2), "semantic_cohesion": ("cos_high", 1),
        "semantic_boundary_clarity": ("cos_distance", 0.5),
        "semantic_chunk_stickiness": ("cos_low", 0.3),
    },
    "embedding": {
        "zero_norm_rate": ("low", 2), "duplicate_vector_rate": ("low", 1),
        "index_coverage": ("high", 3), "query_topk_recall": ("high", 4),
        "ann_recall_at_k": ("high", 2), "positive_similarity": ("cos_high", 0.3),
        "negative_similarity": ("cos_low", 0.3), "similarity_margin": ("margin", 1.5),
    },
    "retrieval": {
        "precision_at_k": ("high", 2), "recall_at_k": ("high", 4), "f1_at_k": ("high", 1),
        "hit_at_k": ("high", 1), "mrr_at_k": ("high", 1.5), "ndcg_at_k": ("high", 2),
        "average_precision_at_k": ("high", 1), "token_recall": ("high", 4),
        "token_precision": ("high", 1.5), "token_iou": ("high", 2),
        "complete_evidence_recall": ("high", 4),
    },
    "reranking": {
        "ndcg_at_k": ("high", 4), "ndcg_gain": ("gain", 1.5), "mrr_at_k": ("high", 3),
        "mrr_gain": ("gain", 1), "precision_gain": ("gain", 0.5),
        "candidate_set_preservation_rate": ("high", 2),
    },
    "augmentation": {
        "context_precision": ("high", 1.5), "context_recall": ("high", 3),
        "context_noise": ("low", 0.5), "context_sufficient": ("high", 5),
        "duplicate_ratio": ("low", 0.5), "budget_compliant": ("high", 2),
        "evidence_edge_placement": ("high", 0.5),
    },
    "generation": {
        "exact_match": ("high", 1), "answer_precision": ("high", 0.5),
        "answer_recall": ("high", 1), "answer_f1": ("high", 4), "rouge_l_f1": ("high", 2),
        "required_fact_coverage": ("high", 3), "citation_precision": ("high", 2),
        "citation_recall": ("high", 1), "citation_f1": ("high", 1),
        "annotated_claim_support": ("high", 5), "abstention_accuracy": ("high", 2),
        "context_grounding": ("high", 2), "unsupported_number_rate": ("low", 2),
        "task_success": ("high", 3),
    },
    "end_to_end": {
        "answer_exact_match": ("high", 1), "answer_f1": ("high", 4),
        "task_success_rate": ("high", 5),
    },
}

# Sizes, latency and cost have no universal "good" value: they are scored only
# when you give a target, e.g. targets={"end_to_end": {"p95_latency_ms": 2000}}.
TARGET_METRICS = {
    "chunking": {"mean_chunk_tokens": 0.5, "min_chunk_tokens": 0.2, "max_chunk_tokens": 0.2},
    "embedding": {"mean_vector_norm": 0.5},
    "retrieval": {"p95_latency_ms": 1},
    "generation": {"p95_latency_ms": 1},
    "end_to_end": {"p50_latency_ms": 1, "p95_latency_ms": 2, "mean_cost_usd": 1,
                   "mean_total_tokens": 0.5},
}

# A metric below its minimum makes the overall status "Action Required",
# however high the score is. Illustrative minimums: set your own.
QUALITY_GATES = {
    "chunking": {"coverage": 0.95, "evidence_completeness": 0.80},
    "embedding": {"index_coverage": 0.95},
    "retrieval": {"recall_at_k": 0.70},
    "augmentation": {"context_sufficient": 0.80},
    "generation": {"annotated_claim_support": 0.80},
    "end_to_end": {"task_success_rate": 0.75},
}

# (minimum score, rating, meaning), best first
RATINGS = [
    (90, "Excellent", "Performance is strong across the available tests."),
    (80, "Very Good", "The evaluated results are strong, with some room to improve."),
    (70, "Good", "Results look promising, but some areas need improvement."),
    (60, "Average", "Several checks need improvement before wider use."),
    (40, "Needs Improvement", "Important weaknesses were found in the evaluated pipeline."),
    (0, "Poor", "Significant problems were detected in the evaluated pipeline."),
]


def rating(score):
    """Return (rating, meaning) for a 0 to 100 score; None means nothing was measured."""
    if score is None:
        return "Not Evaluated", "Not enough applicable measurements to calculate a score."
    return next((name, meaning) for minimum, name, meaning in RATINGS if score >= minimum)
