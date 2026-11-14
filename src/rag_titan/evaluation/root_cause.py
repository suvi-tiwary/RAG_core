"""
Root-cause hints: which stage of the pipeline is the first to fall below its minimum?

    causes = RootCauseAnalyzer().evaluate(report)
    print(causes["first_failing_stage"])   # e.g. "retrieval"

The checks run in pipeline order, because a failure early on (a fact cut in
two by the chunker) explains failures later (a wrong answer). These are simple
rules that point you where to look first; they do not prove the cause.
"""

from rag_titan.evaluation.stages.text import is_number

# (stage, metric, minimum, what to try), in pipeline order
CHECKS = [
    ("chunking", "coverage", 0.98,
     "Restore content lost while parsing or chunking."),
    ("chunking", "evidence_completeness", 0.9,
     "Move chunk boundaries so gold evidence stays in one chunk (bigger chunks or overlap)."),
    ("embedding", "index_coverage", 0.98,
     "Retry failed index writes and compare the stored ids with the chunk ids."),
    ("embedding", "query_topk_recall", 0.8,
     "Check the embedding model and that queries and chunks are embedded the same way."),
    ("retrieval", "recall_at_k", 0.8,
     "Try hybrid retrieval or query rewriting, or raise top-k."),
    ("reranking", "ndcg_gain", -0.03,
     "The reranker makes the order worse: check its model and the relevance labels."),
    ("augmentation", "context_sufficient", 0.9,
     "Pack all the supporting evidence into the context, within the budget."),
    ("generation", "answer_f1", 0.75,
     "Compare the answers with the references: check the prompt and the LLM."),
]

METHOD = "rule-based hints from thresholds, in pipeline order; not a proof of the cause"


class RootCauseAnalyzer:
    """Finds the metrics below their minimum and the first stage that fails."""

    def __init__(self, thresholds=None):
        """
        Args:
            thresholds: {metric: minimum} to change the defaults, e.g.
                {"recall_at_k": 0.9, "answer_f1": 0.6}.
        """
        self.thresholds = {metric: minimum for _, metric, minimum, _ in CHECKS}

        for metric, minimum in (thresholds or {}).items():
            if metric not in self.thresholds:
                raise ValueError(f"Unknown threshold {metric!r}: use one of "
                                 f"{sorted(self.thresholds)}")
            if not is_number(minimum):
                raise ValueError(f"Threshold of {metric!r} must be a number, got {minimum!r}")
            self.thresholds[metric] = minimum

    def evaluate(self, report):
        """
        Check a report from RAGEvaluator (or a dict of {stage: stage result}).

        Returns:
            A dict with "first_failing_stage" (a stage name or None), "failures"
            (a list of {"stage", "metric", "observed", "minimum", "suggestion"})
            and "method" (how the hints were made). Metrics that were not
            measured are skipped, not failed.
        """
        if not isinstance(report, dict):
            raise ValueError("report must be a dict of {stage: stage result}")

        failures = []
        for stage, metric, _, suggestion in CHECKS:
            result = report.get(stage) or {}
            observed = result.get("summary", result).get(metric)
            minimum = self.thresholds[metric]
            if observed is not None and observed < minimum:
                failures.append({"stage": stage, "metric": metric, "observed": observed,
                                 "minimum": minimum, "suggestion": suggestion})

        return {"first_failing_stage": failures[0]["stage"] if failures else None,
                "failures": failures, "method": METHOD}
