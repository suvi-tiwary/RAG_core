"""
Failure summary: which queries went wrong, and at which stage?

    for item in failed_queries(report):
        print(item["query_id"], item["stage"], item["reason"])

A query can appear more than once: a miss in retrieval usually also gives a
wrong answer. The first entry of a query is the earliest stage that failed.
"""

# (stage, per-query key, value that counts as failed, reason), in pipeline order
RULES = [
    ("retrieval", "hit_at_k", 0, "no relevant chunk in the top K"),
    ("augmentation", "context_sufficient", 0, "required evidence is missing from the context"),
    ("generation", "abstention_accuracy", 0, "answered when it should refuse, or the reverse"),
    ("generation", "task_success", 0, "the task was marked unsuccessful"),
]

# an answer F1 below this counts as a wrong answer
LOW_ANSWER_F1 = 0.5


def failed_queries(report):
    """
    Return [{"query_id", "stage", "reason"}] for the queries that failed a check.

    Args:
        report: a report from RAGEvaluator.
    """
    found = []
    for stage, key, bad_value, reason in RULES:
        for scores in _per_query(report, stage):
            if scores.get(key) == bad_value:
                found.append({"query_id": scores["query_id"], "stage": stage, "reason": reason})

    for scores in _per_query(report, "generation"):
        answer_f1 = scores.get("answer_f1")
        if answer_f1 is not None and answer_f1 < LOW_ANSWER_F1:
            found.append({"query_id": scores["query_id"], "stage": "generation",
                          "reason": f"answer differs from the reference (F1 {answer_f1:.2f})"})

    order = [stage for stage, _, _, _ in RULES]
    return sorted(found, key=lambda item: order.index(item["stage"]))


def _per_query(report, stage):
    """Return the per-query scores of a stage, or [] when it was not evaluated."""
    result = report.get(stage)
    if not isinstance(result, dict):
        return []
    return result.get("per_query") or []
