"""
Hints for the terminal summary: which queries to look at, and what was not
measured (with the input that would measure it).

    for line in queries_to_look_at(report):
        print(line)          # "q4 (generation): answer differs from the reference (F1 0.36)"
    print(not_measured(report))   # ["semantic indexing: add embeddings", ...]

A null in a report means "not measured", never zero. These hints say why.
"""

from rag_titan.evaluation.reports.failures import failed_queries

# (stage, a metric that is None when the input is missing, hint), in pipeline order
MISSING_INPUTS = [
    ("embedding", None, "semantic indexing: add \"embeddings\" (chunk and query vectors)"),
    ("reranking", None, "search ordering: add \"reranked_ids\" (only if you have a reranker)"),
    ("augmentation", "context_precision",
     "context and evidence checks: add \"documents\" and \"expected_evidence\" "
     "(build_dataset() does it)"),
    ("generation", "task_success", "task success: add \"task_success\" per query "
                                   "(a person's yes/no)"),
    ("generation", "abstention_accuracy", "refusals: add a question with "
                                          "\"answerable\": false and \"abstained\""),
    ("end_to_end", "p95_latency_ms", "end-to-end time and cost: add \"latency_ms\", "
                                     "\"cost_usd\", \"total_tokens\""),
]


def queries_to_look_at(report, limit=5):
    """
    Return one line per failed query (its earliest failing stage), at most `limit`.

    Args:
        report: a report from RAGEvaluator.
        limit: how many lines to return; a last line counts the rest.
    """
    lines, seen = [], set()
    for item in failed_queries(report):
        if item["query_id"] not in seen:
            seen.add(item["query_id"])
            lines.append(f"{item['query_id']} ({item['stage']}): {item['reason']}")
    if len(lines) > limit:
        lines = lines[:limit] + [f"... and {len(lines) - limit} more (see the .md report)"]
    return lines


def not_measured(report):
    """Return a hint for every part of the report that had no input to measure it."""
    hints = []
    for stage, metric, hint in MISSING_INPUTS:
        result = report.get(stage)
        if not isinstance(result, dict):
            if metric is None:
                hints.append(hint)        # the whole stage had no data
            continue
        summary = result.get("summary", result)
        if metric is not None and metric in summary and summary[metric] is None:
            hints.append(hint)
    return hints
