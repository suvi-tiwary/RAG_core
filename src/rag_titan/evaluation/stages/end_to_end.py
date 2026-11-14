"""
End-to-end evaluation: what the user experienced, from request to answer.

    scores = EndToEndEvaluator().evaluate([
        {"id": "q1", "answer": "25 days", "expected_answer": "25 days",
         "task_success": True, "latency_ms": 840, "cost_usd": 0.002, "total_tokens": 950},
    ])
    print(scores["answer_exact_match"], scores["p95_latency_ms"])   # 1.0 840

"latency_ms" is the whole request (search + reranking + LLM), as the user waits for it.
"""

from rag_titan.evaluation.stages.answer_metrics import exact_match, token_scores
from rag_titan.evaluation.stages.queries import check_queries, reference_answer
from rag_titan.evaluation.stages.text import average, is_number, percentile


class EndToEndEvaluator:
    """Scores the final answers, task outcomes and request telemetry."""

    def evaluate(self, queries):
        """
        Score every query that has an "answer".

        Args:
            queries: list of dicts with "id", "answer" and any of
                "expected_answer", "task_success", "latency_ms", "cost_usd",
                "total_tokens".

        Returns:
            A dict of metric name -> value (None = not measured):

                End-to-End P95 Latency = P95(t_request to answer)
                Task Success Rate      = #successful tasks / #labelled tasks
        """
        check_queries(queries)
        observed = [query for query in queries if "answer" in query]
        # questions the documents cannot answer have no reference answer
        with_reference = [dict(query, expected_answer=reference_answer(query))
                          for query in observed if reference_answer(query) is not None]
        outcomes = [query["task_success"] for query in observed
                    if query.get("task_success") is not None]
        times = self._numbers(observed, "latency_ms")

        return {
            "query_count": len(observed),
            "answer_exact_match": average([exact_match(query["answer"], query["expected_answer"])
                                           for query in with_reference]),
            "answer_f1": average([token_scores(query["answer"],
                                               query["expected_answer"])["answer_f1"]
                                  for query in with_reference]),
            "task_success_rate": average([float(bool(outcome)) for outcome in outcomes]),
            "p50_latency_ms": percentile(times, 50),
            "p95_latency_ms": percentile(times, 95),
            "mean_cost_usd": average(self._numbers(observed, "cost_usd")),
            "mean_total_tokens": average(self._numbers(observed, "total_tokens")),
        }

    def _numbers(self, queries, key):
        """Return the measured values of one telemetry key, checking they are valid."""
        values = []
        for query in queries:
            value = query.get(key)
            if value is None:
                continue
            if not is_number(value) or value < 0:
                raise ValueError(f"Query {query['id']!r}: {key!r} must be a number of 0 or more")
            values.append(value)
        return values
