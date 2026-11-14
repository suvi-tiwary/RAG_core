"""
Generation evaluation: is the answer correct, complete, cited and grounded?

    scores = GenerationEvaluator().evaluate([
        {"id": "q1",
         "answer": "You get 25 days of leave.",            # what your RAG system wrote
         "expected_answer": "Annual leave is 25 days.",    # the validated reference
         "required_facts": ["25 days"],
         "citations": ["c1"], "expected_citations": ["c1"]},
    ])
    print(scores["summary"]["required_fact_coverage"])   # 1.0

No LLM judge is used. Labels that need judgement ("verified_claims",
"task_success", "answerable") come from you; everything else is computed.
"""

from rag_titan.evaluation.stages.answer_metrics import (
    citation_scores,
    context_grounding,
    exact_match,
    required_fact_coverage,
    rouge_l_f1,
    token_scores,
    unsupported_number_rate,
)
from rag_titan.evaluation.stages.queries import check_queries, reference_answer
from rag_titan.evaluation.stages.spans import normalize_chunks
from rag_titan.evaluation.stages.text import average, percentile

SUMMARY_KEYS = [
    "exact_match", "answer_precision", "answer_recall", "answer_f1", "rouge_l_f1",
    "required_fact_coverage", "citation_precision", "citation_recall", "citation_f1",
    "annotated_claim_support", "abstention_accuracy", "context_grounding",
    "unsupported_number_rate", "task_success",
]


class GenerationEvaluator:
    """Scores generated answers against reference answers and your labels."""

    def evaluate(self, queries, chunks=None):
        """
        Score every query that has an "answer".

        Args:
            queries: list of dicts with "id", "answer" and any of:
                "expected_answer", "required_facts" (phrases), "citations" and
                "expected_citations" (source ids), "verified_claims" (list of
                {"supported": True/False}), "answerable" and "abstained",
                "task_success", "generation_latency_ms", and the context as
                "context" (text) or "context_ids". With "answerable": False
                the answer is not compared with a reference, and with
                "abstained": True it is not checked against the context:
                Abstention Accuracy scores those replies.
            chunks: the chunks, needed only to turn "context_ids" into text.

        Returns:
            {"summary": the mean of each metric, "per_query": one dict per query}.
        """
        check_queries(queries)
        items, _ = normalize_chunks(chunks or [], {})
        texts = {item["id"]: item["text"] for item in items}

        per_query = [self._score_query(query, texts) for query in queries if "answer" in query]

        summary = {"query_count": len(per_query)}
        for key in SUMMARY_KEYS:
            summary[key] = average([scores[key] for scores in per_query])
        times = [scores["latency_ms"] for scores in per_query]
        summary["p50_latency_ms"] = percentile(times, 50)
        summary["p95_latency_ms"] = percentile(times, 95)
        return {"summary": summary, "per_query": per_query}

    def _score_query(self, query, texts):
        """Return every generation score of one query."""
        answer = query["answer"]
        if not isinstance(answer, str):
            raise ValueError(f"Query {query['id']!r}: 'answer' must be a text")

        scores = {"query_id": query["id"]}
        scores.update(self._reference_scores(answer, reference_answer(query)))
        scores["required_fact_coverage"] = required_fact_coverage(
            answer, query.get("required_facts"))
        scores.update(citation_scores(query.get("citations"), query.get("expected_citations")))
        scores.update(self._label_scores(query))

        # a refusal ("I don't know") makes no claims, so there is nothing to ground
        refused = query.get("abstained") is True
        context = None if refused else self._context_text(query, texts)
        scores["context_grounding"] = context_grounding(answer, context)
        scores["unsupported_number_rate"] = unsupported_number_rate(answer, context)
        scores["latency_ms"] = query.get("generation_latency_ms")
        return scores

    def _reference_scores(self, answer, expected):
        """Compare the answer with the reference answer (None when there is none)."""
        if expected is None:
            return {key: None for key in ("exact_match", "answer_precision", "answer_recall",
                                          "answer_f1", "rouge_l_f1")}
        scores = {"exact_match": exact_match(answer, expected)}
        scores.update(token_scores(answer, expected))
        scores["rouge_l_f1"] = rouge_l_f1(answer, expected)
        return scores

    def _label_scores(self, query):
        """
        Scores that come from labels you supply.

            annotated_claim_support  claims marked supported / claims
            Abstention Accuracy      1 when "abstained" is right: the system
                                     refused an unanswerable question or
                                     answered an answerable one
        """
        claims = query.get("verified_claims")
        abstained, answerable = query.get("abstained"), query.get("answerable")

        support, abstention = None, None
        if claims:
            support = sum(bool(claim["supported"]) for claim in claims) / len(claims)
        if abstained is not None and answerable is not None:
            abstention = float(bool(abstained) != bool(answerable))

        success = query.get("task_success")
        return {"annotated_claim_support": support, "abstention_accuracy": abstention,
                "task_success": None if success is None else float(bool(success))}

    def _context_text(self, query, texts):
        """Return the context the LLM was given, or None when it is not known."""
        if query.get("context") is not None:
            return query["context"]

        found = [texts[chunk_id] for chunk_id in query.get("context_ids") or []
                 if chunk_id in texts]
        return " ".join(found) if found else None
