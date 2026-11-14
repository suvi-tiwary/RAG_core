"""
Augmentation evaluation: is the context given to the LLM complete, relevant and not wasteful?

    scores = AugmentationEvaluator().evaluate(
        chunks=["Leave is 25 days. ", "Sick pay is full."],
        queries=[{"id": "q1", "context_ids": ["chunk_1"], "expected_evidence": ["25 days"]}],
        document_text="Leave is 25 days. Sick pay is full.",
    )
    print(scores["summary"]["context_sufficient"])   # 1.0

"context_ids" are the chunks that really went into the prompt, in prompt order.
"""

from rag_titan.evaluation.stages.coverage import covered_tokens, document_tokens
from rag_titan.evaluation.stages.queries import check_queries, ranked_ids, relevance_of
from rag_titan.evaluation.stages.spans import (
    as_documents,
    chunk_spans,
    evidence_spans,
    is_inside,
    normalize_chunks,
)
from rag_titan.evaluation.stages.text import average, tokenize

SUMMARY_KEYS = ["context_precision", "context_recall", "context_noise", "context_sufficient",
                "duplicate_ratio", "budget_compliant", "evidence_edge_placement"]


class AugmentationEvaluator:
    """Scores the chunks that were packed into the prompt for each query."""

    def evaluate(self, chunks, queries, document_text=None, documents=None,
                 max_context_tokens=None):
        """
        Score every query that has "context_ids".

        Args:
            chunks: all chunks: texts, or dicts with "id", "text", "start", "end", "doc_id".
            queries: list of dicts with "id", "context_ids" and the gold labels:
                "expected_evidence" (needs the source text) and/or
                "expected_chunk_ids". "max_context_tokens" can be set per query.
            document_text: the source text of a single document.
            documents: {doc_id: text} when there are several documents.
            max_context_tokens: the context budget in words, for every query.

        Returns:
            {"summary": the mean of each metric, "per_query": one dict per
            query, "warnings": chunks that could not be placed in the text}.
        """
        check_queries(queries)
        documents = as_documents(document_text, documents)
        items, warnings = normalize_chunks(chunks, documents)
        by_id = {item["id"]: item for item in items}
        tokens = document_tokens(documents)

        per_query = [self._score_query(query, by_id, documents, tokens, max_context_tokens)
                     for query in queries if "context_ids" in query]

        summary = {"query_count": len(per_query)}
        for key in SUMMARY_KEYS:
            summary[key] = average([scores[key] for scores in per_query])
        return {"summary": summary, "per_query": per_query, "warnings": warnings}

    def _score_query(self, query, by_id, documents, tokens, max_context_tokens):
        """Return every context score of one query."""
        selected = ranked_ids(query, "context_ids")
        used = [by_id[chunk_id] for chunk_id in selected if chunk_id in by_id]
        evidence = []
        if documents:
            evidence = evidence_spans(query.get("expected_evidence"), documents)
        budget = query.get("max_context_tokens", max_context_tokens)

        scores = {"query_id": query["id"]}
        scores.update(self._evidence_scores(used, evidence, tokens))
        scores.update(self._budget_scores(used, tokens, documents, budget))
        scores["evidence_edge_placement"] = self._edge_placement(used, evidence, query)
        scores["missing_chunk_ids"] = [cid for cid in selected if cid not in by_id]
        return scores

    def _evidence_scores(self, used, evidence, tokens):
        """
        Compare the context words (C) with the gold evidence words (G).

            Context Precision  = |C ∩ G| / |C|
            Context Recall     = |C ∩ G| / |G|
            Context Noise      = 1 - Context Precision
            Context Sufficient = 1 when every gold span is fully inside a context chunk
        """
        if not evidence:
            return {key: None for key in SUMMARY_KEYS[:4]}

        spans = chunk_spans(used)
        context, gold = covered_tokens(tokens, spans), covered_tokens(tokens, evidence)
        precision = len(context & gold) / len(context) if context else None
        return {
            "context_precision": precision,
            "context_recall": len(context & gold) / len(gold) if gold else None,
            "context_noise": None if precision is None else 1 - precision,
            "context_sufficient": float(all(is_inside(span, spans) for span in evidence)),
        }

    def _budget_scores(self, used, tokens, documents, budget):
        """
        Size of the context in words, repeated words, and the budget check.

            Duplicate Ratio  = (context words - distinct source words) / context words
            Budget Compliant = 1 when context words <= max_context_tokens
        """
        count = sum(len(tokenize(item["text"])) for item in used)
        scores = {"context_tokens": count, "duplicate_ratio": None, "budget_compliant": None}

        placed = documents and all(item["start"] is not None for item in used)
        if count and placed:
            distinct = len(covered_tokens(tokens, chunk_spans(used)))
            scores["duplicate_ratio"] = max(0.0, (count - distinct) / count)
        if budget is not None:
            scores["budget_compliant"] = float(count <= budget)
        return scores

    def _edge_placement(self, used, evidence, query):
        """
        Is the evidence at the start or end of the context (1.0) or in the middle (0.0)?
        LLMs often overlook text in the middle of a long prompt ("lost in the middle").
        """
        gold_ids = {cid for cid, grade in relevance_of(query).items() if grade > 0}
        placements = []
        for number, item in enumerate(used):
            touches = item["start"] is not None and any(
                doc_id == item["doc_id"] and item["start"] < end and item["end"] > start
                for doc_id, start, end in evidence)
            if item["id"] in gold_ids or touches:
                position = number / (len(used) - 1) if len(used) > 1 else 0.0
                placements.append(abs(2 * position - 1))
        return average(placements)
