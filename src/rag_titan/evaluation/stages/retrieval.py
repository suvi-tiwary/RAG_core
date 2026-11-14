"""
Retrieval evaluation: did the retriever return the chunks and the evidence a query needs?

    scores = RetrievalEvaluator().evaluate(
        chunks=[{"id": "c1", "text": "Leave is 25 days. "},
                {"id": "c2", "text": "Sick pay is full."}],
        queries=[{"id": "q1", "expected_chunk_ids": ["c1"], "retrieved_ids": ["c2", "c1"]}],
        k=2,
    )
    print(scores["summary"]["recall_at_k"])   # 1.0

Add the source text (document_text or documents) and "expected_evidence" on
the queries to also get the token metrics, which do not depend on chunk ids.
"""

from rag_titan.evaluation.stages.coverage import covered_tokens, document_tokens
from rag_titan.evaluation.stages.queries import check_k, check_queries, ranked_ids, relevance_of
from rag_titan.evaluation.stages.ranking import RANKING_KEYS, ranking_scores
from rag_titan.evaluation.stages.spans import (
    as_documents,
    chunk_spans,
    evidence_spans,
    is_inside,
    normalize_chunks,
)
from rag_titan.evaluation.stages.text import average, percentile

TOKEN_KEYS = ["token_recall", "token_precision", "token_iou", "complete_evidence_recall"]


class RetrievalEvaluator:
    """Scores the ranked chunk ids a retriever returned for each query."""

    def evaluate(self, chunks, queries, k=5, document_text=None, documents=None):
        """
        Score every query that has "retrieved_ids".

        Args:
            chunks: all chunks of the index: texts, or dicts with "id", "text"
                and, when you have them, "start", "end" and "doc_id".
            queries: list of dicts with "id", "retrieved_ids" (best first) and
                the gold labels: "expected_chunk_ids" or graded "relevance",
                and optionally "expected_evidence" and "retrieval_latency_ms".
            k: how many results to score per query.
            document_text: the source text of a single document.
            documents: {doc_id: text} when there are several documents.

        Returns:
            {"summary": the mean of each metric, "per_query": one dict per
            query, "warnings": chunks that could not be placed in the text}.
            A metric is None when its inputs were not given.
        """
        check_k(k)
        check_queries(queries)
        documents = as_documents(document_text, documents)
        items, warnings = normalize_chunks(chunks, documents)

        by_id = {item["id"]: item for item in items}
        tokens = document_tokens(documents)
        per_query = [self._score_query(query, by_id, documents, tokens, k)
                     for query in queries if "retrieved_ids" in query]

        return {"summary": self._summary(per_query), "per_query": per_query,
                "warnings": warnings}

    def _score_query(self, query, by_id, documents, tokens, k):
        """Return the ranking and token scores of one query."""
        selected = ranked_ids(query, "retrieved_ids")[:k]
        scores = {"query_id": query["id"]}
        scores.update(ranking_scores(selected, relevance_of(query), k))
        scores.update(self._token_scores(query, selected, by_id, documents, tokens))
        scores["latency_ms"] = query.get("retrieval_latency_ms")
        scores["missing_chunk_ids"] = [cid for cid in selected if cid not in by_id]
        return scores

    def _token_scores(self, query, selected, by_id, documents, tokens):
        """
        Compare the words of the retrieved chunks (R) with the gold evidence words (G).

            Token Recall    = |R ∩ G| / |G|
            Token Precision = |R ∩ G| / |R|
            Token IoU       = |R ∩ G| / |R ∪ G|
            Complete Evidence Recall = #gold spans fully inside one chunk / #gold spans
        """
        evidence = evidence_spans(query.get("expected_evidence"), documents) if documents else []
        if not evidence:
            return {key: None for key in TOKEN_KEYS}

        spans = chunk_spans([by_id[cid] for cid in selected if cid in by_id])
        gold = covered_tokens(tokens, evidence)
        found = covered_tokens(tokens, spans)
        both = gold & found

        return {
            "token_recall": len(both) / len(gold) if gold else None,
            "token_precision": len(both) / len(found) if found else None,
            "token_iou": len(both) / len(gold | found) if gold | found else None,
            "complete_evidence_recall":
                sum(is_inside(span, spans) for span in evidence) / len(evidence),
        }

    def _summary(self, per_query):
        """Average every metric over the queries and add the latency percentiles."""
        summary = {"query_count": len(per_query)}
        for key in RANKING_KEYS + TOKEN_KEYS:
            summary[key] = average([scores[key] for scores in per_query])

        times = [scores["latency_ms"] for scores in per_query]
        summary["p50_latency_ms"] = percentile(times, 50)
        summary["p95_latency_ms"] = percentile(times, 95)
        return summary
