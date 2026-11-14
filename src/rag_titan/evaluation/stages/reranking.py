"""
Reranking evaluation: did the reranker put the relevant chunks higher?

    scores = RerankingEvaluator().evaluate(
        queries=[{"id": "q1", "expected_chunk_ids": ["c2"],
                  "retrieved_ids": ["c1", "c3", "c2"],    # before reranking
                  "reranked_ids": ["c2", "c1", "c3"]}],   # after reranking
        k=3,
    )
    print(scores["summary"]["mrr_gain"])   # 0.667: first relevant chunk moved from rank 3 to 1

A gain is "after - before": above 0 means the reranker helped, below 0 means it hurt.
"""

from rag_titan.evaluation.stages.queries import check_k, check_queries, ranked_ids, relevance_of
from rag_titan.evaluation.stages.ranking import (
    RANKING_KEYS,
    difference,
    first_relevant_rank,
    ranking_scores,
)
from rag_titan.evaluation.stages.text import average

# summary key -> per-query key
SUMMARY_KEYS = {
    "ndcg_at_k": "after_ndcg",
    "ndcg_gain": "ndcg_gain",
    "mrr_at_k": "after_mrr",
    "mrr_gain": "mrr_gain",
    "precision_at_k": "after_precision",
    "precision_gain": "precision_gain",
    "first_relevant_rank_change": "first_relevant_rank_change",
    "candidate_set_preservation_rate": "candidate_set_preserved",
}


class RerankingEvaluator:
    """Compares the order before ("retrieved_ids") and after ("reranked_ids") reranking."""

    def evaluate(self, queries, k=5):
        """
        Score every query that has "reranked_ids".

        Args:
            queries: list of dicts with "id", "reranked_ids" (best first), the
                gold labels ("expected_chunk_ids" or graded "relevance") and,
                to measure the gains, "retrieved_ids".
            k: how many results to score per query.

        Returns:
            {"summary": the mean of each metric, "per_query": one dict per query}.
            Gains are None for a query without "retrieved_ids".
        """
        check_k(k)
        check_queries(queries)
        per_query = [self._score_query(query, k) for query in queries if "reranked_ids" in query]

        summary = {"query_count": len(per_query)}
        for name, key in SUMMARY_KEYS.items():
            summary[name] = average([scores[key] for scores in per_query])
        return {"summary": summary, "per_query": per_query}

    def _score_query(self, query, k):
        """Score one query before and after reranking."""
        relevance = relevance_of(query)
        reranked = ranked_ids(query, "reranked_ids")
        after = ranking_scores(reranked, relevance, k)

        retrieved = ranked_ids(query, "retrieved_ids") if "retrieved_ids" in query else None
        before = {key: None for key in RANKING_KEYS}
        if retrieved is not None:
            before = ranking_scores(retrieved, relevance, k)

        scores = {"query_id": query["id"]}
        for name in ("ndcg", "mrr", "precision"):
            key = f"{name}_at_k"
            scores[f"before_{name}"] = before[key]
            scores[f"after_{name}"] = after[key]
            scores[f"{name}_gain"] = difference(after[key], before[key])

        scores["first_relevant_rank_change"] = self._rank_change(retrieved, reranked, relevance)
        scores["candidate_set_preserved"] = self._preserved(retrieved, reranked)
        return scores

    def _rank_change(self, retrieved, reranked, relevance):
        """
        How many places the first relevant chunk moved up (whole lists, not only top k).
        +2 means it went from rank 3 to rank 1; a negative number means it moved down.
        """
        if retrieved is None:
            return None
        before = first_relevant_rank(retrieved, relevance)
        after = first_relevant_rank(reranked, relevance)
        return difference(before, after)

    def _preserved(self, retrieved, reranked):
        """
        1.0 when the reranker returned exactly the chunks it was given (only reordered).
        A reranker that also cuts the list gives 0.0, which can be what you want.
        """
        if retrieved is None:
            return None
        return float(set(retrieved) == set(reranked))
