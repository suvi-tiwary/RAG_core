"""Reranker that changes nothing, useful as a baseline."""

from rag_titan.retrieval.rerankers.base_reranker import BaseReranker


class NoReranker(BaseReranker):
    """Keeps the retriever's order. Useful to compare against real rerankers."""

    def score(self, query, match):
        """Return the retriever's own score unchanged."""
        return match["score"]
