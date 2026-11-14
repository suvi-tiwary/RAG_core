"""
Dense retrieval: turn the question into a vector
and find the most similar chunks in the vector store.
"""

from rag_titan.retrieval.base_retriever import BaseRetriever
from rag_titan.retrieval.dedupe import remove_duplicates
from rag_titan.utils.logger import get_logger

logger = get_logger(__name__)


class DenseRetriever(BaseRetriever):
    """
    Finds chunks whose vectors are similar to the question's vector.

    Args:
        embedder: a BaseEmbedder (must be the same one used for indexing).
        store: a BaseStore that holds the indexed chunks.
        dedupe_overlap: drop a result that overlaps a better result from the
            same file by at least this share of the shorter chunk (0.5 = half).
            None keeps every result, like before 0.5.0.
    """

    name = "dense"

    def __init__(self, embedder, store, dedupe_overlap=0.5):
        if dedupe_overlap is not None and not 0 < dedupe_overlap <= 1:
            raise ValueError("dedupe_overlap must be more than 0 and at most 1, or None")

        self.embedder = embedder
        self.store = store
        self.dedupe_overlap = dedupe_overlap

    def has_documents(self):
        """Return True when the store holds at least one chunk."""
        return self.store.count() > 0

    def retrieve(self, query, top_k=5):
        """
        Return the top_k most similar chunks, best first.

        Each result is a chunk dict with "score", "retriever" and "score_trace"
        keys (see BaseRetriever). Near-copies of a better result are removed
        (see dedupe_overlap), so twice as many candidates are fetched to still
        return top_k different chunks.
        """
        self._check_top_k(top_k)

        if not self.has_documents():
            logger.warning("Vector store is empty, nothing to retrieve")
            return []

        query_vector = self.embedder.embed(query)
        if self.dedupe_overlap is None:
            matches = self.store.search(query_vector, top_k=top_k)
        else:
            candidates = self.store.search(query_vector, top_k=2 * top_k)
            different = remove_duplicates(candidates, self.dedupe_overlap)
            logger.debug("Removed %d overlapping results", len(candidates) - len(different))
            matches = different[:top_k]

        for match in matches:
            logger.debug("Retrieved %s (score %.4f)", match["id"], match["score"])

        return self._with_trace(matches)
