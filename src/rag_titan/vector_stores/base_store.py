"""
Base class for every vector store (in-memory, Chroma, Qdrant, pgvector, ...).

To add one, inherit from BaseStore and implement all four methods.
"""


class BaseStore:
    """Saves chunks with their vectors and finds the most similar ones."""

    def add(self, chunks, vectors):
        """Save chunks together with their vectors (same order)."""
        raise NotImplementedError

    def search(self, query_vector, top_k=5):
        """
        Return the top_k most similar chunks, best first.
        Each result is the chunk dict plus a "score" key.
        """
        raise NotImplementedError

    def clear(self):
        """Remove everything from the store."""
        raise NotImplementedError

    def count(self):
        """Return how many chunks are stored."""
        raise NotImplementedError
