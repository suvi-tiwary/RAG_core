"""Vector store that keeps everything in memory (lost when Python exits)."""

import numpy as np

from rag_titan.vector_stores.base_store import BaseStore


class MemoryStore(BaseStore):
    """Keeps everything in a Python list. Good for learning and small data."""

    def __init__(self):
        self.records = []

    def add(self, chunks, vectors):
        """Save each chunk with its vector."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        for chunk, vector in zip(chunks, vectors):
            self.records.append({**chunk, "vector": vector})

    def search(self, query_vector, top_k=5):
        """Compare the query with every chunk and return the best top_k."""
        matches = []

        for record in self.records:
            # copy the chunk without its vector, then add the similarity score
            match = {key: value for key, value in record.items() if key != "vector"}
            match["score"] = self._cosine_similarity(query_vector, record["vector"])
            matches.append(match)

        matches.sort(key=lambda match: match["score"], reverse=True)
        return matches[:top_k]

    def clear(self):
        """Remove every stored chunk."""
        self.records = []

    def count(self):
        """Return how many chunks are stored."""
        return len(self.records)

    def _cosine_similarity(self, vector1, vector2):
        """Return a number from -1 to 1. Higher means more similar."""
        denominator = np.linalg.norm(vector1) * np.linalg.norm(vector2)

        if denominator == 0:
            return 0.0

        return float(np.dot(vector1, vector2) / denominator)
