"""
Embedder that needs no model: every word is hashed to a position in a vector.

Texts that share words get similar vectors, so search works by **words**, not by
meaning ("holiday" does not find "leave"). Use it to try rag-titan offline, in
tests, or in the browser playground; use a real embedding model for real work.
"""

import hashlib
import re

import numpy as np

from rag_titan.embeddings.base_embedder import BaseEmbedder

WORD = re.compile(r"\w+")


class HashEmbedder(BaseEmbedder):
    """
    Turns text into word-count vectors of a fixed size (the "hashing trick").

    Args:
        dimensions: length of every vector; more means fewer words share a position.
    """

    def __init__(self, dimensions=512):
        if not isinstance(dimensions, int) or dimensions < 8:
            raise ValueError("dimensions must be a whole number of at least 8")

        super().__init__("hash")
        self.dimensions = dimensions

    def embed_many(self, texts):
        """Return one vector per text. The same text always gives the same vector."""
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text):
        """Add +1 or -1 at the hashed position of every word."""
        vector = np.zeros(self.dimensions, dtype=np.float32)

        for word in WORD.findall(text.lower()):
            # md5 instead of hash(): Python's hash() changes between runs
            number = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            sign = 1.0 if number & 1 else -1.0   # signs make collisions cancel out
            vector[(number >> 1) % self.dimensions] += sign

        return vector
