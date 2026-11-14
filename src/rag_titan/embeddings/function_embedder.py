"""
Use any Python function as the embedding model.

    def my_embed(texts):
        return call_my_api(texts)             # one list of numbers per text

    embedder = FunctionEmbedder(my_embed, model="my-api")

The browser playground uses it to embed with your local Ollama.
"""

import numpy as np

from rag_titan.embeddings.base_embedder import BaseEmbedder


class FunctionEmbedder(BaseEmbedder):
    """
    An embedder made from a function.

    Args:
        embed: function that takes a list of texts and returns one vector
            (list of numbers) per text, in the same order.
        model: name to show for this model.
    """

    def __init__(self, embed, model="function"):
        if not callable(embed):
            raise ValueError("embed must be a function: texts -> list of vectors")

        super().__init__(model)
        self.embed_function = embed

    def embed_many(self, texts):
        """Call the function and check it gave one vector per text."""
        if not texts:
            return []

        vectors = self.embed_function(list(texts))
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ValueError(f"The embed function must return {len(texts)} vectors, "
                             f"got {len(vectors) if isinstance(vectors, list) else vectors!r}")

        return [np.array(vector, dtype=np.float32) for vector in vectors]
