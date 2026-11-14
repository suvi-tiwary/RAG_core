"""
Record real embedding vectors once, then replay them without the model.

    recorder = RecordingEmbedder(OllamaEmbedder("nomic-embed-text"))
    ...index files and ask questions with recorder...
    vectors = recorder.vectors                 # save as JSON

    embedder = ReplayEmbedder(vectors)         # later: same vectors, no model

The browser playground uses this, so its examples search by meaning.
"""

import hashlib

import numpy as np

from rag_titan.embeddings.base_embedder import BaseEmbedder


def text_key(text):
    """Return a short id for a text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


class RecordingEmbedder(BaseEmbedder):
    """
    Passes every text to a real embedder and remembers its vector.

    Args:
        embedder: any BaseEmbedder, for example an OllamaEmbedder.
        decimals: digits kept per number in `vectors` (smaller files).
    """

    def __init__(self, embedder, decimals=5):
        super().__init__(embedder.model)
        self.embedder = embedder
        self.decimals = decimals
        self.vectors = {}

    def embed_many(self, texts):
        """Embed new texts with the real model, save every vector, and return the saved ones."""
        new = [text for text in texts if text_key(text) not in self.vectors]
        if new:
            for text, vector in zip(new, self.embedder.embed_many(new)):
                self.vectors[text_key(text)] = [round(float(x), self.decimals) for x in vector]

        # return the rounded vectors, so a replay later gives exactly the same scores
        return [np.array(self.vectors[text_key(text)], dtype=np.float32) for text in texts]


class ReplayEmbedder(BaseEmbedder):
    """
    Returns recorded vectors.

    Args:
        vectors: dict of {text key: list of numbers}, e.g. RecordingEmbedder.vectors.
        model: name to show, e.g. the model the vectors came from.
    """

    def __init__(self, vectors, model="replay"):
        if not isinstance(vectors, dict):
            raise ValueError("vectors must be a dict of {text key: list of numbers}")

        super().__init__(model)
        self.vectors = vectors

    def embed_many(self, texts):
        """Return the recorded vector for every text, in the same order."""
        missing = [text for text in texts if text_key(text) not in self.vectors]
        if missing:
            raise ValueError(
                f"No recorded vector for {len(missing)} text(s), e.g. {missing[0][:60]!r}. "
                "The documents, chunker settings or question differ from the recording."
            )

        return [np.array(self.vectors[text_key(text)], dtype=np.float32) for text in texts]
