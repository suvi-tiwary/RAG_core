"""
Embedding model that runs locally with Ollama.

Before using it, pull the model:
    ollama pull nomic-embed-text
"""

import numpy as np

from rag_titan.embeddings.base_embedder import BaseEmbedder
from rag_titan.llm.ollama_http import parse_host, post_json
from rag_titan.utils.logger import get_logger

logger = get_logger(__name__)


class OllamaEmbedder(BaseEmbedder):
    """
    Creates embeddings with an Ollama server.

    Args:
        model: Ollama embedding model name.
        host: Ollama server address. None means http://localhost:11434.
        batch_size: how many texts to send in one request.
    """

    def __init__(self, model="nomic-embed-text", host=None, batch_size=32):
        super().__init__(model)
        self.batch_size = batch_size
        self.host = parse_host(host)

    def embed_many(self, texts):
        """Return one numpy vector per text, in the same order."""
        vectors = []

        # send texts in small batches so large documents do not overload Ollama
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            vectors.extend(self._embed_batch(batch))
            logger.debug("Embedded %d / %d texts", len(vectors), len(texts))

        return vectors

    def _embed_batch(self, batch):
        """Send one batch of texts to Ollama and return their vectors."""
        try:
            reply = post_json(self.host, "/api/embed", {"model": self.model, "input": batch})
        except RuntimeError as error:
            raise RuntimeError(
                f"Ollama embedding failed for model '{self.model}': {error}"
            ) from error

        vectors = reply.get("embeddings") if isinstance(reply, dict) else None
        # one vector per text, or the chunks and vectors would get out of step
        if not isinstance(vectors, list) or len(vectors) != len(batch):
            raise RuntimeError(
                f"Ollama sent {len(vectors or [])} embeddings for {len(batch)} texts "
                f"(model '{self.model}')"
            )

        return [np.array(vector, dtype=np.float32) for vector in vectors]
