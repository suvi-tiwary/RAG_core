"""
Base class for every embedding provider.

To add a provider (OpenAI, HuggingFace, ...), inherit from BaseEmbedder
and implement embed_many(). embed() works automatically.
"""


class BaseEmbedder:
    """
    Turns text into vectors (lists of numbers) that capture meaning.

    Args:
        model: the embedding model name.
    """

    def __init__(self, model):
        self.model = model

    def embed(self, text):
        """Return one vector (numpy array) for one text."""
        return self.embed_many([text])[0]

    def embed_many(self, texts):
        """Return a list of vectors, one per text, in the same order."""
        raise NotImplementedError
