"""Chunker that splits text by counting words."""

import re

from rag_titan.chunking.base_chunker import BaseChunker
from rag_titan.chunking.fixed_size.fixed_windows import fixed_windows


class WordChunker(BaseChunker):
    """
    Splits text into fixed-size word windows that overlap.
    Example: size 150, overlap 30 -> each new chunk starts 120 words later.

    Args:
        chunk_size: number of words in each chunk.
        overlap: number of words shared by two neighbouring chunks.
    """

    def __init__(self, chunk_size=150, overlap=30):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be between 0 and chunk_size - 1")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text):
        """Return overlapping pieces of `chunk_size` words each."""
        return [chunk for chunk, _, _ in self.split_spans(text)]

    def split_spans(self, text):
        """(chunk, start, end): from the first word's start to the last word's end."""
        words = list(re.finditer(r"\S+", text))   # the same words as text.split()
        windows = fixed_windows(words, self.chunk_size, self.overlap)
        return [(" ".join(word.group() for word in window), window[0].start(), window[-1].end())
                for window in windows]
