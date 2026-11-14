"""Chunker that splits text by counting characters."""

from rag_titan.chunking.base_chunker import BaseChunker
from rag_titan.chunking.fixed_size.fixed_windows import check_window_size, fixed_windows


class CharacterChunker(BaseChunker):
    """
    Splits text into fixed-size character windows that overlap.
    Example: size 1000, overlap 200 -> each new chunk starts 800 characters later.

    The text is kept exactly as it is (spaces and newlines included), so a
    chunk may start or end in the middle of a word.

    Args:
        chunk_size: number of characters in each chunk.
        overlap: number of characters shared by two neighbouring chunks.
    """

    def __init__(self, chunk_size=1000, overlap=200):
        check_window_size(chunk_size, overlap)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text):
        """Return overlapping pieces of `chunk_size` characters each."""
        return [chunk for chunk, _, _ in self.split_spans(text)]

    def split_spans(self, text):
        """(chunk, start, end) for each window; window k starts at k * (chunk_size - overlap)."""
        # blank text has nothing worth indexing, so it gives no chunks
        if not text.strip():
            return []

        windows = fixed_windows(range(len(text)), self.chunk_size, self.overlap)
        return [(text[w[0]:w[-1] + 1], w[0], w[-1] + 1) for w in windows]
