"""
Shared base for chunkers that keep whole units (sentences, paragraphs, ...).

A subclass only says how to cut text into pieces (pieces()). This class
measures them, packs them up to chunk_size with whole-piece overlap, and
trims spaces at the edges of each chunk.
"""

from rag_titan.chunking.base_chunker import BaseChunker
from rag_titan.chunking.fixed_size.fixed_windows import check_window_size
from rag_titan.chunking.fixed_size.recursive_chunker import RecursiveChunker
from rag_titan.chunking.offsets import group_spans, located_spans
from rag_titan.chunking.packing import join_groups, pack


class PieceChunker(BaseChunker):
    """
    Args:
        chunk_size: largest chunk, in characters (or tokens with a tokenizer).
        overlap: how much of the end of a chunk is repeated at the start of the
            next one. Only whole pieces are repeated, so it may be less.
        tokenizer: optional, any object with encode(text) -> ids and
            decode(ids) -> text. When given, sizes count tokens.
    """

    def __init__(self, chunk_size, overlap, tokenizer=None):
        check_window_size(chunk_size, overlap)

        has_methods = hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode")
        if tokenizer is not None and not has_methods:
            raise ValueError("tokenizer must have encode(text) and decode(ids) methods")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tokenizer
        # last resort for a piece that is too big: words, then a hard cut
        self.word_splitter = RecursiveChunker(chunk_size, 0, (" ", ""), tokenizer)

    def length(self, text):
        """Size of `text` in characters, or in tokens when a tokenizer is set."""
        if self.tokenizer is None:
            return len(text)
        return len(self.tokenizer.encode(text))

    def fit(self, pieces, smaller):
        """Keep pieces that fit; cut the others with `smaller(piece)`."""
        result = []
        for piece in pieces:
            too_big = self.length(piece) > self.chunk_size
            result.extend(smaller(piece) if too_big else [piece])
        return result

    def pieces(self, text):
        """Return the pieces of `text`, each at most chunk_size; "".join() gives `text`."""
        raise NotImplementedError

    def split_text(self, text):
        """Return chunks of whole pieces, each at most chunk_size."""
        return [chunk for chunk, _, _ in self.split_spans(text)]

    def split_spans(self, text):
        """(chunk, start, end) for every chunk; exact, because the pieces join back into text."""
        # blank text has nothing worth indexing, so it gives no chunks
        if not text.strip():
            return []

        pieces = self.pieces(text)
        groups = pack([self.length(piece) for piece in pieces], self.chunk_size, self.overlap)
        spans = group_spans(text, pieces, groups)
        if spans is None:   # a tokenizer changed the text, so search for the chunks instead
            chunks = [chunk.strip() for chunk in join_groups(pieces, groups) if chunk.strip()]
            return located_spans(text, chunks)
        return spans
