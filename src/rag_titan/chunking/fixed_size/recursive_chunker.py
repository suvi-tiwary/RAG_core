"""
Chunker that splits on natural separators first (handbook section 2.5).

It tries the biggest separator first (paragraph, line, sentence, space) and
only falls back to a smaller one, and finally to a hard cut, when a piece is
still too big. Small pieces are then packed together up to chunk_size.
"""

from rag_titan.chunking.base_chunker import BaseChunker
from rag_titan.chunking.fixed_size.fixed_windows import check_window_size, fixed_windows
from rag_titan.chunking.offsets import group_spans, located_spans
from rag_titan.chunking.packing import join_groups, pack

# paragraph, line, sentence, word, then "" = hard cut
SEPARATORS = ("\n\n", "\n", ". ", " ", "")


def split_keep_separator(text, separator):
    """Split `text` on `separator`, keeping it at the end of each part, so no text is lost."""
    parts = text.split(separator)
    pieces = [part + separator for part in parts[:-1]] + [parts[-1]]
    return [piece for piece in pieces if piece]


class RecursiveChunker(BaseChunker):
    """
    Splits text on paragraphs, then lines, sentences and words, packing the
    pieces into chunks of at most `chunk_size` characters (or tokens).
    Example: size 1000, overlap 200 -> paragraphs stay whole when they fit.

    Args:
        chunk_size: largest chunk, in characters (or tokens with a tokenizer).
        overlap: how much of the end of a chunk is repeated at the start of
            the next one. Only whole pieces are repeated, so it may be less.
        separators: separators to try, biggest first. "" means a hard cut.
        tokenizer: optional, any object with encode(text) -> ids and
            decode(ids) -> text. When given, sizes count tokens.
    """

    def __init__(self, chunk_size=1000, overlap=200, separators=SEPARATORS, tokenizer=None):
        check_window_size(chunk_size, overlap)

        if not separators:
            raise ValueError("separators must contain at least one separator")

        has_methods = hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode")
        if tokenizer is not None and not has_methods:
            raise ValueError("tokenizer must have encode(text) and decode(ids) methods")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = tuple(separators)
        self.tokenizer = tokenizer

    def length(self, text):
        """Size of `text` in characters, or in tokens when a tokenizer is set."""
        if self.tokenizer is None:
            return len(text)
        return len(self.tokenizer.encode(text))

    def split_text(self, text):
        """Return chunks of at most `chunk_size`, split on the biggest separators possible."""
        return [chunk for chunk, _, _ in self.split_spans(text)]

    def split_spans(self, text):
        """(chunk, start, end) for every chunk; exact, because the pieces join back into text."""
        # blank text has nothing worth indexing, so it gives no chunks
        if not text.strip():
            return []

        pieces = self._split(text, self.separators)
        groups = pack([self.length(piece) for piece in pieces], self.chunk_size, self.overlap)
        spans = group_spans(text, pieces, groups)
        if spans is None:   # a tokenizer changed the text, so search for the chunks instead
            chunks = [chunk.strip() for chunk in join_groups(pieces, groups) if chunk.strip()]
            return located_spans(text, chunks)
        return spans

    def split_pieces(self, text):
        """
        Cut `text` into pieces that each fit in chunk_size, without packing them.
        Joining the pieces gives back `text` exactly. Other chunkers use this
        as a fallback for pieces that are too big.
        """
        return self._split(text, self.separators)

    def _split(self, text, separators):
        """Break `text` into pieces that each fit in chunk_size."""
        if self.length(text) <= self.chunk_size:
            return [text]

        for index, separator in enumerate(separators):
            if separator == "":
                return self._hard_cut(text)

            # use the biggest separator found; smaller ones handle what is still too big
            if separator in text:
                pieces = []
                for part in split_keep_separator(text, separator):
                    pieces.extend(self._split(part, separators[index + 1:]))
                return pieces

        # no separator matched and no "" fallback was given
        return self._hard_cut(text)

    def _hard_cut(self, text):
        """Last resort: cut `text` into exact chunk_size pieces."""
        if self.tokenizer is None:
            return fixed_windows(text, self.chunk_size)

        ids = list(self.tokenizer.encode(text))
        return [self.tokenizer.decode(window) for window in fixed_windows(ids, self.chunk_size)]
