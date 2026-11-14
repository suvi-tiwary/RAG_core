"""Chunker that packs whole sentences (handbook section 3.1)."""

from rag_titan.chunking.structure_aware.base import PieceChunker
from rag_titan.chunking.structure_aware.sentences import split_sentences


class SentenceChunker(PieceChunker):
    """
    Packs whole sentences into chunks of at most `chunk_size`.
    A sentence is only cut (at words) when it is longer than chunk_size.
    Example: size 500, overlap 100 -> chunks end at sentence boundaries and
    repeat the last sentence or two of the previous chunk.

    Args:
        chunk_size: largest chunk, in characters (or tokens with a tokenizer).
        overlap: whole sentences repeated from the previous chunk, up to this size.
        tokenizer: optional, counts tokens instead of characters.
    """

    def __init__(self, chunk_size=500, overlap=100, tokenizer=None):
        super().__init__(chunk_size, overlap, tokenizer)

    def pieces(self, text):
        """Sentences, with too-long sentences cut at words."""
        return self.fit(split_sentences(text), self.word_splitter.split_pieces)
