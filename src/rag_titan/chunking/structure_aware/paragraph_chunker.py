"""Chunker that packs whole paragraphs (handbook section 3.2)."""

from rag_titan.chunking.structure_aware.base import PieceChunker
from rag_titan.chunking.structure_aware.sentences import split_paragraphs, split_sentences


class ParagraphChunker(PieceChunker):
    """
    Packs whole paragraphs (separated by a blank line) into chunks of at most
    `chunk_size`. A paragraph that is too long is cut into sentences, and a
    sentence that is still too long is cut at words.

    Args:
        chunk_size: largest chunk, in characters (or tokens with a tokenizer).
        overlap: whole paragraphs repeated from the previous chunk, up to this size.
        tokenizer: optional, counts tokens instead of characters.
    """

    def __init__(self, chunk_size=1000, overlap=0, tokenizer=None):
        super().__init__(chunk_size, overlap, tokenizer)

    def pieces(self, text):
        """Paragraphs, with too-long paragraphs cut into sentences (then words)."""
        return self.fit(split_paragraphs(text), self.sentence_pieces)

    def sentence_pieces(self, paragraph):
        """Sentences of one paragraph, with too-long sentences cut at words."""
        return self.fit(split_sentences(paragraph), self.word_splitter.split_pieces)
