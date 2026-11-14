"""Chunker that keeps pages apart (handbook section 3.4)."""

from rag_titan.chunking.offsets import shift
from rag_titan.chunking.structure_aware.base import PieceChunker
from rag_titan.chunking.structure_aware.paragraph_chunker import ParagraphChunker

# the PDF loader puts this character between pages
PAGE_BREAK = "\f"


class PageChunker(PieceChunker):
    """
    Chunks each page on its own, so a chunk never spans two pages; good for
    citing page numbers. A page that fits is one chunk; a bigger page is cut
    into whole paragraphs (then sentences, then words). Empty pages are skipped.
    Text without page breaks (Word, Markdown) is treated as one page.

    Args:
        chunk_size: largest chunk, in characters (or tokens with a tokenizer).
        overlap: whole paragraphs repeated inside a page, up to this size.
        tokenizer: optional, counts tokens instead of characters.
    """

    def __init__(self, chunk_size=2000, overlap=0, tokenizer=None):
        super().__init__(chunk_size, overlap, tokenizer)
        self.paragraphs = ParagraphChunker(chunk_size, overlap, tokenizer)

    def split_text(self, text):
        """Return chunks that each stay inside one page."""
        return [chunk for chunk, _, _ in self.split_spans(text)]

    def split_spans(self, text):
        """(chunk, start, end) for every chunk; positions count from the start of `text`."""
        spans, page_start = [], 0
        for page in text.split(PAGE_BREAK):
            spans.extend(shift(self.paragraphs.split_spans(page), page_start))
            page_start += len(page) + len(PAGE_BREAK)
        return spans
