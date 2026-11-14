"""
Base class for every chunking strategy.

To add a new strategy (sentence, semantic, recursive, ...), inherit from
BaseChunker and implement split_text(). The id/source/offset handling is shared.
"""

from rag_titan.chunking.locations import Locations
from rag_titan.chunking.offsets import located_spans


class BaseChunker:
    """Splits documents into smaller pieces called chunks."""

    def chunk(self, documents):
        """
        Turn a list of document dicts into a list of chunk dicts.

        Args:
            documents: list of {"source": ..., "text": ...} dicts.

        Returns:
            List of {"id": "chunk_0", "source": ..., "text": ...,
            "start_offset": 0, "end_offset": 120} dicts.
            `document["text"][start_offset:end_offset]` is where the chunk comes
            from (the offsets are None if it cannot be found).
            When known, chunks also get "page_numbers" (PDFs) and
            "section_path" (documents with headings).
        """
        chunks = []

        for document in documents:
            locations = Locations(document["text"])
            for text, start, end in self.split_spans(document["text"]):
                chunks.append({
                    "id": f"chunk_{len(chunks)}",
                    "source": document["source"],
                    "text": text,
                    "start_offset": start,
                    "end_offset": end,
                    **locations.describe(start, end),
                })

        return chunks

    def split_spans(self, text):
        """
        Return (chunk_text, start, end) for every chunk of `text`.

        This default searches for each chunk from split_text() in the text.
        Chunkers that know their exact positions override it.
        """
        return located_spans(text, self.split_text(text))

    def split_text(self, text):
        """Return a list of text pieces."""
        raise NotImplementedError
