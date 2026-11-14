"""
Where a chunk is, in words a reader understands (handbook section 1):
page numbers for PDFs and the heading path for documents with headings.
Both are worked out from the chunk's start/end offsets.
"""

from bisect import bisect_right

from rag_titan.chunking.headings import heading_marks

# the PDF loader puts this character between pages
PAGE_BREAK = "\f"


class Locations:
    """
    Page and section lookups for one document, prepared once and used for
    every chunk of that document.

    Args:
        text: the whole document text.
    """

    def __init__(self, text):
        # page k (1-based) starts at page_starts[k - 1]; None when there are no pages
        self.page_starts = None
        if PAGE_BREAK in text:
            breaks = [i + 1 for i, char in enumerate(text) if char == PAGE_BREAK]
            self.page_starts = [0] + breaks

        self.marks = heading_marks(text)
        self.mark_positions = [position for position, _ in self.marks]

    def describe(self, start, end):
        """
        Return the extra chunk fields for the span start..end.

        Returns:
            {"page_numbers": [3, 4]} for documents with pages,
            {"section_path": ["Leave policy", "Sick leave"]} for documents with
            headings ([] before the first heading), both, or {} when unknown.
        """
        if start is None or end is None:
            return {}

        fields = {}
        if self.page_starts:
            first = bisect_right(self.page_starts, start)
            last = bisect_right(self.page_starts, max(start, end - 1))
            fields["page_numbers"] = list(range(first, last + 1))

        if self.marks:
            index = bisect_right(self.mark_positions, start) - 1
            fields["section_path"] = list(self.marks[index][1]) if index >= 0 else []

        return fields
