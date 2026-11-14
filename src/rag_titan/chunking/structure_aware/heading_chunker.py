"""Chunker that follows the heading hierarchy (handbook section 3.3)."""

from rag_titan.chunking.headings import FENCE, HEADING
from rag_titan.chunking.offsets import shift
from rag_titan.chunking.structure_aware.base import PieceChunker
from rag_titan.chunking.structure_aware.paragraph_chunker import ParagraphChunker


def only_headings(lines):
    """True when `lines` hold at least one heading and nothing else but blank lines."""
    headings = [line for line in lines if HEADING.match(line.rstrip("\r\n"))]
    others = [line for line in lines if line.strip() and line not in headings]
    return bool(headings) and not others


def split_sections(text, keep_heading_lines=False):
    """
    Cut Markdown text into sections at its headings.

    Args:
        text: Markdown text.
        keep_heading_lines: keep each "## Title" line at the start of its body.

    Returns:
        List of (path, body): path is the list of heading titles from the top
        level down, for example ["Leave policy", "Sick leave"].
    """
    return [(path, body) for path, body, _ in section_spans(text, keep_heading_lines)]


def section_spans(text, keep_heading_lines=False):
    """Like split_sections, plus where each body starts: (path, body, start)."""
    sections, levels, body, in_code = [], [], [], False
    start = position = 0   # start of the current body, and of the current line

    for line in text.splitlines(keepends=True):
        if FENCE.match(line):
            in_code = not in_code
        match = None if in_code else HEADING.match(line.rstrip("\r\n"))

        if match:
            # a kept heading line with no text under it moves into the next section
            carry = body if keep_heading_lines and only_headings(body) else []
            if not carry:
                sections.append(([title for _, title in levels], "".join(body), start))
                start = position if keep_heading_lines else position + len(line)
            level = len(match.group(1))
            # a heading closes every open heading at its level or deeper
            levels = [(lvl, title) for lvl, title in levels if lvl < level]
            levels.append((level, match.group(2).strip()))
            body = carry + [line] if keep_heading_lines else []
        else:
            body.append(line)
        position += len(line)

    sections.append(([title for _, title in levels], "".join(body), start))
    return [section for section in sections if section[1].strip()]


class HeadingChunker(PieceChunker):
    """
    Splits Markdown text at its headings, so a chunk never mixes two sections.
    Inside a section, whole paragraphs are packed up to `chunk_size` (then
    sentences, then words, when one is too big).

    Args:
        chunk_size: largest chunk, in characters (or tokens with a tokenizer),
            including the heading path.
        overlap: whole paragraphs repeated inside a section, up to this size.
        include_path: start each chunk with its heading path, for example
            "Leave policy > Sick leave". The path is skipped when it would take
            more than half of chunk_size.
        tokenizer: optional, counts tokens instead of characters.
    """

    def __init__(self, chunk_size=1000, overlap=0, include_path=True, tokenizer=None):
        super().__init__(chunk_size, overlap, tokenizer)
        self.include_path = include_path

    def split_text(self, text):
        """Return chunks that each stay inside one section."""
        return [chunk for chunk, _, _ in self.split_spans(text)]

    def split_spans(self, text):
        """(chunk with heading path, start, end); start/end point at the section text."""
        if not text.strip():
            return []

        spans = []
        for path, body, start in section_spans(text, keep_heading_lines=not self.include_path):
            spans.extend(self.section_chunks(path, body, start))
        return spans

    def section_chunks(self, path, body, start):
        """Chunk one section's body; put the heading path in front of each chunk."""
        prefix = " > ".join(path) + "\n\n" if self.include_path and path else ""
        room = self.chunk_size - self.length(prefix)

        if room < self.chunk_size / 2:
            prefix, room = "", self.chunk_size   # a very long path would crowd out the text

        inner = ParagraphChunker(room, min(self.overlap, room - 1), self.tokenizer)
        return [(prefix + chunk, s, e) for chunk, s, e in shift(inner.split_spans(body), start)]
