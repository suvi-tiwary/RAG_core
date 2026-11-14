"""
Where each chunk sits in its document (handbook section 1: offsets).

Most chunkers know their exact positions while they cut (see group_spans).
locate() is the fallback for chunkers that do not (TokenChunker, chunkers you
write yourself): chunks come in document order, so each one is searched for
starting just after the previous chunk's start. An exact match is tried first;
then a match that allows different spacing. With very repetitive text a
search can pick an earlier copy of the same words, which is why chunkers that
can report exact positions do.
"""

import re


def flexible_pattern(text):
    """A regex for `text` where any run of spaces/line breaks matches any other run."""
    words = text.split()
    if not words:
        return None
    return re.compile(r"\s+".join(re.escape(word) for word in words))


def find_part(text, part, search_from):
    """Return (start, end) of `part` in `text` at or after search_from, or (None, None)."""
    start = text.find(part, search_from) if part else -1
    if start != -1:
        return start, start + len(part)

    pattern = flexible_pattern(part)
    match = pattern.search(text, search_from) if pattern else None
    if match:
        return match.start(), match.end()
    return None, None


def locate(text, parts):
    """
    Return (start, end) offsets for each part of `text`, in order.

    Args:
        text: the whole document text.
        parts: the source text of each chunk, in document order.

    Returns:
        One (start, end) per part, so text[start:end] is that part. A part that
        cannot be found gets (None, None) and does not move the search on.
    """
    spans, search_from = [], 0

    for part in parts:
        start, end = find_part(text, part, search_from)
        spans.append((start, end))
        if start is not None:
            search_from = start + 1   # the next chunk starts after this one's start

    return spans


def trim_span(text, start, end):
    """Move start/end inwards past spaces and line breaks, like str.strip()."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def group_spans(text, pieces, groups):
    """
    Exact (chunk, start, end) for chunks made of whole pieces of `text`.

    Args:
        text: the text the pieces were cut from.
        pieces: pieces that join back into `text` exactly.
        groups: piece indexes of each chunk, from packing.pack().

    Returns:
        One (chunk_text, start, end) per non-blank chunk, with the chunk's
        edges trimmed. None when the pieces do not join back into `text`
        (a tokenizer can change the text); then use locate() instead.
    """
    if "".join(pieces) != text:
        return None

    starts, position = [], 0
    for piece in pieces:
        starts.append(position)
        position += len(piece)

    spans = []
    for group in groups:
        first, last = group[0], group[-1]
        start, end = trim_span(text, starts[first], starts[last] + len(pieces[last]))
        if start < end:
            spans.append((text[start:end], start, end))
    return spans


def located_spans(text, chunks):
    """(chunk, start, end) for each chunk, found by searching (see locate)."""
    return [(chunk, start, end) for chunk, (start, end) in zip(chunks, locate(text, chunks))]


def shift(spans, offset):
    """Move spans found inside a part of the text (a page, a section) by `offset`."""
    def move(position):
        return None if position is None else position + offset

    return [(chunk, move(start), move(end)) for chunk, start, end in spans]
