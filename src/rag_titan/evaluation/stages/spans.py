"""
Find where chunks and gold evidence sit in the source documents.

A span is (doc_id, start, end): characters start..end of one document.
Stages compare spans to answer questions like "did the retrieved chunks
cover the gold evidence?" without caring how the text was chunked.

Source text can be given in two ways:
    document_text="one long text"                  a single document
    documents={"hr.md": "...", "it.md": "..."}     many documents
With many documents every chunk needs a "doc_id".
"""

# the doc_id used when only one text is given as document_text
DEFAULT_DOC = "document"


def as_documents(document_text=None, documents=None):
    """
    Return the source text as a dict of {doc_id: text}.
    The dict is empty when no source text was given.
    """
    if document_text is not None and documents is not None:
        raise ValueError("Give document_text or documents, not both")

    if document_text is not None:
        if not isinstance(document_text, str):
            raise ValueError("document_text must be a text")
        return {DEFAULT_DOC: document_text}

    if documents is None:
        return {}
    if not isinstance(documents, dict) or not documents:
        raise ValueError("documents must be a non-empty dict of {doc_id: text}")
    for doc_id, text in documents.items():
        if not isinstance(text, str):
            raise ValueError(f"Document {doc_id!r} must be a text")
    return documents


def normalize_chunks(chunks, documents):
    """
    Give every chunk the same shape and find its place in the documents.

    Args:
        chunks: list of texts, or dicts with "id", "text", "start", "end",
            "doc_id" and "metadata" (all optional except what you have).
        documents: the dict from as_documents().

    Returns:
        (items, warnings). Each item has "id", "text", "doc_id", "start",
        "end" and "metadata". "start" and "end" are None when the chunk
        could not be placed; a warning then says why.
    """
    if not isinstance(chunks, list):
        raise ValueError("chunks must be a list")

    items, warnings, seen = [], [], set()
    for number, chunk in enumerate(chunks, start=1):
        item = {"text": chunk} if isinstance(chunk, str) else dict(chunk)
        item.setdefault("id", f"chunk_{number}")
        item.setdefault("metadata", {})
        if item["id"] in seen:
            raise ValueError(f"Chunk id {item['id']!r} is used more than once")
        seen.add(item["id"])

        _place_chunk(item, documents, warnings)
        items.append(item)

    return items, warnings


def _place_chunk(item, documents, warnings):
    """Fill in doc_id, start, end and text of one chunk (changes the item)."""
    start, end = item.get("start"), item.get("end")
    item["start"], item["end"] = None, None
    item["doc_id"] = _doc_id_of(item, documents)
    text = documents.get(item["doc_id"])

    if text is None:
        # no source text: ranking metrics still work, span metrics do not
        item.setdefault("text", "")
        return

    if start is None or end is None:
        found = _find_once(item.get("text") or "", text)
        if found is None:
            warnings.append(f"{item['id']}: text not found exactly once; give start and end")
        else:
            item["start"], item["end"] = found
        item.setdefault("text", "")
        return

    if not (_is_offset(start) and _is_offset(end) and start <= end <= len(text)):
        raise ValueError(f"Chunk {item['id']!r}: start and end must fit inside the document")
    if item.get("text") is not None and text[start:end] != item["text"]:
        raise ValueError(f"Chunk {item['id']!r}: text does not match the document at its offsets")

    item["start"], item["end"] = start, end
    item["text"] = text[start:end]


def _doc_id_of(item, documents):
    """Return the chunk's document id, or the only document when there is one."""
    doc_id = item.get("doc_id")
    if doc_id is None:
        if len(documents) > 1:
            raise ValueError(f"Chunk {item['id']!r} needs a 'doc_id': there are several documents")
        return next(iter(documents), DEFAULT_DOC)

    if documents and doc_id not in documents:
        raise ValueError(f"Chunk {item['id']!r}: unknown doc_id {doc_id!r}")
    return doc_id


def _is_offset(value):
    """Return True for a whole number of at least 0."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _find_once(part, text):
    """Return (start, end) of `part` in `text`, or None when it is missing or repeated."""
    first = text.find(part) if part else -1
    if first < 0 or text.find(part, first + 1) >= 0:
        return None
    return first, first + len(part)


def evidence_spans(values, documents):
    """
    Turn gold evidence into spans.

    Args:
        values: list of exact quotes (each must occur exactly once in the
            documents) or dicts {"start": 10, "end": 42, "doc_id": "hr.md"}.
        documents: the dict from as_documents().
    """
    spans = []
    for value in values or []:
        if isinstance(value, str):
            spans.append(_quote_span(value, documents))
        else:
            spans.append(_offset_span(value, documents))
    return spans


def _quote_span(quote, documents):
    """Find a quote that occurs exactly once in all the documents."""
    found = []
    for doc_id, text in documents.items():
        first = text.find(quote) if quote else -1
        if first >= 0:
            found.append((doc_id, first, first + len(quote)))
            if text.find(quote, first + 1) >= 0:
                found.append(None)   # a second match in the same document

    if len(found) != 1:
        raise ValueError(f"Gold evidence {quote[:40]!r} must occur exactly once in the "
                         "documents; give start and end instead")
    return found[0]


def _offset_span(value, documents):
    """Check a gold {"start", "end", "doc_id"} dict and return it as a span."""
    if not isinstance(value, dict):
        raise ValueError(f"Gold evidence must be a quote or a dict, got {value!r}")

    doc_id = _doc_id_of({"id": "gold evidence", "doc_id": value.get("doc_id")}, documents)
    start, end = value.get("start"), value.get("end")
    size = len(documents.get(doc_id, ""))
    if not (_is_offset(start) and _is_offset(end) and start < end <= size):
        raise ValueError(f"Gold evidence span {value!r} does not fit inside the document")
    return doc_id, start, end


def chunk_spans(items):
    """Return the spans of the chunks that were placed in a document."""
    return [(item["doc_id"], item["start"], item["end"])
            for item in items if item["start"] is not None]


def is_inside(span, spans):
    """Return True when `span` lies completely inside one of `spans`."""
    doc_id, start, end = span
    return any(other_doc == doc_id and other_start <= start and end <= other_end
               for other_doc, other_start, other_end in spans)
