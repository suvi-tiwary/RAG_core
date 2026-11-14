"""Helper that turns chunks into one text block for the LLM."""


def describe_source(chunk):
    """
    The header line of one chunk: its id, file, and page/section when known.

    Example:
        [chunk_4] Source: handbook.pdf, page 3, section: Leave policy > Sick leave
    """
    parts = [str(chunk["source"])]

    pages = chunk.get("page_numbers")
    if pages:
        parts.append(f"page {pages[0]}" if len(pages) == 1 else f"pages {pages[0]}-{pages[-1]}")

    if chunk.get("section_path"):
        parts.append("section: " + " > ".join(chunk["section_path"]))

    return f"[{chunk['id']}] Source: {', '.join(parts)}"


def build_context(chunks):
    """
    Join chunks into one text block the LLM can read and cite.

    Each chunk looks like:
        [chunk_3] Source: handbook.pdf, page 2
        ...chunk text...
    """
    return "\n\n".join(f"{describe_source(chunk)}\n{chunk['text']}" for chunk in chunks)
