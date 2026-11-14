"""
Packing: the shared step of every chunker that keeps whole units.

A chunker first cuts the text into pieces it never wants to break (recursive
pieces, sentences, paragraphs). pack() then groups neighbouring pieces into
chunks of at most chunk_size, and starts each new chunk by repeating the last
whole pieces of the previous one, up to `overlap`.
"""


def total(indexes, sizes):
    """Sum of the sizes of the pieces at `indexes`."""
    return sum(sizes[i] for i in indexes)


def overlap_tail(indexes, sizes, overlap):
    """Return the last whole pieces of `indexes` whose total size is at most `overlap`."""
    tail = []

    for index in reversed(indexes):
        if total(tail, sizes) + sizes[index] > overlap:
            break
        tail.insert(0, index)

    return tail


def pack(sizes, chunk_size, overlap=0):
    """
    Group pieces into chunks, by index.

    Args:
        sizes: the size of each piece, in order (characters, tokens, ...).
        chunk_size: largest total size of one chunk. A single piece bigger
            than this gets a chunk of its own; split such pieces first.
        overlap: largest total size repeated from the previous chunk.

    Returns:
        List of chunks, each a list of piece indexes, for example
        [[0, 1], [1, 2], [2, 3]] when every chunk repeats one piece.
    """
    groups, current = [], []

    for index, size in enumerate(sizes):
        if current and total(current, sizes) + size > chunk_size:
            groups.append(current)
            current = overlap_tail(current, sizes, overlap)

            # drop repeated pieces until the new piece fits
            while current and total(current, sizes) + size > chunk_size:
                current.pop(0)

        current.append(index)

    if current:
        groups.append(current)
    return groups


def join_groups(pieces, groups):
    """Turn groups of piece indexes into chunk strings."""
    return ["".join(pieces[i] for i in group) for group in groups]
