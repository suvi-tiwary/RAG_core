"""
Remove near-copies from search results (chunking handbook 2.4).

Chunks that overlap share text, so a search can return two hits that are
mostly the same passage. Using the chunk offsets, a hit is dropped when it
overlaps a better hit from the same file by at least `min_overlap` of the
shorter chunk.
"""


def has_offsets(chunk):
    """True when the chunk knows where it is in its document."""
    return chunk.get("start_offset") is not None and chunk.get("end_offset") is not None


def overlap_fraction(a, b):
    """
    How much two chunks overlap, from 0.0 (not at all) to 1.0.

    Measured as shared characters divided by the length of the shorter chunk,
    so a chunk fully inside another counts as 1.0. Chunks from different files,
    or without offsets, give 0.0.
    """
    if a.get("source") != b.get("source") or not (has_offsets(a) and has_offsets(b)):
        return 0.0

    shared = min(a["end_offset"], b["end_offset"]) - max(a["start_offset"], b["start_offset"])
    shorter = min(a["end_offset"] - a["start_offset"], b["end_offset"] - b["start_offset"])
    return max(shared, 0) / shorter if shorter > 0 else 0.0


def is_duplicate(match, kept, min_overlap):
    """True when `match` repeats one of the `kept` matches."""
    for other in kept:
        if overlap_fraction(match, other) >= min_overlap:
            return True
        # without offsets, only the very same text from the same file is a copy
        same_file = match.get("source") == other.get("source")
        if same_file and not has_offsets(match) and match.get("text") == other.get("text"):
            return True
    return False


def remove_duplicates(matches, min_overlap=0.5):
    """
    Keep matches best-first and drop the ones that repeat a better match.

    Args:
        matches: search results, best first.
        min_overlap: drop a match that overlaps a kept one by at least this
            share of the shorter chunk (0.5 = half of it).

    Returns:
        The matches that are left, in the same order.
    """
    if not 0 < min_overlap <= 1:
        raise ValueError("min_overlap must be more than 0 and at most 1")

    kept = []
    for match in matches:
        if not is_duplicate(match, kept, min_overlap):
            kept.append(match)
    return kept
