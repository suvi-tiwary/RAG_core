"""
Boundary Alignment F1: do the chunker's split points match the human-marked ones?

A boundary is the place where one chunk ends and the next begins. The end of
a document is not a boundary: every chunker "finds" it.

    P_b = #matched boundaries / #predicted boundaries
    R_b = #matched boundaries / #ideal boundaries
    Boundary F1 = 2 P_b R_b / (P_b + R_b)
"""


def boundaries(items):
    """Return {doc_id: sorted chunk ends}, without the last end of each document."""
    ends = {}
    for item in items:
        if item["end"] is not None:
            ends.setdefault(item["doc_id"], set()).add(item["end"])
    return {doc_id: sorted(values)[:-1] for doc_id, values in ends.items()}


def count_matches(predicted, ideal, tolerance):
    """
    Count the predicted boundaries that pair up with an ideal one.
    Each ideal boundary can be used once; the nearest free one is taken.
    """
    free = set(range(len(ideal)))
    hits = 0
    for position in predicted:
        near = [index for index in free if abs(position - ideal[index]) <= tolerance]
        if near:
            free.remove(min(near, key=lambda index: abs(position - ideal[index])))
            hits += 1
    return hits


def boundary_f1(items, expected_items, tolerance=0):
    """
    F1 of the predicted split points against the ideal ones (0 to 1).

    Args:
        items: the normalized chunks of your chunker.
        expected_items: the normalized ideal chunks (None = not measured).
        tolerance: how many characters apart two boundaries may be and still match.
    """
    if expected_items is None or any(item["start"] is None for item in items):
        return None
    if not isinstance(tolerance, int) or isinstance(tolerance, bool) or tolerance < 0:
        raise ValueError(f"boundary_tolerance must be a whole number of at least 0, "
                         f"got {tolerance!r}")

    predicted, ideal = boundaries(items), boundaries(expected_items)
    hits = sum(count_matches(predicted.get(doc_id, []), ideal.get(doc_id, []), tolerance)
               for doc_id in predicted)
    predicted_count = sum(len(ends) for ends in predicted.values())
    ideal_count = sum(len(ends) for ends in ideal.values())

    # nothing to split and nothing split is a perfect match
    precision = hits / predicted_count if predicted_count else float(not ideal_count)
    recall = hits / ideal_count if ideal_count else float(not predicted_count)
    if not precision + recall:
        return 0.0
    return 2 * precision * recall / (precision + recall)
