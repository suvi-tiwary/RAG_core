"""
Chunk shape metrics: sizes, coverage of the source, gold spans kept whole, metadata.

Notation used in the formulas:
    C_i   chunk i, |C_i| its number of words
    N     number of words in the source documents
    U     set of source words covered by at least one chunk
    G     gold spans: evidence or structural blocks (tables, lists, code)
"""

import math

from rag_titan.evaluation.stages.coverage import covered_tokens, document_tokens
from rag_titan.evaluation.stages.spans import chunk_spans, is_inside
from rag_titan.evaluation.stages.text import average, is_number, tokenize


def check_size_limits(size_limits):
    """Raise a clear error unless size_limits is None or a (min, max) pair."""
    if size_limits is None:
        return
    valid = (isinstance(size_limits, (list, tuple)) and len(size_limits) == 2
             and all(is_number(limit) and limit >= 0 for limit in size_limits)
             and size_limits[0] <= size_limits[1])
    if not valid:
        raise ValueError(f"size_limits must be a (min, max) pair of word counts, "
                         f"got {size_limits!r}")


def size_scores(items, size_limits=None):
    """
    Size statistics of the chunks, counted in words.

        Mean Chunk Size  μ = (1/n) Σ |C_i|
        Size Variation   CV = σ / μ          (σ = population standard deviation)
        Size Compliance  = #{C_i : L_min <= |C_i| <= L_max} / #C
    """
    lengths = [len(tokenize(item["text"])) for item in items]
    mean = average(lengths)

    cv, compliance = None, None
    if mean:
        variance = sum((length - mean) ** 2 for length in lengths) / len(lengths)
        cv = math.sqrt(variance) / mean
    if lengths and size_limits is not None:
        low, high = size_limits
        compliance = sum(low <= length <= high for length in lengths) / len(lengths)

    return {
        "mean_chunk_tokens": mean,
        "min_chunk_tokens": min(lengths, default=None),
        "max_chunk_tokens": max(lengths, default=None),
        "size_cv": cv,
        "size_compliance": compliance,
    }


def coverage_scores(items, documents):
    """
    How much of the source the chunks keep, and how much they repeat.

        Document Coverage = |U| / N
        Redundancy Ratio  = (Σ |C_i| - |U|) / |U|      ("overlap_ratio")

    Both are None when there is no source text or a chunk could not be placed.
    """
    tokens = document_tokens(documents)
    total = sum(len(words) for words in tokens.values())
    scores = {"total_source_tokens": total if documents else None,
              "coverage": None, "overlap_ratio": None}

    if not documents or any(item["start"] is None for item in items):
        return scores

    covered = covered_tokens(tokens, chunk_spans(items))
    indexed = sum(len(tokenize(item["text"])) for item in items)
    if total:
        scores["coverage"] = len(covered) / total
    if covered:
        scores["overlap_ratio"] = (indexed - len(covered)) / len(covered)
    return scores


def intact_rate(gold_spans, items):
    """
    Share of the gold spans that lie completely inside one chunk.

        Evidence Completeness = #gold spans intact in one chunk / #G
        Structural Integrity  = #intact structural blocks / #blocks
    """
    if not gold_spans or any(item["start"] is None for item in items):
        return None
    spans = chunk_spans(items)
    return sum(is_inside(span, spans) for span in gold_spans) / len(gold_spans)


def fragmentation(gold_spans, items):
    """
    Average number of chunks a gold span touches. 1.0 is the best: no span is split.
    """
    if not gold_spans or any(item["start"] is None for item in items):
        return None

    counts = []
    for doc_id, start, end in gold_spans:
        counts.append(sum(item["doc_id"] == doc_id and item["start"] < end and item["end"] > start
                          for item in items))
    return average(counts)


def metadata_completeness(items, required):
    """Share of the chunks that have a value for every required metadata key."""
    if not required or not items:
        return None
    complete = sum(all(item["metadata"].get(key) is not None for key in required)
                   for item in items)
    return complete / len(items)


def metadata_accuracy(items, expected):
    """
    Share of the expected metadata values the chunks really have.

    Args:
        expected: {chunk_id: {key: value}}, e.g. {"c1": {"page": 1}}.
    """
    if not expected:
        return None
    if not isinstance(expected, dict):
        raise ValueError("expected_metadata must be a dict of {chunk_id: {key: value}}")

    by_id = {item["id"]: item["metadata"] for item in items}
    checks = [by_id.get(chunk_id, {}).get(key) == value
              for chunk_id, fields in expected.items()
              for key, value in fields.items()]
    return sum(checks) / len(checks) if checks else None
