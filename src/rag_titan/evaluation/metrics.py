"""
Retrieval metrics: how well did a retriever find the evidence a question needs?

Evidence ids name a file, or a page of a file:
    "hr.pdf"      any chunk from hr.pdf
    "hr.pdf#p4"   a chunk on page 4 of hr.pdf
They do not use chunk ids, so the labels keep working when you change the chunker.

Each required evidence counts once: two overlapping chunks from the same page
do not make the score look better.
"""

import math


def parse_evidence_id(evidence_id):
    """
    Split an evidence id into (source, page). page is None for a whole file.

    Args:
        evidence_id: "hr.pdf" or "hr.pdf#p4".
    """
    if not isinstance(evidence_id, str) or not evidence_id.strip():
        raise ValueError(f"An evidence id must be a non-empty text, got {evidence_id!r}")

    source, _, page = evidence_id.strip().partition("#")
    if not page:
        return source, None

    if not page.startswith("p") or not page[1:].isdigit() or int(page[1:]) < 1:
        raise ValueError(f"Evidence id {evidence_id!r}: write the page as '#p4'")
    return source, int(page[1:])


def matches_evidence(chunk, evidence_id):
    """Return True when the chunk is (part of) the evidence."""
    source, page = parse_evidence_id(evidence_id)

    if chunk.get("source") != source:
        return False
    if page is None:
        return True
    return page in (chunk.get("page_numbers") or [])


def evidence_hits(retrieved, required, k):
    """
    Return, for each of the first k chunks, how many required evidences it
    found that no better chunk had found yet.
    """
    _check(required, k)
    found = set()
    hits = []

    for chunk in retrieved[:k]:
        new = {label for label in required if label not in found and matches_evidence(chunk, label)}
        found |= new
        hits.append(len(new))

    return hits


def recall_at_k(retrieved, required, k):
    """
    Share of the required evidence found in the first k chunks (0 to 1).

        Recall@k = |Relevant ∩ Retrieved_k| / |Relevant|
    """
    return sum(evidence_hits(retrieved, required, k)) / len(set(required))


def reciprocal_rank(retrieved, required, k):
    """
    1 / rank of the first chunk that is evidence, or 0 when none is in the first k.
    The mean of this over all questions is the MRR.
    """
    for rank, hit in enumerate(evidence_hits(retrieved, required, k), start=1):
        if hit:
            return 1 / rank
    return 0.0


def ndcg_at_k(retrieved, required, k):
    """
    Normalised discounted cumulative gain (0 to 1): finding evidence early counts more.

        DCG  = Σ gain_rank / log2(rank + 1),  gain = new evidences found at that rank
        IDCG = the DCG of one new evidence at each of the first min(|Relevant|, k) ranks

    One chunk can hold two evidences (say pages 4 and 5), which can beat the
    ideal, so the result is capped at 1.
    """
    hits = evidence_hits(retrieved, required, k)
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, start=1))

    ideal_ranks = min(len(set(required)), k)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_ranks + 1))

    return min(1.0, dcg / idcg)


def _check(required, k):
    """Raise a clear error for inputs that would give a meaningless score."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a whole number of at least 1, got {k!r}")
    if not required:
        raise ValueError("A question needs at least one required evidence id")
    for label in required:
        parse_evidence_id(label)
