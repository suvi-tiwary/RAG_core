"""
Sampling: choose which documents to write golden questions for.

    ids = sample_documents(documents, size=20, strategy="stratified", seed=7)

Strategies:
    "random"      every document has the same chance
    "stratified"  every category gets its fair share (needs "category" on the documents)
    "risk"        documents with profiling issues first (short, duplicates, ...),
                  because that is where a RAG system fails most; the rest random

The same seed always gives the same sample, so a run can be repeated.
"""

import random

from rag_titan.evaluation.profiling import as_records, profile_documents

STRATEGIES = ["random", "stratified", "risk"]


def sample_documents(documents, size, strategy="random", seed=0, profile=None):
    """
    Return the ids of a sample of the documents.

    Args:
        documents: {doc_id: text} or a list of document dicts.
        size: how many documents to pick (at most all of them).
        strategy: "random", "stratified" or "risk".
        seed: the random seed, for a repeatable sample.
        profile: a profile_documents() result for "risk"; made when missing.
    """
    records = as_records(documents)
    if not isinstance(size, int) or isinstance(size, bool) or not 1 <= size <= len(records):
        raise ValueError(f"size must be a whole number from 1 to {len(records)}, got {size!r}")
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}: use one of {STRATEGIES}")

    rng = random.Random(seed)
    if strategy == "stratified":
        return _stratified(records, size, rng)
    if strategy == "risk":
        return _risk_first(records, size, rng, profile or profile_documents(records))
    return rng.sample([record["id"] for record in records], size)


def _stratified(records, size, rng):
    """
    Pick from every category in proportion to its size.
    Seats are shared out by the largest-remainder method, so they add up to `size`.
    """
    groups = {}
    for record in records:
        groups.setdefault(record["category"], []).append(record["id"])

    exact = {category: size * len(ids) / len(records) for category, ids in groups.items()}
    seats = {category: int(share) for category, share in exact.items()}
    # on a tie the category with fewer seats wins, so a small one is not left out
    by_remainder = sorted(groups, key=lambda category: (exact[category] - seats[category],
                                                        -seats[category]), reverse=True)
    for category in by_remainder[:size - sum(seats.values())]:
        seats[category] += 1

    sample = []
    for category, ids in groups.items():
        sample += rng.sample(ids, seats[category])
    return sample


def _risk_first(records, size, rng, profile):
    """Pick the documents with the most profiling issues first, then fill up at random."""
    issues = profile["issues"]
    ids = [record["id"] for record in records]
    rng.shuffle(ids)      # a random order among documents with the same number of issues
    risky = sorted((doc_id for doc_id in ids if doc_id in issues),
                   key=lambda doc_id: len(issues[doc_id]), reverse=True)
    safe = [doc_id for doc_id in ids if doc_id not in issues]
    return (risky + safe)[:size]
