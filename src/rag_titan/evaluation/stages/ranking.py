"""
Ranking metrics: how good is an ordered list of chunk ids?

They need only ids and gold labels, so they work for any retriever or reranker:
    scores = ranking_scores(["c1", "c3", "c2"], {"c2": 3, "c1": 1}, k=3)

A chunk id that appears twice counts once, at its first rank: repeating a
relevant chunk never makes a score look better.
"""

import math

# every key ranking_scores() returns
RANKING_KEYS = ["precision_at_k", "recall_at_k", "f1_at_k", "hit_at_k", "mrr_at_k",
                "ndcg_at_k", "average_precision_at_k"]


def gains_at_k(ids, relevance, k):
    """
    Return the gold grade of each of the first k ids (0 = not relevant).
    A repeated id gets 0 the second time.
    """
    seen = set()
    gains = []
    for chunk_id in ids[:k]:
        gains.append(0 if chunk_id in seen else relevance.get(chunk_id, 0))
        seen.add(chunk_id)
    return gains


def first_relevant_rank(ids, relevance, k=None):
    """Return the rank (1 = top) of the first relevant id, or None when there is none."""
    limit = len(ids) if k is None else k
    for rank, gain in enumerate(gains_at_k(ids, relevance, limit), start=1):
        if gain > 0:
            return rank
    return None


def dcg(gains):
    """
    Discounted cumulative gain: high grades count more, late ranks count less.

        DCG = Σ (2^grade - 1) / log2(rank + 1)
    """
    return sum((2 ** gain - 1) / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_at_k(ids, relevance, k):
    """
    DCG of the list divided by the DCG of the best possible list (0 to 1).

        nDCG@K = DCG@K / IDCG@K
    """
    ideal = sorted(relevance.values(), reverse=True)[:k]
    best = dcg(ideal)
    if not best:
        return None
    return dcg(gains_at_k(ids, relevance, k)) / best


def average_precision_at_k(ids, relevance, k):
    """
    Mean of the precision measured at each relevant rank (0 to 1).

        AP@K = Σ Precision@rank (at relevant ranks) / min(#relevant, K)
    """
    positives = sum(grade > 0 for grade in relevance.values())
    if not positives:
        return None

    found, total = 0, 0.0
    for rank, gain in enumerate(gains_at_k(ids, relevance, k), start=1):
        if gain > 0:
            found += 1
            total += found / rank
    return total / min(positives, k)


def ranking_scores(ids, relevance, k):
    """
    Score one ranked list against its gold labels.

    Args:
        ids: chunk ids, best first.
        relevance: {chunk_id: grade}; ids that are not listed have grade 0.
        k: how many results to score.

    Returns:
        A dict with every key of RANKING_KEYS. All values are None when the
        query has no relevant chunk: it cannot be scored.

        Precision@K = #relevant in top K / K
        Recall@K    = #relevant in top K / #relevant
        F1@K        = 2 P R / (P + R)
        Hit@K       = 1 when a relevant chunk is in the top K
        MRR@K       = 1 / rank of the first relevant chunk (0 when none)
    """
    positives = sum(grade > 0 for grade in relevance.values())
    if not positives:
        return {key: None for key in RANKING_KEYS}

    found = sum(gain > 0 for gain in gains_at_k(ids, relevance, k))
    precision = found / k
    recall = found / positives
    first = first_relevant_rank(ids, relevance, k)

    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_at_k": 2 * precision * recall / (precision + recall) if found else 0.0,
        "hit_at_k": float(found > 0),
        "mrr_at_k": 1 / first if first else 0.0,
        "ndcg_at_k": ndcg_at_k(ids, relevance, k),
        "average_precision_at_k": average_precision_at_k(ids, relevance, k),
    }


def difference(after, before):
    """Return after - before, or None when one of them was not measured."""
    if after is None or before is None:
        return None
    return after - before
