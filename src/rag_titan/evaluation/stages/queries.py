"""
Checks for the inputs every stage evaluator shares: the queries list and k.

A query is a dict. Only "id" is always needed; each stage reads the keys it
scores and skips the queries that do not have them:
    {"id": "q1",
     "expected_chunk_ids": ["c2"],          # or "relevance": {"c2": 3, "c1": 1}
     "retrieved_ids": ["c1", "c2", "c3"],   # what your retriever returned, best first
     "reranked_ids": ["c2", "c1", "c3"]}    # what your reranker returned
"""

from rag_titan.evaluation.stages.text import is_number


def check_k(k):
    """Raise a clear error when k is not a whole number of at least 1."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a whole number of at least 1, got {k!r}")
    return k


def check_queries(queries):
    """Check that queries is a list of dicts with unique ids, and return it."""
    if not isinstance(queries, list):
        raise ValueError("queries must be a list of dicts")

    seen = set()
    for number, query in enumerate(queries, start=1):
        if not isinstance(query, dict) or query.get("id") is None:
            raise ValueError(f"Query {number} must be a dict with an 'id'")
        if query["id"] in seen:
            raise ValueError(f"Query id {query['id']!r} is used more than once")
        seen.add(query["id"])

    return queries


def reference_answer(query):
    """
    Return the reference answer of a query, or None when there is none.

    A question the documents cannot answer ("answerable": False) has no
    reference: the right reply is a refusal, and Abstention Accuracy scores
    that. Comparing a refusal with an empty reference would call it wrong.
    """
    if query.get("answerable") is False:
        return None
    return query.get("expected_answer")


def relevance_of(query):
    """
    Return the gold labels of a query as {chunk_id: grade}.

    "relevance" gives graded labels (3 = perfect, 1 = related, 0 = not relevant).
    Without it every id in "expected_chunk_ids" gets grade 1.
    """
    if "relevance" not in query:
        return {chunk_id: 1 for chunk_id in query.get("expected_chunk_ids") or []}

    relevance = query["relevance"]
    if not isinstance(relevance, dict):
        raise ValueError(f"Query {query['id']!r}: 'relevance' must be a dict "
                         "of {chunk_id: grade}")
    for chunk_id, grade in relevance.items():
        if not is_number(grade) or grade < 0:
            raise ValueError(f"Query {query['id']!r}: grade of {chunk_id!r} must be 0 or more")
    return relevance


def ranked_ids(query, key):
    """Return the list of chunk ids under `key`, e.g. "retrieved_ids"."""
    ids = query[key]
    if not isinstance(ids, list):
        raise ValueError(f"Query {query['id']!r}: {key!r} must be a list of chunk ids")
    return ids
