"""
Data profiling: check the quality of a document collection before you evaluate on it.

    profile = profile_documents({"hr.md": "...", "it.md": "..."})
    print(profile["clean_share"])      # 0.5: half the documents have no issue
    print(profile["issues"])           # {"it.md": ["empty"]}

Documents can be {doc_id: text}, or a list of dicts with "id" (or "source"),
"text" and optionally "category", which is what rag-titan's loaders return.
"""

from collections import Counter

from rag_titan.evaluation.stages.text import average, is_number, tokenize

UNCATEGORIZED = "uncategorized"


def as_records(documents):
    """Return the documents as a list of {"id", "text", "category"} with unique ids."""
    if isinstance(documents, dict):
        documents = [{"id": doc_id, "text": text} for doc_id, text in documents.items()]
    if not isinstance(documents, list) or not documents:
        raise ValueError("documents must be a non-empty dict of {doc_id: text} or list of dicts")

    records, seen = [], set()
    for number, document in enumerate(documents, start=1):
        if not isinstance(document, dict) or not isinstance(document.get("text"), str):
            raise ValueError(f"Document {number} must be a dict with a 'text'")
        doc_id = document.get("id", document.get("source"))
        if doc_id is None or doc_id in seen:
            raise ValueError(f"Document {number} needs a unique 'id' (or 'source')")
        seen.add(doc_id)
        records.append({"id": doc_id, "text": document["text"],
                        "category": document.get("category") or UNCATEGORIZED})
    return records


def profile_documents(documents, min_words=20, near_duplicate=0.9):
    """
    Profile a document collection: sizes, empty and short documents, duplicates, categories.

    Args:
        documents: {doc_id: text} or a list of document dicts.
        min_words: documents with fewer words are reported as "short".
        near_duplicate: two documents sharing at least this share of their
            words (Jaccard) are near duplicates. None skips the check, which
            compares every pair and is slow above a few thousand documents.

    Returns:
        A dict with the counts and word statistics, "categories" (count per
        category), "exact_duplicates" (groups of ids), "near_duplicates"
        (pairs with their similarity), "issues" ({doc_id: [reasons]}) and
        "clean_share" (documents without any issue / all documents).
    """
    if near_duplicate is not None and (not is_number(near_duplicate)
                                       or not 0 < near_duplicate <= 1):
        raise ValueError("near_duplicate must be above 0 and at most 1, or None")

    records = as_records(documents)
    words = {record["id"]: tokenize(record["text"]) for record in records}
    lengths = [len(tokens) for tokens in words.values()]
    exact = _exact_duplicates(words)
    near = _near_duplicates(words, near_duplicate, exact) if near_duplicate else []

    issues = _issues(words, min_words, exact, near)
    return {
        "document_count": len(records),
        "total_words": sum(lengths),
        "mean_words": average(lengths),
        "min_words": min(lengths),
        "max_words": max(lengths),
        "categories": dict(Counter(record["category"] for record in records)),
        "exact_duplicates": exact,
        "near_duplicates": near,
        "issues": issues,
        "clean_share": (len(records) - len(issues)) / len(records),
    }


def _exact_duplicates(words):
    """Return groups of ids whose documents have exactly the same words in the same order."""
    groups = {}
    for doc_id, tokens in words.items():
        if tokens:      # empty documents are reported as "empty", not as duplicates
            groups.setdefault(tuple(tokens), []).append(doc_id)
    return [ids for ids in groups.values() if len(ids) > 1]


def _near_duplicates(words, threshold, exact):
    """Return the pairs of documents that share most of their words but are not identical."""
    same = {frozenset((first, second)) for ids in exact for first in ids for second in ids}
    sets = {doc_id: set(tokens) for doc_id, tokens in words.items() if tokens}
    ids = list(sets)

    pairs = []
    for index, first in enumerate(ids):
        for second in ids[index + 1:]:
            if frozenset((first, second)) in same:
                continue
            shared = len(sets[first] & sets[second]) / len(sets[first] | sets[second])
            if shared >= threshold:
                pairs.append({"ids": [first, second], "similarity": round(shared, 4)})
    return pairs


def _issues(words, min_words, exact, near):
    """Return {doc_id: [reasons]} for every document that needs a look."""
    issues = {}

    def flag(doc_id, reason):
        if reason not in issues.setdefault(doc_id, []):
            issues[doc_id].append(reason)

    for doc_id, tokens in words.items():
        if not tokens:
            flag(doc_id, "empty")
        elif len(tokens) < min_words:
            flag(doc_id, "short")
    for ids in exact:
        for doc_id in ids[1:]:      # the first copy is the one to keep
            flag(doc_id, "exact duplicate")
    for pair in near:
        flag(pair["ids"][1], "near duplicate")
    return issues
