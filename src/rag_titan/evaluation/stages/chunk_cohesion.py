"""
Chunk cohesion metrics: do chunks hold one topic, and are neighbours really different?

Each idea is measured two ways:
    "lexical_..."   shared words (Jaccard). Always available, a rough proxy.
    "semantic_..."  cosine of embedding vectors you pass in. Better, needs vectors.

No model is called here: you bring the vectors your embedder already made.
"""

import re

from rag_titan.evaluation.stages.text import average, cosine, jaccard

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# how many words on each side of a boundary the lexical clarity compares
EDGE_WORDS = 12


def neighbours(items):
    """
    Return the pairs of chunks that follow each other in the same document.
    Chunks are ordered by their place in the text, or by list order when
    their place is unknown.
    """
    ordered = items
    if all(item["start"] is not None for item in items):
        doc_order = {doc_id: number for number, doc_id in
                     enumerate(dict.fromkeys(item["doc_id"] for item in items))}
        ordered = sorted(items, key=lambda item: (doc_order[item["doc_id"]], item["start"]))

    return [(left, right) for left, right in zip(ordered, ordered[1:])
            if left["doc_id"] == right["doc_id"]]


def pair_mean(values, similarity):
    """
    Mean similarity of every pair of values, None with fewer than two.

        2 / (m (m - 1)) Σ_{i<j} similarity(v_i, v_j)
    """
    scores = [similarity(values[i], values[j])
              for i in range(len(values)) for j in range(i + 1, len(values))]
    return average(scores)


def lexical_scores(items):
    """
    Word-overlap proxies for cohesion, boundary clarity and stickiness.

        lexical_cohesion          mean Jaccard of the sentence pairs inside a chunk
        lexical_boundary_clarity  1 - Jaccard(last 12 words left, first 12 words right)
        lexical_chunk_stickiness  mean Jaccard of neighbouring chunks
    """
    cohesion = []
    for item in items:
        sentences = [part for part in SENTENCE_END.split(item["text"]) if part.strip()]
        cohesion.append(pair_mean(sentences, jaccard))

    clarity, stickiness = [], []
    for left, right in neighbours(items):
        stickiness.append(jaccard(left["text"], right["text"]))
        edge = jaccard(" ".join(left["text"].split()[-EDGE_WORDS:]),
                       " ".join(right["text"].split()[:EDGE_WORDS]))
        clarity.append(None if edge is None else 1 - edge)

    return {"lexical_cohesion": average(cohesion),
            "lexical_boundary_clarity": average(clarity),
            "lexical_chunk_stickiness": average(stickiness)}


def semantic_scores(items, sentence_vectors=None, chunk_vectors=None):
    """
    Embedding-based cohesion, boundary clarity and stickiness.

        Semantic Cohesion  = 2 / (m (m - 1)) Σ_{i<j} cos(e_i, e_j)   per chunk, then averaged
        Boundary Clarity   = 1 - cos(e_left, e_right)
        Chunk Stickiness   = 1 / (n - 1) Σ_i cos(e_{C_i}, e_{C_i+1})

    Args:
        items: the normalized chunks.
        sentence_vectors: {chunk_id: [one vector per sentence, in order]}.
            e_left is the last sentence of a chunk, e_right the first of the next.
        chunk_vectors: {chunk_id: vector of the whole chunk}.
    """
    sentence_vectors = _check_vectors(sentence_vectors, "sentence_vectors")
    chunk_vectors = _check_vectors(chunk_vectors, "chunk_vectors")

    cohesion = [pair_mean(vectors, cosine) for vectors in sentence_vectors.values()]

    clarity, stickiness = [], []
    for left, right in neighbours(items):
        before, after = sentence_vectors.get(left["id"]), sentence_vectors.get(right["id"])
        if before and after:
            edge = cosine(before[-1], after[0])
            clarity.append(None if edge is None else 1 - edge)
        if left["id"] in chunk_vectors and right["id"] in chunk_vectors:
            stickiness.append(cosine(chunk_vectors[left["id"]], chunk_vectors[right["id"]]))

    return {"semantic_cohesion": average(cohesion),
            "semantic_boundary_clarity": average(clarity),
            "semantic_chunk_stickiness": average(stickiness)}


def _check_vectors(vectors, name):
    """Return the vectors dict ({} when None), or raise when it is not a dict."""
    if vectors is None:
        return {}
    if not isinstance(vectors, dict):
        raise ValueError(f"{name} must be a dict keyed by chunk id")
    return vectors
