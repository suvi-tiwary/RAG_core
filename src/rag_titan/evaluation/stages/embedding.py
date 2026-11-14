"""
Embedding evaluation: are the vectors healthy, indexed, and able to find the gold chunks?

    scores = EmbeddingEvaluator().evaluate(
        vectors={"c1": [0.9, 0.1], "c2": [0.1, 0.9]},      # one vector per chunk
        query_vectors={"q1": [1.0, 0.0]},                  # one vector per query
        relevant_ids={"q1": ["c1"]},                       # gold chunks per query
        k=1,
    )
    print(scores["query_topk_recall"])   # 1.0

No model is called: pass the vectors your embedder and vector index produced.
"""

import math

from rag_titan.evaluation.stages.queries import check_k
from rag_titan.evaluation.stages.ranking import difference
from rag_titan.evaluation.stages.text import average, cosine, is_number


class EmbeddingEvaluator:
    """Scores chunk and query vectors, and the results of a vector index."""

    def evaluate(self, vectors, query_vectors=None, relevant_ids=None, indexed_ids=None,
                 index_results=None, k=5):
        """
        Score the vectors.

        Args:
            vectors: {chunk_id: vector} of every chunk.
            query_vectors: {query_id: vector}.
            relevant_ids: {query_id: [gold chunk ids]}.
            indexed_ids: the chunk ids really stored in your vector index.
            index_results: {query_id: [chunk ids the index returned, best first]}.
            k: how many neighbours to compare.

        Returns:
            A dict of metric name -> value (None = not measured).
        """
        check_k(k)
        if not isinstance(vectors, dict):
            raise ValueError("vectors must be a dict of {chunk_id: vector}")

        scores = self._health(vectors)
        scores["index_coverage"] = None
        if indexed_ids is not None and vectors:
            scores["index_coverage"] = len(set(vectors) & set(indexed_ids)) / len(vectors)

        scores.update(self._search_scores(vectors, query_vectors or {}, relevant_ids or {},
                                          index_results, k))
        return scores

    def _health(self, vectors):
        """
        Basic checks on the stored vectors.

            zero_norm_rate         vectors of length 0 / all vectors (cannot be searched)
            duplicate_vector_rate  vectors equal to an earlier one / all vectors
        """
        sizes = {len(vector) for vector in vectors.values()}
        if len(sizes) > 1:
            raise ValueError(f"Vectors have different sizes: {sorted(sizes)}")

        norms = []
        for chunk_id, vector in vectors.items():
            if not all(is_number(value) for value in vector):
                raise ValueError(f"Vector of {chunk_id!r} has a value that is not a finite number")
            norms.append(math.sqrt(sum(value * value for value in vector)))

        count = len(vectors)
        distinct = len({tuple(vector) for vector in vectors.values()})
        return {
            "vector_count": count,
            "dimension": next(iter(sizes), None),
            "zero_norm_rate": sum(norm == 0 for norm in norms) / count if count else None,
            "duplicate_vector_rate": (count - distinct) / count if count else None,
            "mean_vector_norm": average(norms),
        }

    def _search_scores(self, vectors, query_vectors, relevant_ids, index_results, k):
        """
        Search every query by exact cosine and compare with the gold chunks and the index.

            query_topk_recall    gold chunks in the exact top K / gold chunks
            ann_recall_at_k      exact top K also returned by your index / exact top K
            positive_similarity  mean cos(query, gold chunk)
            negative_similarity  mean cos(query, other chunk)
            similarity_margin    positive - negative (higher = easier to separate)
        """
        recalls, ann_recalls, positive, negative = [], [], [], []
        for query_id, query_vector in query_vectors.items():
            scored = [(chunk_id, cosine(query_vector, vector))
                      for chunk_id, vector in vectors.items()]
            scored = [(chunk_id, score) for chunk_id, score in scored if score is not None]
            exact = [chunk_id for chunk_id, _ in
                     sorted(scored, key=lambda pair: pair[1], reverse=True)[:k]]

            gold = set(relevant_ids.get(query_id) or [])
            if gold:
                recalls.append(len(set(exact) & gold) / len(gold))
                positive += [score for chunk_id, score in scored if chunk_id in gold]
                negative += [score for chunk_id, score in scored if chunk_id not in gold]
            if index_results is not None and query_id in index_results and exact:
                returned = set(index_results[query_id][:k])
                ann_recalls.append(len(returned & set(exact)) / len(exact))

        return {
            "query_topk_recall": average(recalls),
            "ann_recall_at_k": average(ann_recalls),
            "positive_similarity": average(positive),
            "negative_similarity": average(negative),
            "similarity_margin": difference(average(positive), average(negative)),
        }
