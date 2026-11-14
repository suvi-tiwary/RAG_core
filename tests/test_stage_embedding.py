"""Tests for EmbeddingEvaluator: vector health, index coverage and search quality."""

import pytest

from rag_titan.evaluation.stages import EmbeddingEvaluator

VECTORS = {"c1": [1, 0], "c2": [0, 1], "c3": [1, 0], "c4": [0, 0]}   # c3 repeats c1, c4 is zero


def test_vector_health():
    scores = EmbeddingEvaluator().evaluate(VECTORS)

    assert scores["vector_count"] == 4
    assert scores["dimension"] == 2
    assert scores["zero_norm_rate"] == 0.25
    assert scores["duplicate_vector_rate"] == 0.25
    assert scores["mean_vector_norm"] == 0.75
    assert scores["index_coverage"] is None
    assert scores["query_topk_recall"] is None


def test_search_scores_hand_worked():
    scores = EmbeddingEvaluator().evaluate(
        VECTORS, query_vectors={"q1": [1, 0]}, relevant_ids={"q1": ["c1"]},
        indexed_ids=["c1", "c2"], index_results={"q1": ["c3"]}, k=1)

    assert scores["index_coverage"] == 0.5
    assert scores["query_topk_recall"] == 1            # exact top 1 is c1 (first of the tie)
    assert scores["ann_recall_at_k"] == 0              # the index returned c3 instead
    assert scores["positive_similarity"] == pytest.approx(1)
    assert scores["negative_similarity"] == pytest.approx(0.5)     # c2: 0, c3: 1, c4 skipped
    assert scores["similarity_margin"] == pytest.approx(0.5)


def test_matches_the_reference_sample():
    scores = EmbeddingEvaluator().evaluate(
        vectors={"c1": [0.95, 0.05], "c2": [0.8, 0.2], "c3": [0.0, 1.0], "c4": [0.1, 0.9]},
        query_vectors={"q1": [1.0, 0.0], "q2": [0.0, 1.0]},
        relevant_ids={"q1": ["c1", "c2"], "q2": ["c4"]},
        indexed_ids=["c1", "c2", "c3", "c4"],
        index_results={"q1": ["c1", "c2", "c4"], "q2": ["c3", "c4", "c2"]},
        k=3,
    )

    assert scores["index_coverage"] == 1
    assert scores["query_topk_recall"] == 1
    assert scores["ann_recall_at_k"] == 1
    assert scores["mean_vector_norm"] == pytest.approx(0.9203686296)
    assert scores["positive_similarity"] == pytest.approx(0.9875480214)
    assert scores["negative_similarity"] == pytest.approx(0.2811051968)
    assert scores["similarity_margin"] == pytest.approx(0.7064428245)


def test_query_without_gold_only_checks_the_index():
    scores = EmbeddingEvaluator().evaluate(
        {"c1": [1, 0], "c2": [0, 1]}, query_vectors={"q1": [1, 0]},
        index_results={"q1": ["c1"]}, k=1)

    assert scores["query_topk_recall"] is None
    assert scores["similarity_margin"] is None
    assert scores["ann_recall_at_k"] == 1


def test_empty_vectors():
    scores = EmbeddingEvaluator().evaluate({})

    assert scores["vector_count"] == 0
    assert scores["dimension"] is None
    assert scores["zero_norm_rate"] is None


def test_errors():
    evaluator = EmbeddingEvaluator()
    with pytest.raises(ValueError, match="different sizes"):
        evaluator.evaluate({"c1": [1, 0], "c2": [1]})
    with pytest.raises(ValueError, match="not a finite number"):
        evaluator.evaluate({"c1": [1, float("nan")]})
    with pytest.raises(ValueError, match="different sizes"):
        evaluator.evaluate({"c1": [1, 0]}, query_vectors={"q1": [1, 0, 0]})
    with pytest.raises(ValueError, match="must be a dict"):
        evaluator.evaluate([[1, 0]])
    with pytest.raises(ValueError, match="k must be"):
        evaluator.evaluate({"c1": [1, 0]}, k=0)
