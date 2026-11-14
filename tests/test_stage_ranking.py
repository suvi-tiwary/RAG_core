"""Tests for the shared stage helpers: text, queries and the ranking metrics."""

import math

import pytest

from rag_titan.evaluation.stages.queries import check_k, check_queries, relevance_of
from rag_titan.evaluation.stages.ranking import (
    RANKING_KEYS,
    first_relevant_rank,
    gains_at_k,
    ranking_scores,
)
from rag_titan.evaluation.stages.text import (
    average,
    cosine,
    jaccard,
    percentile,
    token_positions,
    tokenize,
)

# ranks:  1 (grade 1)  2 (not relevant)  3 (grade 3)
RANKED = ["c1", "c3", "c2"]
RELEVANCE = {"c1": 1, "c2": 3}


def test_tokenize_and_positions():
    assert tokenize("Hi, Bob! 25 days") == ["hi", "bob", "25", "days"]
    assert token_positions("Hi, Bob") == [(0, 2), (4, 7)]
    assert tokenize(None) == []


def test_average_ignores_missing_values():
    assert average([1, None, 3]) == 2
    assert average([None]) is None
    assert average([]) is None


def test_percentile_interpolates():
    assert percentile([10, 20, 30, 40], 50) == 25
    assert percentile([100, None, 200], 95) == pytest.approx(195)
    assert percentile([], 95) is None
    with pytest.raises(ValueError, match="percent"):
        percentile([1], 101)


def test_cosine_and_jaccard():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0)
    assert cosine([0, 0], [1, 0]) is None
    with pytest.raises(ValueError, match="different sizes"):
        cosine([1], [1, 2])

    assert jaccard("red blue", "blue green") == pytest.approx(1 / 3)
    assert jaccard("", "") is None


def test_ranking_scores_hand_worked():
    scores = ranking_scores(RANKED, RELEVANCE, 3)

    assert scores["precision_at_k"] == pytest.approx(2 / 3)
    assert scores["recall_at_k"] == 1
    assert scores["f1_at_k"] == pytest.approx(0.8)          # 2 * 2/3 * 1 / (2/3 + 1)
    assert scores["hit_at_k"] == 1
    assert scores["mrr_at_k"] == 1

    dcg = 1 / math.log2(2) + 7 / math.log2(4)               # grade 1 at rank 1, grade 3 at rank 3
    idcg = 7 / math.log2(2) + 1 / math.log2(3)              # best order: grade 3, then grade 1
    assert scores["ndcg_at_k"] == pytest.approx(dcg / idcg)
    assert scores["average_precision_at_k"] == pytest.approx((1 / 1 + 2 / 3) / 2)


def test_ranking_scores_cut_at_k():
    scores = ranking_scores(RANKED, RELEVANCE, 2)

    assert scores["precision_at_k"] == 0.5
    assert scores["recall_at_k"] == 0.5
    assert scores["f1_at_k"] == 0.5


def test_ranking_scores_nothing_found():
    scores = ranking_scores(["x", "y"], RELEVANCE, 2)

    assert scores["recall_at_k"] == 0
    assert scores["f1_at_k"] == 0
    assert scores["mrr_at_k"] == 0
    assert scores["ndcg_at_k"] == 0


def test_ranking_scores_without_gold_are_not_measured():
    assert ranking_scores(RANKED, {}, 3) == {key: None for key in RANKING_KEYS}
    assert ranking_scores(RANKED, {"c1": 0}, 3)["recall_at_k"] is None


def test_repeated_id_counts_once():
    assert gains_at_k(["c2", "c2", "c1"], RELEVANCE, 3) == [3, 0, 1]

    scores = ranking_scores(["c2", "c2"], {"c2": 1}, 2)
    assert scores["precision_at_k"] == 0.5
    assert scores["ndcg_at_k"] == 1


def test_first_relevant_rank():
    assert first_relevant_rank(["x", "y", "c2"], RELEVANCE) == 3
    assert first_relevant_rank(["x", "y", "c2"], RELEVANCE, k=2) is None


@pytest.mark.parametrize("k", [0, -1, 1.5, True, "3"])
def test_check_k_rejects_bad_values(k):
    with pytest.raises(ValueError, match="k must be"):
        check_k(k)


def test_check_queries():
    with pytest.raises(ValueError, match="must be a list"):
        check_queries({"id": "q1"})
    with pytest.raises(ValueError, match="Query 2"):
        check_queries([{"id": "q1"}, {"text": "no id"}])
    with pytest.raises(ValueError, match="more than once"):
        check_queries([{"id": "q1"}, {"id": "q1"}])


def test_relevance_of():
    assert relevance_of({"id": "q", "expected_chunk_ids": ["a", "b"]}) == {"a": 1, "b": 1}
    assert relevance_of({"id": "q", "relevance": {"a": 3}, "expected_chunk_ids": ["b"]}) == {"a": 3}
    assert relevance_of({"id": "q"}) == {}
    with pytest.raises(ValueError, match="grade"):
        relevance_of({"id": "q", "relevance": {"a": -1}})
    with pytest.raises(ValueError, match="must be a dict"):
        relevance_of({"id": "q", "relevance": ["a"]})
