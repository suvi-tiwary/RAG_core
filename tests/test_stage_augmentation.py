"""Tests for AugmentationEvaluator: the context that reaches the LLM."""

import pytest

from rag_titan.evaluation.stages import AugmentationEvaluator

TEXT = "Leave is 25 days. Sick pay is full. Travel needs approval."
CHUNKS = [
    {"id": "c1", "text": "Leave is 25 days. "},        # 4 words
    {"id": "c2", "text": "Sick pay is full. "},        # 4 words
    {"id": "c3", "text": "Travel needs approval."},    # 3 words
]


def query(**changes):
    record = {"id": "q1", "context_ids": ["c1", "c3"],
              "expected_evidence": ["25 days", "Sick pay is full"]}
    record.update(changes)
    return record


def evaluate(queries, **options):
    options.setdefault("document_text", TEXT)
    return AugmentationEvaluator().evaluate(CHUNKS, queries, **options)


def test_context_scores_hand_worked():
    result = evaluate([query()], max_context_tokens=6)
    scores = result["per_query"][0]

    # context words: 4 + 3 = 7; gold words: 6; shared: "25 days" = 2
    assert scores["context_tokens"] == 7
    assert scores["context_precision"] == pytest.approx(2 / 7)
    assert scores["context_recall"] == pytest.approx(2 / 6)
    assert scores["context_noise"] == pytest.approx(5 / 7)
    assert scores["context_sufficient"] == 0             # "Sick pay is full" is missing
    assert scores["duplicate_ratio"] == 0
    assert scores["budget_compliant"] == 0               # 7 words > budget of 6
    assert result["summary"]["query_count"] == 1
    assert result["warnings"] == []


def test_sufficient_context_and_per_query_budget():
    scores = evaluate([query(context_ids=["c1", "c2"], max_context_tokens=8)],
                      max_context_tokens=1)["per_query"][0]

    assert scores["context_sufficient"] == 1
    assert scores["context_recall"] == 1
    assert scores["budget_compliant"] == 1               # the query's own budget wins


def test_repeated_chunk_raises_duplicate_ratio():
    scores = evaluate([query(context_ids=["c1", "c1"])])["per_query"][0]

    assert scores["context_tokens"] == 8
    assert scores["duplicate_ratio"] == 0.5


def test_evidence_edge_placement():
    first = evaluate([query(context_ids=["c1", "c3"], expected_evidence=["25 days"])])
    assert first["summary"]["evidence_edge_placement"] == 1          # evidence opens the context

    middle = evaluate([query(context_ids=["c3", "c1", "c2"], expected_evidence=["25 days"])])
    assert middle["summary"]["evidence_edge_placement"] == 0         # lost in the middle

    both = evaluate([query(context_ids=["c3", "c1", "c2"])])
    assert both["summary"]["evidence_edge_placement"] == 0.5         # c1 middle, c2 last


def test_without_source_text_uses_gold_chunk_ids():
    record = {"id": "q1", "context_ids": ["c3", "c1", "c2"], "expected_chunk_ids": ["c2"]}
    result = AugmentationEvaluator().evaluate(CHUNKS, [record], max_context_tokens=20)
    scores = result["per_query"][0]

    assert scores["evidence_edge_placement"] == 1
    assert scores["budget_compliant"] == 1
    assert scores["context_precision"] is None
    assert scores["duplicate_ratio"] is None


def test_skipped_queries_and_unknown_ids():
    queries = [query(context_ids=["c1", "zz"]), {"id": "q2", "expected_evidence": ["25 days"]}]
    result = evaluate(queries)

    assert result["summary"]["query_count"] == 1
    assert result["per_query"][0]["missing_chunk_ids"] == ["zz"]


def test_errors():
    with pytest.raises(ValueError, match="must be a list of chunk ids"):
        evaluate([query(context_ids="c1")])
    with pytest.raises(ValueError, match="exactly once"):
        evaluate([query(expected_evidence=["is"])])
