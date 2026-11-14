"""Tests for RetrievalEvaluator, RerankingEvaluator and the span helpers they use."""

import pytest

from rag_titan.evaluation.stages import RerankingEvaluator, RetrievalEvaluator
from rag_titan.evaluation.stages.spans import as_documents, evidence_spans, normalize_chunks

TEXT = "Leave is 25 days. Sick pay is full. Travel needs approval."
CHUNKS = [
    {"id": "c1", "text": "Leave is 25 days. "},
    {"id": "c2", "text": "Sick pay is full. "},
    {"id": "c3", "text": "Travel needs approval."},
]


def query(**changes):
    record = {"id": "q1", "expected_chunk_ids": ["c1", "c2"], "retrieved_ids": ["c1", "c3"],
              "expected_evidence": ["25 days", "Sick pay is full"]}
    record.update(changes)
    return record


def test_retrieval_ranking_metrics_need_no_source_text():
    result = RetrievalEvaluator().evaluate(CHUNKS, [query()], k=2)
    summary = result["summary"]

    assert summary["query_count"] == 1
    assert summary["precision_at_k"] == 0.5
    assert summary["recall_at_k"] == 0.5
    assert summary["hit_at_k"] == 1
    assert summary["token_recall"] is None         # no source text given
    assert summary["p95_latency_ms"] is None
    assert result["warnings"] == []


def test_retrieval_token_metrics_hand_worked():
    result = RetrievalEvaluator().evaluate(CHUNKS, [query()], k=2, document_text=TEXT)
    scores = result["per_query"][0]

    # gold words: 25 days Sick pay is full (6); retrieved words: c1 (4) + c3 (3) = 7; shared: 2
    assert scores["token_recall"] == pytest.approx(2 / 6)
    assert scores["token_precision"] == pytest.approx(2 / 7)
    assert scores["token_iou"] == pytest.approx(2 / 11)
    assert scores["complete_evidence_recall"] == 0.5
    assert scores["missing_chunk_ids"] == []


def test_retrieval_with_several_documents():
    documents = {"leave.md": "Leave is 25 days.", "sick.md": "Sick pay is full."}
    chunks = [{"id": "c1", "doc_id": "leave.md", "text": "Leave is 25 days."},
              {"id": "c2", "doc_id": "sick.md", "start": 0, "end": 17}]
    record = query(retrieved_ids=["c2"], expected_evidence=[
        "25 days", {"doc_id": "sick.md", "start": 0, "end": 8}])

    result = RetrievalEvaluator().evaluate(chunks, [record], k=1, documents=documents)
    scores = result["per_query"][0]

    assert scores["token_recall"] == pytest.approx(2 / 4)    # "Sick pay" of 25 days Sick pay
    assert scores["token_precision"] == pytest.approx(2 / 4)
    assert scores["complete_evidence_recall"] == 0.5


def test_retrieval_latency_and_skipped_queries():
    queries = [query(id="q1", retrieval_latency_ms=10), query(id="q2", retrieval_latency_ms=30),
               {"id": "q3", "expected_chunk_ids": ["c1"]}]       # no retrieved_ids: not scored
    summary = RetrievalEvaluator().evaluate(CHUNKS, queries, k=2)["summary"]

    assert summary["query_count"] == 2
    assert summary["p50_latency_ms"] == 20
    assert summary["p95_latency_ms"] == pytest.approx(29)


def test_retrieval_reports_unknown_ids_and_unplaced_chunks():
    chunks = CHUNKS + [{"id": "c4", "text": "not in the document"}]
    result = RetrievalEvaluator().evaluate(chunks, [query(retrieved_ids=["zz", "c1"])], k=2,
                                           document_text=TEXT)

    assert result["per_query"][0]["missing_chunk_ids"] == ["zz"]
    assert result["warnings"] == ["c4: text not found exactly once; give start and end"]


def test_retrieval_errors():
    evaluator = RetrievalEvaluator()
    with pytest.raises(ValueError, match="k must be"):
        evaluator.evaluate(CHUNKS, [query()], k=0)
    with pytest.raises(ValueError, match="more than once"):
        evaluator.evaluate(CHUNKS + [{"id": "c1", "text": "x"}], [query()])
    with pytest.raises(ValueError, match="must be a list of chunk ids"):
        evaluator.evaluate(CHUNKS, [query(retrieved_ids="c1")])
    with pytest.raises(ValueError, match="exactly once"):
        evaluator.evaluate(CHUNKS, [query(expected_evidence=["is"])], document_text=TEXT)
    with pytest.raises(ValueError, match="not both"):
        evaluator.evaluate(CHUNKS, [query()], document_text=TEXT, documents={"a": TEXT})


def test_normalize_chunks():
    documents = as_documents(document_text=TEXT)
    items, warnings = normalize_chunks(["Leave is 25 days. ", {"id": "b", "start": 18, "end": 36}],
                                       documents)

    assert warnings == []
    assert [(item["id"], item["start"], item["end"]) for item in items] == [
        ("chunk_1", 0, 18), ("b", 18, 36)]
    assert items[1]["text"] == "Sick pay is full. "

    with pytest.raises(ValueError, match="does not match"):
        normalize_chunks([{"id": "a", "start": 0, "end": 5, "text": "Sick "}], documents)
    with pytest.raises(ValueError, match="fit inside"):
        normalize_chunks([{"id": "a", "start": 5, "end": 500}], documents)
    with pytest.raises(ValueError, match="needs a 'doc_id'"):
        normalize_chunks(["x"], {"a": "x", "b": "y"})
    with pytest.raises(ValueError, match="unknown doc_id"):
        normalize_chunks([{"id": "a", "doc_id": "zz", "text": "x"}], {"a": "x", "b": "y"})


def test_evidence_spans_errors():
    documents = {"a": "same words", "b": "same words"}
    with pytest.raises(ValueError, match="exactly once"):
        evidence_spans(["same"], documents)                 # found in two documents
    with pytest.raises(ValueError, match="does not fit"):
        evidence_spans([{"doc_id": "a", "start": 4, "end": 4}], documents)


def test_reranking_hand_worked():
    record = {"id": "q1", "expected_chunk_ids": ["c2"],
              "retrieved_ids": ["c1", "c3", "c2"], "reranked_ids": ["c2", "c1", "c3"]}
    result = RerankingEvaluator().evaluate([record], k=2)
    scores, summary = result["per_query"][0], result["summary"]

    assert (scores["before_mrr"], scores["after_mrr"]) == (0, 1)     # rank 3 is outside k=2
    assert summary["mrr_gain"] == 1
    assert summary["ndcg_at_k"] == 1
    assert summary["ndcg_gain"] == 1
    assert summary["precision_gain"] == 0.5
    assert summary["first_relevant_rank_change"] == 2                # rank 3 -> rank 1
    assert summary["candidate_set_preservation_rate"] == 1


def test_reranking_that_hurts_and_drops_a_chunk():
    record = {"id": "q1", "relevance": {"c1": 2}, "retrieved_ids": ["c1", "c2"],
              "reranked_ids": ["c2"]}
    summary = RerankingEvaluator().evaluate([record], k=2)["summary"]

    assert summary["mrr_gain"] == -1
    assert summary["first_relevant_rank_change"] is None     # not in the reranked list any more
    assert summary["candidate_set_preservation_rate"] == 0


def test_reranking_without_retrieved_ids_or_reranked_ids():
    queries = [{"id": "q1", "expected_chunk_ids": ["c1"], "reranked_ids": ["c1"]},
               {"id": "q2", "expected_chunk_ids": ["c1"], "retrieved_ids": ["c1"]}]   # skipped
    result = RerankingEvaluator().evaluate(queries, k=1)

    assert result["summary"]["query_count"] == 1
    assert result["summary"]["mrr_at_k"] == 1
    assert result["summary"]["mrr_gain"] is None
    assert result["summary"]["candidate_set_preservation_rate"] is None
