"""Tests for RAGEvaluator (all stages in one call) and RootCauseAnalyzer."""

import json

import pytest

from rag_titan.evaluation import RAGEvaluator, RootCauseAnalyzer
from rag_titan.evaluation.rag_evaluator import STAGES


def test_full_pipeline_report(rag_dataset):
    report = RAGEvaluator(k=3).evaluate(**rag_dataset)

    assert list(report) == STAGES + ["root_causes", "scorecard"]
    assert report["chunking"]["coverage"] == 1
    assert report["chunking"]["evidence_completeness"] == 1
    assert report["chunking"]["boundary_f1"] == 1
    assert report["chunking"]["size_compliance"] == 1
    assert report["embedding"]["query_topk_recall"] == 1
    assert report["retrieval"]["summary"]["recall_at_k"] == 1
    assert report["retrieval"]["summary"]["mrr_at_k"] == 0.75         # ranks 2 and 1
    assert report["retrieval"]["summary"]["p95_latency_ms"] == pytest.approx(39)
    assert report["reranking"]["summary"]["mrr_gain"] == 0.25
    assert report["augmentation"]["summary"]["context_sufficient"] == 1
    assert report["generation"]["summary"]["answer_f1"] == 1
    assert report["generation"]["summary"]["context_grounding"] == 1
    assert report["end_to_end"]["task_success_rate"] == 1
    assert report["root_causes"]["first_failing_stage"] is None
    assert report["root_causes"]["failures"] == []


def test_report_can_be_saved_as_json(rag_dataset):
    report = RAGEvaluator(k=3).evaluate(**rag_dataset)

    assert json.loads(json.dumps(report)) == report


def test_chunk_vectors_default_to_the_embeddings(rag_dataset):
    report = RAGEvaluator(k=3).evaluate(**rag_dataset)
    assert report["chunking"]["semantic_chunk_stickiness"] is not None

    del rag_dataset["embeddings"]
    report = RAGEvaluator(k=3).evaluate(**rag_dataset)
    assert report["chunking"]["semantic_chunk_stickiness"] is None
    assert "embedding" not in report


def test_only_stages_with_data_are_evaluated():
    queries = [{"id": "q1", "expected_chunk_ids": ["c1"], "retrieved_ids": ["c2", "c1"]}]
    report = RAGEvaluator(k=2).evaluate(queries=queries)

    assert list(report) == ["retrieval", "root_causes", "scorecard"]
    assert report["retrieval"]["summary"]["recall_at_k"] == 1

    assert "chunking" in RAGEvaluator().evaluate(chunks=["one chunk"])
    assert list(RAGEvaluator().evaluate()) == ["root_causes", "scorecard"]


def test_first_failing_stage_follows_pipeline_order(rag_dataset):
    for query in rag_dataset["queries"]:
        query["retrieved_ids"] = ["c3"]                    # the retriever misses the gold chunk
        query["answer"] = "I do not know."                 # ... so the answer is wrong too
    causes = RAGEvaluator(k=3).evaluate(**rag_dataset)["root_causes"]

    assert causes["first_failing_stage"] == "retrieval"
    assert [(item["stage"], item["metric"]) for item in causes["failures"]] == [
        ("retrieval", "recall_at_k"), ("generation", "answer_f1")]
    assert causes["failures"][0]["observed"] == 0
    assert causes["failures"][0]["minimum"] == 0.8
    assert "hybrid" in causes["failures"][0]["suggestion"]


def test_thresholds_can_be_changed(rag_dataset):
    report = RAGEvaluator(k=1, thresholds={"recall_at_k": 0.4}).evaluate(**rag_dataset)
    assert report["retrieval"]["summary"]["recall_at_k"] == 0.5        # q1's gold chunk is rank 2
    assert report["root_causes"]["first_failing_stage"] is None

    strict = RootCauseAnalyzer({"recall_at_k": 0.9}).evaluate(report)
    assert strict["first_failing_stage"] == "retrieval"


def test_root_cause_skips_metrics_that_were_not_measured():
    causes = RootCauseAnalyzer().evaluate({"chunking": {"coverage": None},
                                           "generation": {"summary": {"answer_f1": 0.5}}})

    assert causes["first_failing_stage"] == "generation"
    assert len(causes["failures"]) == 1


def test_errors(rag_dataset):
    with pytest.raises(ValueError, match="k must be"):
        RAGEvaluator(k=0)
    with pytest.raises(ValueError, match="Unknown threshold 'recal'"):
        RAGEvaluator(thresholds={"recal": 0.5})
    with pytest.raises(ValueError, match="must be a number"):
        RootCauseAnalyzer({"recall_at_k": "high"})
    with pytest.raises(ValueError, match=r"Unknown input \['expected_evidnce'\]"):
        RAGEvaluator().evaluate(chunks=["a"], expected_evidnce=["a"])
    with pytest.raises(ValueError, match="dict with 'vectors'"):
        RAGEvaluator().evaluate(embeddings={"query_vectors": {}})
    with pytest.raises(ValueError, match="Unknown embeddings key"):
        RAGEvaluator().evaluate(embeddings={"vectors": {}, "vektors": {}})
    with pytest.raises(ValueError, match="report must be a dict"):
        RootCauseAnalyzer().evaluate([])
