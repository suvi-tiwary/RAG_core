"""Tests for evaluate_retriever, compare_retrievers and format_report."""

import json
from pathlib import Path

import pytest

from rag_titan.evaluation import (
    compare_retrievers,
    evaluate_retriever,
    format_report,
    load_questions,
)
from rag_titan.evaluation.benchmark import summarize_rows
from rag_titan.retrieval import BaseRetriever, DenseRetriever
from rag_titan.vector_stores import MemoryStore

# the fixture chunks: chunk_0 + chunk_1 come from hr.pdf, chunk_2 from it.pdf
QUESTIONS = [
    {"id": "q1", "question": "leave days", "required_evidence_ids": ["hr.pdf"]},
    {"id": "q2", "question": "remote work", "required_evidence_ids": ["it.pdf"]},
    {"id": "q3", "question": "salary", "required_evidence_ids": ["it.pdf"], "task_type": "fact"},
]


class EmptyRetriever(BaseRetriever):
    """Has documents but never finds anything."""

    name = "empty"

    def has_documents(self):
        return True

    def retrieve(self, query, top_k=5):
        return []


def test_evaluate_retriever_scores_every_question(parts):
    result = evaluate_retriever(parts["retriever"], QUESTIONS, top_k=1)

    assert result["name"] == "dense" and result["top_k"] == 1
    assert [row["id"] for row in result["rows"]] == ["q1", "q2", "q3"]
    assert [row["recall"] for row in result["rows"]] == [1.0, 1.0, 0.0]   # q3 finds hr.pdf
    assert result["rows"][0]["sources"] == ["hr.pdf"]
    assert result["rows"][2]["task_type"] == "fact"
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["mrr"] == pytest.approx(2 / 3)
    assert 0 <= result["p50_ms"] <= result["p95_ms"]


def test_compare_uses_the_same_questions_and_top_k(parts):
    results = compare_retrievers({"dense": parts["retriever"], "empty": EmptyRetriever()},
                                 QUESTIONS, top_k=2)

    assert [result["name"] for result in results] == ["dense", "empty"]
    assert all(result["top_k"] == 2 and len(result["rows"]) == 3 for result in results)
    assert results[1]["recall"] == results[1]["mrr"] == results[1]["ndcg"] == 0.0


def test_latency_percentiles():
    rows = [{"recall": 1, "reciprocal_rank": 1, "ndcg": 1, "ms": ms} for ms in range(1, 101)]
    summary = summarize_rows(rows)
    assert summary["p50_ms"] == pytest.approx(50.5)
    assert summary["p95_ms"] == pytest.approx(95.05)


def test_format_report(parts):
    results = compare_retrievers({"dense": parts["retriever"], "empty": EmptyRetriever()},
                                 QUESTIONS, top_k=2)
    lines = format_report(results).splitlines()

    assert lines[0] == "Retrieval report: 3 questions, top_k=2"
    assert lines[2].split() == ["retriever", "Recall@2", "MRR", "nDCG@2", "p50", "ms", "p95", "ms"]
    assert lines[4].split()[:4] == ["dense", "0.667", "0.667", "0.667"]   # q3 is missed
    assert lines[5].split()[1:4] == ["0.000", "0.000", "0.000"]


@pytest.mark.parametrize("bad", [0, -1, 2.5, None])
def test_bad_top_k(parts, bad):
    with pytest.raises(ValueError, match="top_k"):
        evaluate_retriever(parts["retriever"], QUESTIONS, top_k=bad)


def test_bad_inputs_raise_clear_errors(parts):
    with pytest.raises(ValueError, match="index your files"):
        evaluate_retriever(DenseRetriever(parts["embedder"], MemoryStore()), QUESTIONS)
    with pytest.raises(ValueError, match="non-empty list"):
        evaluate_retriever(parts["retriever"], [])
    with pytest.raises(ValueError, match="non-empty dict"):
        compare_retrievers({}, QUESTIONS)
    with pytest.raises(ValueError, match="no results"):
        format_report([])


def test_example_questions_match_the_example_files():
    examples = Path(__file__).parent.parent / "examples"
    files = {path.name for path in (examples / "sample_docs").glob("*.md")}

    questions = load_questions(examples / "questions.json")
    for question in questions:
        for label in question["required_evidence_ids"]:
            assert label.split("#")[0] in files, f"{question['id']}: {label} is not in sample_docs/"
    assert json.loads((examples / "questions.json").read_text())   # valid JSON
