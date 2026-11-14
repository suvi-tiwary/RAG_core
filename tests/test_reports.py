"""Tests for the reports: JSON, Markdown, regressions, trends and failed queries."""

import copy

import pytest

from rag_titan.evaluation import RAGEvaluator, ReportManager
from rag_titan.evaluation.reports import failed_queries, find_regressions, flatten, trends
from rag_titan.evaluation.reports.compare import better_direction


@pytest.fixture
def report(rag_dataset):
    return RAGEvaluator(k=3).evaluate(**rag_dataset)


@pytest.fixture
def broken_report(rag_dataset):
    """The same run, but the retriever misses q1 and its answer is wrong."""
    query = rag_dataset["queries"][0]
    query.update(retrieved_ids=["c3", "c2"], context_ids=["c3"], answer="Leave is 30 days.",
                 task_success=False, latency_ms=2000)
    return RAGEvaluator(k=3).evaluate(**rag_dataset)


def test_json_round_trip(report, tmp_path):
    manager = ReportManager()
    path = manager.save_json(report, tmp_path / "runs" / "today.json")

    assert path.is_file()
    assert manager.load_json(path) == report


def test_load_json_errors(tmp_path):
    manager = ReportManager()
    with pytest.raises(FileNotFoundError, match="Report file not found"):
        manager.load_json(tmp_path / "missing.json")

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        manager.load_json(bad)

    bad.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="does not hold a report"):
        manager.load_json(bad)


def test_flatten_keeps_numeric_stage_metrics(report):
    flat = flatten(report)

    assert flat["retrieval.recall_at_k"] == 1
    assert flat["chunking.coverage"] == 1
    assert flat["end_to_end.p95_latency_ms"] == pytest.approx(877.5)
    assert not any(name.startswith(("scorecard", "root_causes")) for name in flat)
    assert "chunking.warnings" not in flat


def test_better_direction_comes_from_the_policy():
    assert better_direction("retrieval.recall_at_k") == "higher"
    assert better_direction("augmentation.context_noise") == "lower"
    assert better_direction("embedding.negative_similarity") == "lower"
    assert better_direction("end_to_end.mean_cost_usd") == "lower"
    assert better_direction("chunking.chunk_count") is None
    assert better_direction("chunking.mean_chunk_tokens") is None


def test_regressions_in_both_directions(report, broken_report):
    found = {item["metric"]: item for item in find_regressions(broken_report, report)}

    assert found["retrieval.recall_at_k"]["delta"] == -0.5          # higher is better: went down
    assert found["end_to_end.p95_latency_ms"]["delta"] > 0          # lower is better: went up
    assert found["generation.unsupported_number_rate"]["current"] == 0.5
    assert "chunking.coverage" not in found                         # unchanged
    assert "chunking.chunk_count" not in found                      # no better direction

    # the other way round only the reranking gains shrink: the fixed retriever
    # leaves the reranker less to repair. Improvements are not listed.
    back = [item["metric"] for item in find_regressions(report, broken_report)]
    assert back and all(name.startswith("reranking.") and name.endswith("_gain") for name in back)
    assert find_regressions(report, report) == []


def test_regression_tolerance():
    old = {"retrieval": {"summary": {"recall_at_k": 0.80}}, "end_to_end": {"p95_latency_ms": 1000}}
    new = {"retrieval": {"summary": {"recall_at_k": 0.79}}, "end_to_end": {"p95_latency_ms": 1015}}

    assert find_regressions(new, old) == []                         # within 0.02 and within 2%
    assert [r["metric"] for r in find_regressions(new, old, tolerance=0)] == \
        ["retrieval.recall_at_k"]                                   # +15 ms is latency noise
    strict = find_regressions(new, old, tolerance=0, latency_share=0, latency_min_ms=0)
    assert len(strict) == 2
    with pytest.raises(ValueError, match="tolerance"):
        find_regressions(new, old, tolerance=-1)


def test_trends(report, broken_report):
    changes = trends([broken_report, report, report])

    recall = changes["retrieval.recall_at_k"]
    assert (recall["first"], recall["last"], recall["change"]) == (0.5, 1, 0.5)
    assert recall["improved"] is True
    assert recall["runs"] == 3
    assert changes["end_to_end.p95_latency_ms"]["improved"] is True     # it went down
    assert changes["chunking.coverage"]["improved"] is None             # no change
    assert trends([report]) == {}


def test_failed_queries(report, broken_report):
    assert failed_queries(report) == []

    found = failed_queries(broken_report)
    assert [item["stage"] for item in found] == [
        "retrieval", "augmentation", "generation", "generation"]
    assert {item["query_id"] for item in found} == {"q1"}
    assert found[0]["reason"] == "no relevant chunk in the top K"


def test_markdown_of_a_healthy_report(report):
    text = ReportManager().markdown(report, generated="2026-11-12")

    assert text.startswith("# RAG Pipeline Evaluation\n\nGenerated: 2026-11-12\n")
    assert "## Executive scorecard" in text
    assert "| Information search |" in text
    assert "## Retrieval (Information search)" in text
    assert "| recall_at_k | 1.0000 |" in text
    assert "| chunk_count | 4 |" in text
    assert "No threshold failure was detected." in text
    assert "## Queries to look at" not in text
    assert "## Regressions" not in text


def test_markdown_of_a_broken_report(report, broken_report, tmp_path):
    manager = ReportManager()
    broken = copy.deepcopy(broken_report)
    broken["regressions"] = manager.regressions(broken, report)
    path = manager.save_markdown(broken, tmp_path / "out" / "report.md", title="Nightly run")
    text = path.read_text(encoding="utf-8")

    assert text.startswith("# Nightly run")
    assert "**Assessment: Action Required**" in text
    assert "- **Quality check:** Information search: recall at k is below" in text
    assert "## Regressions against the baseline" in text
    assert "| retrieval.recall_at_k | 1.0000 | 0.5000 | -0.5000 |" in text
    assert "- `q1` (retrieval): no relevant chunk in the top K" in text
    assert "First failing stage: **retrieval**" in text


def test_markdown_without_regressions_found(report):
    report["regressions"] = []
    assert "No metric got worse than the baseline." in ReportManager().markdown(report)

    with pytest.raises(ValueError, match="must be a dict"):
        ReportManager().markdown([])
