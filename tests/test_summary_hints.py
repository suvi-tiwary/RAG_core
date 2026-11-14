"""Tests for the clearer terminal summary: queries to look at, not-measured hints, latency noise."""

from rag_titan.evaluation import RAGEvaluator, ReportManager, build_dataset
from rag_titan.evaluation.reports import find_regressions, summary_text
from rag_titan.evaluation.reports.hints import not_measured, queries_to_look_at

DOC = "Economy class is required under six hours.\n\nThe meal allowance is 45 euros.\n"
CHUNKS = [{"id": "p1", "text": "Economy class is required under six hours."},
          {"id": "p2", "text": "The meal allowance is 45 euros."}]
GOLDEN = [{"id": "q1", "question": "Meals?", "expected_answer": "The meal allowance is 45 euros.",
           "expected_evidence": ["The meal allowance is 45 euros."], "validated": True},
          {"id": "q2", "question": "Business class on a short flight?",
           "expected_answer": "No, economy class is required under six hours.",
           "expected_evidence": ["Economy class is required under six hours."],
           "validated": True}]
RUNS = [{"id": "q1", "retrieved_ids": ["p2"], "context_ids": ["p2"],
         "answer": "The meal allowance is 45 euros."},
        {"id": "q2", "retrieved_ids": ["p1"], "context_ids": ["p1"],
         "answer": "Yes, business is fine."}]


def report_for(data):
    return RAGEvaluator(k=1).evaluate(**data)


def bare_data():
    data = build_dataset(CHUNKS, RUNS, GOLDEN, {"policy.md": DOC})
    return {"chunks": CHUNKS, "queries": [{k: v for k, v in q.items() if k != "expected_evidence"}
                                          for q in data["queries"]]}


def test_wrong_answer_is_listed_in_the_summary():
    text = summary_text(report_for(bare_data()))
    assert "Queries to look at:\n  - q2 (generation): answer differs from the reference" in text
    assert "q1 (" not in text


def test_nulls_explain_themselves():
    hints = not_measured(report_for(bare_data()))
    assert any(hint.startswith("semantic indexing") for hint in hints)
    assert any(hint.startswith("context and evidence checks") for hint in hints)
    assert "Not measured (null in the report)" in summary_text(report_for(bare_data()))


def test_build_dataset_removes_the_documents_hint():
    full = report_for(build_dataset(CHUNKS, RUNS, GOLDEN, {"policy.md": DOC}))
    assert not any(hint.startswith("context and evidence") for hint in not_measured(full))


def test_long_lists_are_cut():
    report = {"generation": {"per_query": [{"query_id": f"q{i}", "answer_f1": 0.1}
                                           for i in range(8)]}}
    lines = queries_to_look_at(report, limit=3)
    assert len(lines) == 4 and lines[-1] == "... and 5 more (see the .md report)"


def test_one_line_per_query_earliest_stage_first():
    report = {"retrieval": {"per_query": [{"query_id": "q1", "hit_at_k": 0}]},
              "generation": {"per_query": [{"query_id": "q1", "answer_f1": 0.0}]}}
    assert queries_to_look_at(report) == ["q1 (retrieval): no relevant chunk in the top K"]


def latency(value, name="p50_latency_ms"):
    return {"retrieval": {"summary": {name: value}}}


def test_tiny_latency_changes_are_noise():
    assert find_regressions(latency(7.55), latency(7.2)) == []          # +0.35 ms
    assert find_regressions(latency(1080), latency(1000)) == []         # +8%


def test_real_slowdowns_still_count():
    assert find_regressions(latency(20.0), latency(7.2))[0]["metric"] == "retrieval.p50_latency_ms"
    assert len(find_regressions(latency(1200), latency(1000))) == 1      # +20%, +200 ms


def test_report_manager_passes_the_latency_rule_on():
    manager = ReportManager()
    assert manager.regressions(latency(7.55), latency(7.2)) == []
    strict = manager.regressions(latency(7.55), latency(7.2), latency_share=0, latency_min_ms=0)
    assert strict[0]["metric"] == "retrieval.p50_latency_ms"
