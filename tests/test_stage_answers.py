"""Tests for the answer metrics, GenerationEvaluator and EndToEndEvaluator."""

import pytest

from rag_titan.evaluation.stages import EndToEndEvaluator, GenerationEvaluator
from rag_titan.evaluation.stages.answer_metrics import (
    citation_scores,
    context_grounding,
    exact_match,
    lcs_length,
    required_fact_coverage,
    rouge_l_f1,
    token_scores,
    unsupported_number_rate,
)

ANSWER = "Leave is 25 days per year"       # 6 words
EXPECTED = "Annual leave is 25 days"       # 5 words, 4 shared: leave is 25 days


def test_exact_match_ignores_case_and_punctuation():
    assert exact_match("Hybrid search.", "hybrid search") == 1
    assert exact_match("Hybrid search", "search hybrid") == 0


def test_token_scores_hand_worked():
    scores = token_scores(ANSWER, EXPECTED)

    assert scores["answer_precision"] == pytest.approx(4 / 6)
    assert scores["answer_recall"] == pytest.approx(4 / 5)
    assert scores["answer_f1"] == pytest.approx(8 / 11)


def test_token_scores_count_repeated_words_once_each():
    scores = token_scores("days days days", "25 days")

    assert scores["answer_precision"] == pytest.approx(1 / 3)
    assert scores["answer_recall"] == 0.5


def test_token_scores_with_empty_texts():
    assert token_scores("", "")["answer_f1"] == 1
    assert token_scores("", "25 days")["answer_f1"] == 0
    assert token_scores("no overlap", "25 days")["answer_f1"] == 0


def test_rouge_l_needs_the_same_word_order():
    assert lcs_length(["a", "b", "c", "d"], ["b", "x", "d"]) == 2
    assert rouge_l_f1(ANSWER, EXPECTED) == pytest.approx(8 / 11)      # LCS = 4

    # same words, opposite order: token F1 is 1, but only 1 word stays in order
    assert token_scores("days 25 is leave", "leave is 25 days")["answer_f1"] == 1
    assert rouge_l_f1("days 25 is leave", "leave is 25 days") == 0.25
    assert rouge_l_f1("", "leave") == 0


def test_required_fact_coverage():
    assert required_fact_coverage(ANSWER, ["25 DAYS", "annual"]) == 0.5
    assert required_fact_coverage(ANSWER, []) is None


def test_citation_scores():
    scores = citation_scores(["c1", "c9"], ["c1", "c2"])
    assert (scores["citation_precision"], scores["citation_recall"]) == (0.5, 0.5)
    assert scores["citation_f1"] == 0.5

    nothing_cited = citation_scores([], ["c1"])
    assert nothing_cited["citation_precision"] is None
    assert nothing_cited["citation_recall"] == 0
    assert citation_scores(None, ["c1"])["citation_recall"] is None


def test_grounding_and_unsupported_numbers():
    context = "Leave is 25 days. Up to 1,000 EUR."

    assert context_grounding("Leave is 30 days", context) == 0.75
    assert unsupported_number_rate("Leave is 30 days", context) == 1
    assert unsupported_number_rate("25 days and 1000 EUR, not 40", context) == pytest.approx(1 / 3)
    assert unsupported_number_rate("No numbers here", context) is None
    assert context_grounding("Leave", None) is None


def query(**changes):
    record = {"id": "q1", "answer": ANSWER, "expected_answer": EXPECTED}
    record.update(changes)
    return record


def test_generation_summary():
    queries = [
        query(required_facts=["25 days", "annual"], citations=["c1", "c9"],
              expected_citations=["c1", "c2"], task_success=True, generation_latency_ms=100,
              verified_claims=[{"supported": True}, {"supported": False}],
              answerable=True, abstained=False),
        query(id="q2", answer="I do not know", expected_answer=None, answerable=False,
              abstained=True, task_success=False, generation_latency_ms=300),
        {"id": "q3"},                                      # no answer: not scored
    ]
    summary = GenerationEvaluator().evaluate(queries)["summary"]

    assert summary["query_count"] == 2
    assert summary["exact_match"] == 0                     # only q1 has a reference
    assert summary["answer_f1"] == pytest.approx(8 / 11)
    assert summary["rouge_l_f1"] == pytest.approx(8 / 11)
    assert summary["required_fact_coverage"] == 0.5
    assert summary["citation_precision"] == 0.5
    assert summary["annotated_claim_support"] == 0.5
    assert summary["abstention_accuracy"] == 1             # q1 answered, q2 rightly refused
    assert summary["task_success"] == 0.5
    assert summary["p50_latency_ms"] == 200
    assert summary["context_grounding"] is None            # no context given


def test_generation_grounding_from_context_ids_or_text():
    chunks = [{"id": "c1", "text": "Leave is 25 days."}, {"id": "c2", "text": "Sick pay is full."}]
    queries = [query(answer="Leave is 30 days", context_ids=["c1", "zz"]),
               query(id="q2", answer="Sick pay is full", context="Sick pay is full.")]
    result = GenerationEvaluator().evaluate(queries, chunks=chunks)

    first, second = result["per_query"]
    assert first["context_grounding"] == 0.75
    assert first["unsupported_number_rate"] == 1
    assert second["context_grounding"] == 1
    assert second["unsupported_number_rate"] is None
    assert result["summary"]["context_grounding"] == pytest.approx(0.875)


def test_generation_errors():
    with pytest.raises(ValueError, match="'answer' must be a text"):
        GenerationEvaluator().evaluate([query(answer=None)])
    with pytest.raises(ValueError, match="more than once"):
        GenerationEvaluator().evaluate([query(), query()])


def test_end_to_end_hand_worked():
    queries = [
        query(answer="25 days", expected_answer="25 days", task_success=True, latency_ms=100,
              cost_usd=0.002, total_tokens=900),
        query(id="q2", task_success=False, latency_ms=300, cost_usd=0.004),
        {"id": "q3", "latency_ms": 9999},                  # no answer: not counted
    ]
    scores = EndToEndEvaluator().evaluate(queries)

    assert scores["query_count"] == 2
    assert scores["answer_exact_match"] == 0.5
    assert scores["answer_f1"] == pytest.approx((1 + 8 / 11) / 2)
    assert scores["task_success_rate"] == 0.5
    assert scores["p50_latency_ms"] == 200
    assert scores["p95_latency_ms"] == pytest.approx(290)
    assert scores["mean_cost_usd"] == pytest.approx(0.003)
    assert scores["mean_total_tokens"] == 900


def test_end_to_end_without_data_and_bad_telemetry():
    scores = EndToEndEvaluator().evaluate([])
    assert scores["query_count"] == 0
    assert scores["answer_f1"] is None
    assert scores["p95_latency_ms"] is None

    with pytest.raises(ValueError, match="latency_ms"):
        EndToEndEvaluator().evaluate([query(latency_ms=-5)])
