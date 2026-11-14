"""
Refusals ("I don't know") are scored by Abstention Accuracy, not as wrong answers.

A question the documents cannot answer has no reference answer, and a refusal
makes no claims to ground. Before 0.8.5 a correct refusal got Answer F1 0 and a
low Context Grounding, so a RAG that learned to refuse looked like a regression.
"""

import pytest

from rag_titan.evaluation import RAGEvaluator
from rag_titan.evaluation.reports.failures import failed_queries
from rag_titan.evaluation.stages import EndToEndEvaluator, GenerationEvaluator


def query(**changes):
    record = {"id": "q1", "answer": "Leave is 25 days per year",
              "expected_answer": "Annual leave is 25 days"}
    record.update(changes)
    return record


def test_refusals_are_scored_by_abstention_not_as_wrong_answers():
    context = "Leave is 25 days per year"
    queries = [
        # rightly refused: no reference to compare with, nothing to ground
        query(id="refused", answer="I don't know.", expected_answer="", answerable=False,
              abstained=True, context=context),
        # made something up for an unanswerable question: caught twice
        query(id="invented", answer="The gym allowance is 50 euros", expected_answer="",
              answerable=False, abstained=False, context=context),
        # refused although the documents answer it: still a wrong answer
        query(id="gave_up", answer="I don't know.", answerable=True, abstained=True,
              context=context),
    ]
    refused, invented, gave_up = GenerationEvaluator().evaluate(queries)["per_query"]

    assert refused["abstention_accuracy"] == 1
    assert refused["answer_f1"] is None and refused["exact_match"] is None
    assert refused["context_grounding"] is None
    assert refused["unsupported_number_rate"] is None

    assert invented["abstention_accuracy"] == 0
    assert invented["answer_f1"] is None                   # there is no reference answer
    assert invented["context_grounding"] == pytest.approx(1 / 6)    # only "is" is in the context
    assert invented["unsupported_number_rate"] == 1        # "50" is not in the context

    assert gave_up["abstention_accuracy"] == 0
    assert gave_up["answer_f1"] == 0                       # compared with the reference
    assert gave_up["context_grounding"] is None


def test_end_to_end_skips_the_reference_of_unanswerable_questions():
    queries = [query(answer="25 days", expected_answer="25 days"),
               query(id="q2", answer="I don't know.", expected_answer="", answerable=False)]
    scores = EndToEndEvaluator().evaluate(queries)

    assert scores["query_count"] == 2
    assert scores["answer_f1"] == 1                        # q2 has no reference to miss
    assert scores["answer_exact_match"] == 1


def test_a_correct_refusal_is_not_a_query_to_look_at():
    chunks = [{"id": "c1", "text": "Leave is 25 days per year."}]
    gym = {"id": "gym", "answer": "I don't know.", "expected_answer": "", "answerable": False,
           "abstained": True, "context_ids": ["c1"], "retrieved_ids": ["c1"]}
    report = RAGEvaluator(k=1).evaluate(chunks=chunks, queries=[gym])
    assert failed_queries(report) == []

    gym.update(answer="The gym allowance is 50 euros.", abstained=False)
    report = RAGEvaluator(k=1).evaluate(chunks=chunks, queries=[gym])
    assert [(item["query_id"], item["stage"]) for item in failed_queries(report)] == \
        [("gym", "generation")]
