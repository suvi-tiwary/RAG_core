"""Tests for the questions loader and the retrieval metrics (RAG handbook 5 and 6)."""

import json
import math

import pytest

from rag_titan.evaluation import (
    load_questions,
    matches_evidence,
    ndcg_at_k,
    parse_evidence_id,
    recall_at_k,
    reciprocal_rank,
)
from rag_titan.evaluation.metrics import evidence_hits


def chunk(name, source, pages=None):
    found = {"id": name, "source": source, "text": name}
    if pages:
        found["page_numbers"] = pages
    return found


# ranks:   1          2          3 (evidence)  4 (same page again)  5 (evidence)
RETRIEVED = [chunk("c1", "a.pdf", [1]), chunk("c2", "b.pdf", [2]), chunk("c3", "a.pdf", [4]),
             chunk("c4", "a.pdf", [4]), chunk("c5", "c.txt")]
REQUIRED = ["a.pdf#p4", "c.txt"]


def test_parse_evidence_id():
    assert parse_evidence_id("hr.pdf") == ("hr.pdf", None)
    assert parse_evidence_id("hr.pdf#p12") == ("hr.pdf", 12)
    for bad in ["", "  ", None, "hr.pdf#4", "hr.pdf#page4", "hr.pdf#p0", "hr.pdf#p-1"]:
        with pytest.raises(ValueError):
            parse_evidence_id(bad)


def test_matches_evidence_by_file_and_page():
    page_4_and_5 = chunk("x", "hr.pdf", [4, 5])
    assert matches_evidence(page_4_and_5, "hr.pdf")
    assert matches_evidence(page_4_and_5, "hr.pdf#p5")
    assert not matches_evidence(page_4_and_5, "hr.pdf#p6")
    assert not matches_evidence(page_4_and_5, "it.pdf")
    assert not matches_evidence(chunk("y", "hr.txt"), "hr.txt#p1")   # no pages known


def test_each_evidence_counts_once():
    assert evidence_hits(RETRIEVED, REQUIRED, k=5) == [0, 0, 1, 0, 1]


def test_recall_at_k():
    assert recall_at_k(RETRIEVED, REQUIRED, k=5) == 1.0
    assert recall_at_k(RETRIEVED, REQUIRED, k=3) == 0.5
    assert recall_at_k(RETRIEVED, REQUIRED, k=2) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RETRIEVED, REQUIRED, k=5) == 1 / 3
    assert reciprocal_rank(RETRIEVED, REQUIRED, k=2) == 0.0


def test_ndcg_at_k_by_hand():
    ideal = 1 + 1 / math.log2(3)                              # 2 evidences at ranks 1 and 2
    found = 1 / math.log2(4) + 1 / math.log2(6)               # found at ranks 3 and 5
    assert ndcg_at_k(RETRIEVED, REQUIRED, k=5) == pytest.approx(found / ideal)   # 0.5438
    assert ndcg_at_k(RETRIEVED, REQUIRED, k=3) == pytest.approx(0.5 / ideal)
    assert ndcg_at_k(RETRIEVED[2:], REQUIRED[:1], k=5) == 1.0   # evidence at rank 1


def test_one_chunk_holding_two_evidences_is_capped_at_one():
    both = [chunk("x", "hr.pdf", [4, 5])]
    assert recall_at_k(both, ["hr.pdf#p4", "hr.pdf#p5"], k=2) == 1.0
    assert ndcg_at_k(both, ["hr.pdf#p4", "hr.pdf#p5"], k=2) == 1.0


def test_nothing_retrieved_scores_zero():
    assert recall_at_k([], REQUIRED, k=5) == 0.0
    assert reciprocal_rank([], REQUIRED, k=5) == 0.0
    assert ndcg_at_k([], REQUIRED, k=5) == 0.0


@pytest.mark.parametrize("bad_k", [0, -2, 1.5, "5", None, True])
def test_bad_k_raises(bad_k):
    with pytest.raises(ValueError, match="k must be"):
        recall_at_k(RETRIEVED, REQUIRED, k=bad_k)


def test_no_required_evidence_raises():
    with pytest.raises(ValueError, match="at least one"):
        recall_at_k(RETRIEVED, [], k=5)


def write(tmp_path, data):
    path = tmp_path / "questions.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


GOOD = {"id": "q_001", "question": "How many leave days?",
        "required_evidence_ids": ["hr.pdf#p4"], "task_type": "fact", "tenant": "demo"}


def test_load_questions_keeps_extra_keys(tmp_path):
    questions = load_questions(write(tmp_path, [GOOD]))
    assert questions == [GOOD]


def test_missing_file_and_bad_json(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_questions(tmp_path / "nope.json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_questions(write(tmp_path, "[{"))


@pytest.mark.parametrize("records, message", [
    ([], "non-empty list"),
    ({"questions": []}, "non-empty list"),
    (["text"], "JSON object"),
    ([{**GOOD, "id": ""}], "'id'"),
    ([{**GOOD, "question": "  "}], "'question'"),
    ([{**GOOD, "required_evidence_ids": []}], "required_evidence_ids"),
    ([{**GOOD, "required_evidence_ids": "hr.pdf"}], "required_evidence_ids"),
    ([{**GOOD, "required_evidence_ids": ["hr.pdf#4"]}], "'#p4'"),
    ([{**GOOD, "task_type": 3}], "task_type"),
    ([GOOD, GOOD], "more than once"),
])
def test_bad_questions_raise_a_clear_error(tmp_path, records, message):
    with pytest.raises(ValueError, match=message):
        load_questions(write(tmp_path, records))
