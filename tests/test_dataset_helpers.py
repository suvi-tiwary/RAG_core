"""Tests for the dataset helpers: profiling, sampling and the golden-set check."""

from pathlib import Path

import pytest

from rag_titan.evaluation import check_golden, profile_documents, sample_documents

EXAMPLES = Path(__file__).parent.parent / "examples"

DOCUMENTS = [
    {"id": "leave", "text": "Leave is 25 days per year for all staff", "category": "hr"},
    {"id": "leave_copy", "text": "leave is 25 days per year, for all staff!", "category": "hr"},
    {"id": "leave_v2", "text": "Leave is 25 days per year for all staff members",
     "category": "hr"},
    {"id": "laptop", "text": "Laptops are replaced every three years by the IT desk",
     "category": "it"},
    {"id": "blank", "text": "   ", "category": "it"},
    {"source": "note.md", "text": "Short note"},
]


def test_profile_hand_worked():
    profile = profile_documents(DOCUMENTS, min_words=5)

    assert profile["document_count"] == 6
    assert profile["total_words"] == 9 + 9 + 10 + 10 + 0 + 2
    assert (profile["min_words"], profile["max_words"]) == (0, 10)
    assert profile["categories"] == {"hr": 3, "it": 2, "uncategorized": 1}
    assert profile["exact_duplicates"] == [["leave", "leave_copy"]]
    assert profile["near_duplicates"] == [
        {"ids": ["leave", "leave_v2"], "similarity": 0.9},        # 9 shared of 10 words
        {"ids": ["leave_copy", "leave_v2"], "similarity": 0.9}]
    assert profile["issues"] == {"blank": ["empty"], "note.md": ["short"],
                                 "leave_copy": ["exact duplicate"],
                                 "leave_v2": ["near duplicate"]}
    assert profile["clean_share"] == pytest.approx(2 / 6)


def test_profile_options_and_dict_input():
    profile = profile_documents({"a": "one two three", "b": "one two four"}, min_words=1,
                                near_duplicate=None)
    assert profile["near_duplicates"] == []
    assert profile["clean_share"] == 1

    loose = profile_documents({"a": "one two three", "b": "one two four"}, min_words=1,
                              near_duplicate=0.5)
    assert loose["near_duplicates"][0]["similarity"] == 0.5


def test_profile_errors():
    with pytest.raises(ValueError, match="non-empty"):
        profile_documents({})
    with pytest.raises(ValueError, match="must be a dict with a 'text'"):
        profile_documents([{"id": "a"}])
    with pytest.raises(ValueError, match="unique 'id'"):
        profile_documents([{"id": "a", "text": "x"}, {"id": "a", "text": "y"}])
    with pytest.raises(ValueError, match="near_duplicate"):
        profile_documents({"a": "x"}, near_duplicate=0)


def test_random_sample_is_repeatable():
    first = sample_documents(DOCUMENTS, 3, seed=7)

    assert first == sample_documents(DOCUMENTS, 3, seed=7)
    assert len(set(first)) == 3
    assert set(sample_documents(DOCUMENTS, 6)) == {"leave", "leave_copy", "leave_v2", "laptop",
                                                   "blank", "note.md"}


def test_stratified_sample_keeps_the_category_shares():
    documents = ([{"id": f"hr{n}", "text": "x", "category": "hr"} for n in range(6)]
                 + [{"id": f"it{n}", "text": "x", "category": "it"} for n in range(3)]
                 + [{"id": "legal0", "text": "x", "category": "legal"}])
    sample = sample_documents(documents, 5, strategy="stratified", seed=1)

    counts = {name: sum(doc_id.startswith(name) for doc_id in sample)
              for name in ("hr", "it", "legal")}
    # exact shares 3.0, 1.5, 0.5: the last seat is a tie, the category with none gets it
    assert counts == {"hr": 3, "it": 1, "legal": 1}
    assert len(sample) == 5


def test_risk_sample_takes_flagged_documents_first():
    flagged = profile_documents(DOCUMENTS, min_words=5)
    sample = sample_documents(DOCUMENTS, 4, strategy="risk", seed=3, profile=flagged)
    assert set(sample) == {"blank", "note.md", "leave_copy", "leave_v2"}

    profile = {"issues": {"laptop": ["short", "near duplicate"], "leave": ["short"]}}
    assert sample_documents(DOCUMENTS, 2, strategy="risk", profile=profile) == ["laptop", "leave"]


def test_sample_errors():
    with pytest.raises(ValueError, match="size must be"):
        sample_documents(DOCUMENTS, 7)
    with pytest.raises(ValueError, match="size must be"):
        sample_documents(DOCUMENTS, 0)
    with pytest.raises(ValueError, match="Unknown strategy 'smart'"):
        sample_documents(DOCUMENTS, 2, strategy="smart")


def test_check_golden_finds_problems():
    documents = {"hr.md": "Leave is 25 days. Sick pay is full pay."}
    golden = [
        {"id": "ok", "question": "Leave?", "expected_evidence": ["Leave is 25 days"],
         "expected_answer": "25 days", "required_facts": ["25"], "validated": True},
        {"id": "bad_quote", "question": "Pay?", "expected_evidence": ["pay"],
         "expected_answer": "Full pay", "required_facts": ["half"]},
        {"id": "bare", "question": "Travel?"},
        {"id": "refuse", "question": "Stock price?", "answerable": False, "validated": True},
        {"id": "odd", "question": "Weather?", "answerable": False,
         "required_evidence_ids": ["hr.md"]},
    ]
    result = check_golden(golden, documents)

    assert result["question_count"] == 5
    assert result["validated_count"] == 2
    assert result["validated_share"] == 0.4
    assert result["ready"] is False
    assert result["problems"] == [
        {"id": "bad_quote", "problem": "required fact 'half' is not in the expected_answer"},
        {"id": "bad_quote", "problem": "quote 'pay' is not found exactly once"},
        {"id": "bare", "problem": "no gold evidence: add expected_evidence quotes"},
        {"id": "bare", "problem": "no expected_answer"},
        {"id": "odd", "problem": "marked unanswerable but has gold evidence"},
    ]


def test_check_golden_ready_needs_validation():
    golden = [{"id": "q1", "question": "Leave?", "expected_evidence": ["Leave is 25 days"],
               "expected_answer": "25 days"}]
    assert check_golden(golden)["ready"] is False            # nobody validated it yet

    golden[0]["validated"] = True
    assert check_golden(golden)["ready"] is True
    assert check_golden(golden, {"hr.md": "No such text."})["ready"] is False


def test_example_golden_set_has_no_problems():
    documents = {path.name: path.read_text(encoding="utf-8")
                 for path in (EXAMPLES / "sample_docs").glob("*.md")}
    result = check_golden(EXAMPLES / "golden_questions.json", documents)

    assert result["problems"] == []
    assert result["validated_share"] == 1
    assert result["ready"] is True
