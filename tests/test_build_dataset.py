"""Tests for build_dataset(): logs of any RAG + golden questions -> evaluation data."""

import json

import pytest

from rag_titan import cli
from rag_titan.evaluation import RAGEvaluator, build_dataset

POLICY = ("# Travel policy\n\n"
          "Economy class is required for all flights under six hours.\n\n"
          "The daily meal allowance is 45 euros.\n\n"
          "Expense reports must be submitted within 30 days.\n")
DOCS = {"policy.md": POLICY}
CHUNKS = [{"id": "p1", "text": "Economy class is required for all flights under six hours."},
          {"id": "p2", "text": "The daily meal allowance is 45 euros."},
          {"id": "p3", "text": "Expense reports must be submitted within 30 days."}]
GOLDEN = [
    {"id": "q1", "question": "How much is the meal allowance?",
     "expected_answer": "The daily meal allowance is 45 euros.",
     "expected_evidence": ["The daily meal allowance is 45 euros."],
     "required_facts": ["45"], "answerable": True, "validated": True},
    {"id": "q2", "question": "Can I fly business on a four-hour flight?",
     "expected_answer": "No, economy class is required under six hours.",
     "expected_evidence": ["Economy class is required for all flights under six hours."],
     "required_facts": ["economy"], "answerable": True, "validated": True},
]
RUNS = [
    {"question": "How much is the meal allowance?", "retrieved_ids": ["p2", "p1"],
     "context_ids": ["p2", "p1"], "answer": "It is 45 euros a day [p2]."},
    {"id": "q2", "retrieved_ids": ["p3", "p2"], "context_ids": ["p3", "p2"],
     "answer": "Yes, you can.", "generation_latency_ms": 120.0},
]


def build(**changes):
    args = {"chunks": CHUNKS, "runs": RUNS, "golden": GOLDEN, "documents": DOCS}
    args.update(changes)
    return build_dataset(**args)


def test_chunks_get_their_place_in_the_document():
    data = build()
    first = data["chunks"][0]
    assert first["doc_id"] == "policy.md"
    assert POLICY[first["start"]:first["end"]] == first["text"]
    assert data["documents"] == DOCS


def test_labels_come_from_golden_and_outputs_from_the_run():
    q1, q2 = build()["queries"]
    assert q1["id"] == "q1" and q1["text"] == "How much is the meal allowance?"
    assert q1["expected_chunk_ids"] == ["p2"] and q1["expected_citations"] == ["p2"]
    # q2's search missed p1: the label still says p1, it is never copied from the run
    assert q2["retrieved_ids"] == ["p3", "p2"] and q2["expected_chunk_ids"] == ["p1"]
    assert q2["generation_latency_ms"] == 120.0
    assert q2["required_facts"] == ["economy"]


def test_citation_markers_become_citations():
    q1 = build()["queries"][0]
    assert q1["answer"] == "It is 45 euros a day." and q1["citations"] == ["p2"]
    assert q1["raw_answer"] == "It is 45 euros a day [p2]."


def test_given_citations_are_kept():
    runs = [dict(RUNS[0], citations=["p1"]), RUNS[1]]
    q1 = build(runs=runs)["queries"][0]
    assert q1["citations"] == ["p1"] and q1["answer"] == "It is 45 euros a day [p2]."


def test_all_quotes_are_listed_once_for_chunking():
    assert build()["expected_evidence"] == [GOLDEN[0]["expected_evidence"][0],
                                           GOLDEN[1]["expected_evidence"][0]]


def test_more_measured_than_without_documents():
    full = RAGEvaluator(k=2).evaluate(**build())
    bare = RAGEvaluator(k=2).evaluate(
        chunks=[{"id": c["id"], "text": c["text"]} for c in CHUNKS],
        queries=[{k: v for k, v in q.items() if k != "expected_evidence"}
                 for q in build()["queries"]])
    full_cov = full["scorecard"]["overall"]["evaluation_coverage"]
    assert full_cov > bare["scorecard"]["overall"]["evaluation_coverage"]
    assert full["augmentation"]["summary"]["context_sufficient"] is not None
    assert full["chunking"]["evidence_completeness"] == 1.0


def test_works_with_the_cli(tmp_path, capsys):
    data_file = tmp_path / "data.json"
    data_file.write_text(json.dumps(build()), encoding="utf-8")
    cli.main(["evaluate", "--input", str(data_file), "--output", str(tmp_path / "r.json"),
              "--k", "2"])
    assert "Overall score:" in capsys.readouterr().out


def test_several_documents():
    docs = {"a.md": "Meals cost 45 euros.", "b.md": "Hotels cost 180 euros."}
    chunks = [{"id": "h", "text": "Hotels cost 180 euros."}]
    golden = [{"id": "q1", "question": "Hotel?", "expected_answer": "180 euros.",
               "expected_evidence": ["Hotels cost 180 euros."], "validated": True}]
    data = build_dataset(chunks, [{"id": "q1", "answer": "180"}], golden, docs)
    assert data["chunks"][0]["doc_id"] == "b.md"


@pytest.mark.parametrize("chunks, message", [
    ([{"id": "x", "text": "Not in the policy at all."}], "not in document"),
    ([{"id": "x", "text": "The daily meal allowance is 45 euros.", "doc_id": "nope.md"}],
     "unknown doc_id"),
    ([CHUNKS[0], dict(CHUNKS[1], id="p1")], "more than once"),
    ([{"text": "no id"}], "'id' and a 'text'"),
    ([], "non-empty"),
])
def test_bad_chunks(chunks, message):
    with pytest.raises(ValueError, match=message):
        build(chunks=chunks)


def test_text_in_two_documents_needs_a_doc_id():
    docs = {"a.md": "Same text.", "b.md": "Same text."}
    with pytest.raises(ValueError, match="give its 'doc_id'"):
        build_dataset([{"id": "x", "text": "Same text."}], RUNS, GOLDEN, docs)


def test_run_without_golden_question():
    with pytest.raises(ValueError, match="Run 1: no golden question"):
        build(runs=[{"question": "Something else?", "answer": "?"}])


def test_unknown_run_keys_name_the_right_ones():
    with pytest.raises(ValueError, match=r"unknown keys \['hits'\].*retrieved_ids"):
        build(runs=[{"question": GOLDEN[0]["question"], "hits": ["p2"]}])


def test_unreviewed_golden_is_refused():
    golden = [dict(GOLDEN[0], validated=False), GOLDEN[1]]
    with pytest.raises(ValueError, match="not reviewed yet: q1"):
        build(golden=golden)
    assert build(golden=golden, require_validated=False)["queries"][0]["id"] == "q1"


def test_empty_runs():
    with pytest.raises(ValueError, match="runs must be"):
        build(runs=[])
