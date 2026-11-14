"""Tests for collect_dataset and RAGTitan.evaluate_stages (fake models, no Ollama)."""

import json
from pathlib import Path

import pytest
from test_cli_evaluate import fake_parts

import rag_titan.app as app
from rag_titan import RAGTitan
from rag_titan.evaluation import RAGEvaluator, collect_dataset
from rag_titan.evaluation.collect import load_golden, split_citations

EXAMPLES = Path(__file__).parent.parent / "examples"

GOLDEN = [
    {"id": "q1", "question": "leave days", "expected_evidence": ["20 leave days"],
     "expected_answer": "the answer", "required_facts": ["answer"], "answerable": True},
    {"id": "q2", "question": "remote work", "expected_evidence": ["needs a laptop"]},
]


@pytest.fixture
def rag_and_files(tmp_path, monkeypatch):
    """A RAGTitan built from fakes, and two small files."""
    monkeypatch.setattr(app, "build_rag_parts", fake_parts)
    # the line break checks that chunks are placed by offsets: WordChunker turns it into a space
    (tmp_path / "hr.md").write_text("You get 20 leave days\nevery year.", encoding="utf-8")
    (tmp_path / "it.md").write_text("Remote work needs a laptop.", encoding="utf-8")
    return RAGTitan(), [str(tmp_path / "hr.md"), str(tmp_path / "it.md")]


def test_collect_dataset_records_every_stage(rag_and_files):
    rag, files = rag_and_files
    dataset = collect_dataset(rag, files, GOLDEN)

    assert dataset["documents"] == {"hr.md": "You get 20 leave days\nevery year.",
                                    "it.md": "Remote work needs a laptop."}
    first = dataset["chunks"][0]
    assert (first["id"], first["doc_id"], first["start"]) == ("chunk_0", "hr.md", 0)
    assert first["text"] == "You get 20 leave days\nevery year."
    assert first["metadata"]["source"] == "hr.md"

    query = dataset["queries"][0]
    assert query["expected_chunk_ids"] == ["chunk_0"]         # the chunk holding the quote
    assert query["retrieved_ids"][0] == "chunk_0"
    assert query["reranked_ids"][0] == "chunk_0"
    assert query["context_ids"] == query["reranked_ids"]
    assert query["answer"] == "the answer"
    assert query["abstained"] is False
    assert query["latency_ms"] >= query["generation_latency_ms"] >= 0
    assert "embeddings" not in dataset


def test_collected_dataset_evaluates_every_stage(rag_and_files):
    rag, files = rag_and_files
    dataset = collect_dataset(rag, files, GOLDEN, include_embeddings=True)
    report = RAGEvaluator(k=2).evaluate(**dataset)

    assert json.loads(json.dumps(dataset)) == dataset          # can be saved as a data file
    assert report["chunking"]["coverage"] == 1
    assert report["chunking"]["warnings"] == []
    assert report["embedding"]["index_coverage"] == 1
    assert report["retrieval"]["summary"]["recall_at_k"] == 1
    assert report["augmentation"]["summary"]["context_sufficient"] == 1
    assert report["generation"]["summary"]["exact_match"] == 1          # q1 has a reference
    assert report["end_to_end"]["p95_latency_ms"] is not None
    assert report["scorecard"]["overall"]["evaluated_stages"] == 7


def test_file_and_page_labels_are_a_fallback(rag_and_files):
    rag, files = rag_and_files
    golden = [
        {"id": "q1", "question": "leave days", "required_evidence_ids": ["hr.md"]},
        {"id": "q2", "question": "remote work", "required_evidence_ids": ["it.md#p9"]},
        {"id": "q3", "question": "leave days", "required_evidence_ids": ["it.md"],
         "expected_evidence": ["20 leave days"]},                    # quotes win over labels
        {"id": "q4", "question": "leave days", "required_evidence_ids": ["hr.md"],
         "expected_chunk_ids": ["chunk_1"]},                         # own chunk labels win
        {"id": "q5", "question": "leave days"},                      # no evidence labels
    ]
    queries = collect_dataset(rag, files, golden)["queries"]

    assert queries[0]["expected_chunk_ids"] == ["chunk_0"]           # every chunk of hr.md
    assert queries[1]["expected_chunk_ids"] == []                    # it.md has no page 9
    assert queries[2]["expected_chunk_ids"] == ["chunk_0"]
    assert queries[3]["expected_chunk_ids"] == ["chunk_1"]
    assert "expected_chunk_ids" not in queries[4]

    report = RAGEvaluator(k=2).evaluate(**collect_dataset(rag, files, golden[:1]))
    assert report["retrieval"]["summary"]["hit_at_k"] == 1


def test_existing_questions_file_works_with_the_bridge(rag_and_files):
    rag, _ = rag_and_files
    files = sorted(str(path) for path in (EXAMPLES / "sample_docs").glob("*.md"))
    dataset = collect_dataset(rag, files, EXAMPLES / "questions.json")

    assert all(query["expected_chunk_ids"] for query in dataset["queries"])


def test_split_citations():
    answer = "You get 20 days [chunk_1, page 1]. Sick pay is full [chunk_12] [chunk_1]."
    text, cited = split_citations(answer, ["chunk_1", "chunk_12", "chunk_2"])

    assert text == "You get 20 days. Sick pay is full."
    assert cited == ["chunk_1", "chunk_12"]              # chunk_1 is not found inside chunk_12
    assert split_citations("See [1] and [note].", ["chunk_1"]) == ("See [1] and [note].", [])


def test_citations_are_collected_from_the_answer(rag_and_files):
    rag, files = rag_and_files
    rag.parts["llm"].chat = lambda messages, tools=None: {
        "content": "20 days [chunk_0].", "tool_calls": [], "message": {"role": "assistant"}}
    query = collect_dataset(rag, files, GOLDEN)["queries"][0]

    assert query["answer"] == "20 days."
    assert query["raw_answer"] == "20 days [chunk_0]."
    assert query["citations"] == ["chunk_0"]
    assert query["expected_citations"] == ["chunk_0"]


def test_evaluate_stages_shortcut(rag_and_files, capsys):
    rag, files = rag_and_files
    report = rag.evaluate_stages(files, GOLDEN, k=2)
    out = capsys.readouterr().out

    assert out.startswith("Overall score: ")
    assert "First failing stage: " in out
    assert "embedding" not in report
    assert report["retrieval"]["summary"]["hit_at_k"] == 1


def test_golden_questions_from_a_file(rag_and_files, tmp_path):
    rag, files = rag_and_files
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(GOLDEN), encoding="utf-8")

    assert len(collect_dataset(rag, files, path)["queries"]) == 2


def test_errors(rag_and_files, tmp_path):
    rag, files = rag_and_files
    with pytest.raises(FileNotFoundError, match="Golden questions file not found"):
        load_golden(tmp_path / "missing.json")
    with pytest.raises(ValueError, match="non-empty list"):
        load_golden([])
    with pytest.raises(ValueError, match="'question' must be"):
        load_golden([{"id": "q1"}])
    with pytest.raises(ValueError, match="files must be"):
        collect_dataset(rag, [], GOLDEN)
    with pytest.raises(ValueError, match="exactly once"):
        collect_dataset(rag, files, [{"id": "q1", "question": "x",
                                      "expected_evidence": ["not in the files"]}])


def test_example_golden_quotes_are_in_the_sample_docs():
    texts = [path.read_text(encoding="utf-8") for path in (EXAMPLES / "sample_docs").glob("*.md")]
    for record in load_golden(EXAMPLES / "golden_questions.json"):
        for quote in record["expected_evidence"]:
            assert sum(text.count(quote) for text in texts) == 1, f"{record['id']}: {quote}"
