"""Tests for end-to-end evaluation: evaluate_rag, compare_setups, format_rag_report."""

import pytest
from conftest import FakeEmbedder, PromptLLM

import rag_titan.evaluation.end_to_end as end_to_end
from rag_titan import AdaptiveRAG, SimpleRAG
from rag_titan.chunking import WordChunker
from rag_titan.evaluation import compare_setups, evaluate_rag, format_rag_report
from rag_titan.generation import GroundedGenerator
from rag_titan.pipelines import IndexingPipeline
from rag_titan.prompts import NO_ANSWER
from rag_titan.retrieval import Retriever
from rag_titan.retrieval.rerankers import KeywordReranker
from rag_titan.vector_stores import MemoryStore

QUESTIONS = [
    {"id": "q1", "question": "leave days", "required_evidence_ids": ["hr.pdf"]},
    {"id": "q2", "question": "remote work", "required_evidence_ids": ["it.pdf"]},
]


def test_evaluate_rag_scores_the_sources(parts):
    rag = SimpleRAG(parts["retriever"], parts["reranker"], GroundedGenerator(PromptLLM()),
                    top_n=1)
    result = evaluate_rag(rag, QUESTIONS, name="simple")

    assert result["name"] == "simple"
    assert result["recall"] == result["mrr"] == result["ndcg"] == 1.0
    assert result["answered"] == 1.0 and result["steps"] == 3.0
    assert 0 <= result["p50_s"] <= result["p95_s"]
    row = result["rows"][0]
    assert row["answer"] == "the answer" and row["sources"] == ["hr.pdf"]
    assert [step["step"] for step in row["steps"]] == ["retrieve", "rerank", "generate"]


def test_giving_up_counts_as_not_answered(parts):
    llm = PromptLLM(enough="no")
    rag = AdaptiveRAG(llm, parts["retriever"], parts["reranker"], GroundedGenerator(llm))
    result = evaluate_rag(rag, QUESTIONS)

    assert result["name"] == "AdaptiveRAG"
    assert result["answered"] == 0.0
    assert all(row["answer"] == NO_ANSWER for row in result["rows"])
    assert result["recall"] == 0.0          # no chunks were given to the LLM
    # route + 2 x (retrieve, rerank, check) + rewrite + give_up = 1 + 6 + 1 + 1
    assert result["steps"] == 9.0


@pytest.fixture
def builds(monkeypatch):
    """Replace build_rag_parts with fakes and count how often an index is built."""
    calls = []

    def fake_build(settings):
        calls.append(settings["chunk_size"])
        llm, embedder, store = PromptLLM(), FakeEmbedder(), MemoryStore()
        chunker = WordChunker(settings["chunk_size"], 0)
        return {"llm": llm, "retriever": Retriever(embedder, store),
                "pipeline": IndexingPipeline(chunker, embedder, store),
                "reranker": KeywordReranker(), "generator": GroundedGenerator(llm)}

    monkeypatch.setattr(end_to_end, "build_rag_parts", fake_build)
    return calls


@pytest.fixture
def files(tmp_path):
    (tmp_path / "hr.pdf.txt").write_text("you get 20 leave days every year", encoding="utf-8")
    (tmp_path / "it.txt").write_text("remote work needs a laptop", encoding="utf-8")
    return [str(tmp_path / "hr.pdf.txt"), str(tmp_path / "it.txt")]


def test_compare_setups_shares_an_index_per_ingestion(builds, files):
    questions = [{"id": "q1", "question": "leave days", "required_evidence_ids": ["hr.pdf.txt"]}]
    setups = [
        {"name": "simple + 5", "rag_type": "simple", "settings": {"chunk_size": 5}},
        {"name": "adaptive + 5", "rag_type": "adaptive", "settings": {"chunk_size": 5}},
        {"name": "simple + 50", "rag_type": "simple", "settings": {"chunk_size": 50}},
    ]
    results = compare_setups(files, questions, setups)

    assert builds == [5, 50]                       # the two chunk_size 5 setups share one
    assert [result["name"] for result in results] == ["simple + 5", "adaptive + 5", "simple + 50"]
    assert results[1]["rag_type"] == "adaptive" and results[1]["settings"] == {"chunk_size": 5}
    assert all(result["answered"] == 1.0 for result in results)
    assert results[0]["steps"] == 3.0 and results[1]["steps"] == 5.0


def test_format_rag_report(parts):
    rag = SimpleRAG(parts["retriever"], parts["reranker"], GroundedGenerator(PromptLLM()))
    lines = format_rag_report([evaluate_rag(rag, QUESTIONS, name="simple")]).splitlines()

    assert lines[0].startswith("End-to-end report: 2 questions")
    assert lines[2].split() == ["setup", "Recall", "MRR", "nDCG", "Answered", "Steps",
                                "p50", "s", "p95", "s"]
    assert lines[4].split()[4:6] == ["100%", "3.0"]


@pytest.mark.parametrize("setups, message", [
    ([], "non-empty list"),
    ([{"rag_type": "simple"}], "'name'"),
    ([{"name": "a"}], "'rag_type'"),
    ([{"name": "a", "rag_type": "simple"}, {"name": "a", "rag_type": "simple"}], "more than once"),
    ([{"name": "a", "rag_type": "simple", "settings": "word"}], "must be a dict"),
])
def test_bad_setups_raise_a_clear_error(builds, files, setups, message):
    with pytest.raises(ValueError, match=message):
        compare_setups(files, QUESTIONS, setups)


def test_bad_inputs(builds, files):
    with pytest.raises(ValueError, match="files"):
        compare_setups([], QUESTIONS, [{"name": "a", "rag_type": "simple"}])
    with pytest.raises(ValueError, match="Unknown RAG type"):
        compare_setups(files, QUESTIONS, [{"name": "a", "rag_type": "graph"}])
    with pytest.raises(ValueError, match="no results"):
        format_rag_report([])
