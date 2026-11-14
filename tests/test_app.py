"""Tests for the RAGTitan shortcuts: run() and evaluate_retrieval()."""

import json

import pytest
from conftest import FakeLLM

import rag_titan.app as app
from rag_titan import RAGTitan
from rag_titan.generation import GroundedGenerator
from rag_titan.pipelines import IndexingPipeline

QUESTIONS = [
    {"id": "q1", "question": "leave days", "required_evidence_ids": ["hr.pdf"]},
    {"id": "q2", "question": "remote work", "required_evidence_ids": ["it.pdf"]},
]


@pytest.fixture
def titan(parts, monkeypatch):
    """A RAGTitan built from the fake parts (already indexed), no Ollama needed."""
    llm = FakeLLM(["20 leave days [chunk_0]"])
    fake = {**parts, "llm": llm, "generator": GroundedGenerator(llm),
            "pipeline": IndexingPipeline(parts["chunker"], parts["embedder"], parts["store"])}
    monkeypatch.setattr(app, "build_rag_parts", lambda settings: fake)
    return RAGTitan(rag_type="simple")


def test_run_returns_the_steps_and_a_readable_log(titan):
    result = titan.run("how many leave days?")

    assert result["answer"] == "20 leave days [chunk_0]" == titan.ask("how many leave days?")
    assert [step["step"] for step in result["steps"]] == ["retrieve", "rerank", "generate"]
    assert result["log"].startswith("1. retrieve")
    assert "chunk_0 hr.pdf" in result["log"]          # the log names the file


def test_evaluate_runs_the_rag_end_to_end(titan, capsys):
    result = titan.evaluate(QUESTIONS)

    assert result["name"] == "simple" and result["answered"] == 1.0
    assert [row["id"] for row in result["rows"]] == ["q1", "q2"]
    assert "End-to-end report: 2 questions" in capsys.readouterr().out


def test_evaluate_retrieval_from_a_file(titan, tmp_path, capsys):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(QUESTIONS), encoding="utf-8")

    result = titan.evaluate_retrieval(str(path), top_k=1)

    assert result["recall"] == 1.0 and result["top_k"] == 1
    assert "Recall@1" in capsys.readouterr().out      # the report was printed


def test_evaluate_retrieval_from_a_list_without_printing(titan, capsys):
    result = titan.evaluate_retrieval(QUESTIONS, top_k=2, show=False)

    assert [row["id"] for row in result["rows"]] == ["q1", "q2"]
    assert capsys.readouterr().out == ""


def test_evaluate_retrieval_checks_the_questions(titan):
    with pytest.raises(ValueError, match="non-empty list"):
        titan.evaluate_retrieval([])
    with pytest.raises(FileNotFoundError):
        titan.evaluate_retrieval("missing.json")
