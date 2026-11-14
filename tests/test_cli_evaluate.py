"""Tests for `rag-titan evaluate` and `rag-titan ask --trace` (fake models, no Ollama)."""

import json

import pytest
from conftest import FakeEmbedder, PromptLLM

import rag_titan.app as app
import rag_titan.evaluation.end_to_end as end_to_end
from rag_titan import cli, get_settings
from rag_titan.factory import CHUNKERS, build_chunker
from rag_titan.generation import GroundedGenerator
from rag_titan.pipelines import IndexingPipeline
from rag_titan.retrieval import Retriever
from rag_titan.retrieval.rerankers import KeywordReranker
from rag_titan.vector_stores import MemoryStore


def fake_parts(settings):
    """Real chunker from the settings, fake LLM and embedder."""
    llm, embedder, store = PromptLLM(), FakeEmbedder(), MemoryStore()
    return {"llm": llm, "retriever": Retriever(embedder, store),
            "pipeline": IndexingPipeline(build_chunker(settings), embedder, store),
            "reranker": KeywordReranker(), "generator": GroundedGenerator(llm)}


@pytest.fixture
def setup_files(tmp_path, monkeypatch):
    """Two small files, a questions file, and fakes instead of Ollama."""
    monkeypatch.setattr(end_to_end, "build_rag_parts", fake_parts)
    monkeypatch.setattr(app, "build_rag_parts", fake_parts)

    (tmp_path / "hr.md").write_text("You get 20 leave days every year.", encoding="utf-8")
    (tmp_path / "it.md").write_text("Remote work needs a laptop.", encoding="utf-8")
    questions = [
        {"id": "q1", "question": "leave days", "required_evidence_ids": ["hr.md"]},
        {"id": "q2", "question": "remote work", "required_evidence_ids": ["it.md"]},
        {"id": "q3", "question": "salary", "required_evidence_ids": ["pay.md"]},   # no such file
    ]
    (tmp_path / "q.json").write_text(json.dumps(questions), encoding="utf-8")
    return ["--files", str(tmp_path / "hr.md"), str(tmp_path / "it.md"),
            "--questions", str(tmp_path / "q.json")]


def test_every_chunker_name_builds():
    for name in CHUNKERS:
        if name != "token":   # needs the optional tiktoken extra
            build_chunker(get_settings({"chunker": name}))


def test_build_setups_makes_every_combination():
    args = cli.build_parser().parse_args(
        ["evaluate", "--files", "a.md", "--questions", "q.json", "--rag-type", "simple",
         "adaptive", "--chunker", "word", "paragraph", "--chunk-size", "50"])
    setups = cli.build_setups(args)

    assert [setup["name"] for setup in setups] == [
        "simple + word", "adaptive + word", "simple + paragraph", "adaptive + paragraph"]
    assert setups[2]["settings"] == {"chunker": "paragraph", "chunk_size": 50}


def test_evaluate_prints_the_report_and_weak_questions(setup_files, capsys):
    cli.main(["evaluate", *setup_files, "--rag-type", "simple", "adaptive",
              "--chunker", "word", "sentence"])
    out = capsys.readouterr().out

    assert out.startswith("End-to-end report: 3 questions")
    for name in ["simple + word", "adaptive + word", "simple + sentence", "adaptive + sentence"]:
        assert name in out
    assert "Questions to look at:" in out and "q3  recall 0.00" in out


def test_evaluate_retrieval_only(setup_files, capsys):
    cli.main(["evaluate", *setup_files, "--retrieval-only", "--top-k", "1",
              "--chunker", "word", "paragraph"])
    out = capsys.readouterr().out

    assert out.startswith("Retrieval report: 3 questions, top_k=1")
    assert "\nword " in out and "\nparagraph " in out


def test_ask_with_trace_prints_the_steps(setup_files, capsys):
    files = setup_files[:3]
    cli.main(["ask", *files, "--question", "leave days", "--trace"])
    out = capsys.readouterr().out

    assert "1. retrieve" in out and "3. generate" in out and "hr.md" in out
    assert out.rstrip().endswith("the answer")


def test_unknown_chunker_is_refused(capsys):
    with pytest.raises(SystemExit):
        cli.main(["evaluate", "--files", "a.md", "--questions", "q.json", "--chunker", "lines"])
    assert "invalid choice" in capsys.readouterr().err
