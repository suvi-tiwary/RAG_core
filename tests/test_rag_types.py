"""Tests for SimpleRAG, AdaptiveRAG and the indexing pipeline."""

from conftest import FakeLLM

from rag_titan import AdaptiveRAG, SimpleRAG
from rag_titan.generation import GroundedGenerator
from rag_titan.pipelines import IndexingPipeline
from rag_titan.prompts import NO_ANSWER


def test_simple_rag_answers_with_llm(parts):
    rag = SimpleRAG(parts["retriever"], parts["reranker"], parts["generator"])
    assert rag.answer("how many leave days?") == "fake answer"


def test_simple_rag_sends_best_chunk_to_llm(parts):
    rag = SimpleRAG(parts["retriever"], parts["reranker"], parts["generator"], top_n=1)
    rag.answer("how many leave days?")

    user_message = parts["llm"].calls[-1][1]["content"]
    assert "20 leave days" in user_message


def test_simple_rag_rejects_empty_question(parts):
    rag = SimpleRAG(parts["retriever"], parts["reranker"], parts["generator"])
    assert rag.answer("   ") == "Please enter a question."


def test_adaptive_rag_direct_route_skips_search(parts):
    llm = FakeLLM(["direct", "Hello there!"])
    rag = AdaptiveRAG(llm, parts["retriever"], parts["reranker"], GroundedGenerator(llm))
    assert rag.answer("hi") == "Hello there!"


def test_adaptive_rag_answers_when_evidence_is_enough(parts):
    llm = FakeLLM(["simple", "yes", "20 days [chunk_0]"])
    rag = AdaptiveRAG(llm, parts["retriever"], parts["reranker"], GroundedGenerator(llm))
    assert rag.answer("how many leave days?") == "20 days [chunk_0]"


def test_adaptive_rag_gives_up_without_evidence(parts):
    llm = FakeLLM(["simple", "no", "leave days", "no"])
    rag = AdaptiveRAG(llm, parts["retriever"], parts["reranker"], GroundedGenerator(llm))
    assert rag.answer("how many leave days?") == NO_ANSWER


def test_indexing_pipeline_stores_chunks(parts, monkeypatch):
    # pretend to load a file so the test does not need a real PDF
    fake_document = {"source": "a.pdf", "file_type": ".pdf", "text": "one two three four five six"}
    monkeypatch.setattr(
        "rag_titan.pipelines.indexing_pipeline.load_document", lambda path: fake_document
    )

    pipeline = IndexingPipeline(parts["chunker"], parts["embedder"], parts["store"])
    count = pipeline.run(["a.pdf"])

    assert count == 2
    assert parts["store"].count() == 2
