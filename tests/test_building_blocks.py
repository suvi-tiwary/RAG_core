"""Tests for chunking, the vector store, retrieval and reranking."""

import numpy as np
import pytest

from rag_titan.chunking import WordChunker
from rag_titan.generation import build_context
from rag_titan.loaders import get_loader
from rag_titan.retrieval.rerankers import KeywordReranker, NoReranker
from rag_titan.vector_stores import MemoryStore


def test_word_chunker_makes_overlapping_chunks():
    chunker = WordChunker(chunk_size=4, overlap=2)
    pieces = chunker.split_text("a b c d e f")
    assert pieces == ["a b c d", "c d e f"]


def test_word_chunker_rejects_bad_overlap():
    with pytest.raises(ValueError):
        WordChunker(chunk_size=5, overlap=5)


def test_chunk_keeps_source_and_numbers_ids():
    chunker = WordChunker(chunk_size=2, overlap=0)
    chunks = chunker.chunk([{"source": "a.pdf", "text": "one two three"}])
    assert [chunk["id"] for chunk in chunks] == ["chunk_0", "chunk_1"]
    assert chunks[0]["source"] == "a.pdf"


def test_memory_store_returns_most_similar_first():
    store = MemoryStore()
    chunks = [{"id": "x", "text": "x"}, {"id": "y", "text": "y"}]
    store.add(chunks, [np.array([1.0, 0.0]), np.array([0.0, 1.0])])

    matches = store.search(np.array([0.0, 1.0]), top_k=1)

    assert matches[0]["id"] == "y"
    assert "vector" not in matches[0]


def test_retriever_finds_matching_chunk(parts):
    matches = parts["retriever"].retrieve("how many leave days", top_k=1)
    assert matches[0]["id"] == "chunk_0"


def test_keyword_reranker_rewards_shared_words():
    matches = [
        {"id": "a", "text": "nothing in common", "score": 0.5},
        {"id": "b", "text": "remote work rules", "score": 0.45},
    ]
    best = KeywordReranker().rerank("remote work?", matches, top_n=1)
    assert best[0]["id"] == "b"


def test_no_reranker_keeps_order():
    matches = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.1}]
    assert [m["id"] for m in NoReranker().rerank("q", matches)] == ["a", "b"]


def test_build_context_shows_id_and_source():
    context = build_context([{"id": "chunk_0", "source": "a.pdf", "text": "hello"}])
    assert "[chunk_0] Source: a.pdf" in context


def test_unknown_file_type_is_rejected():
    with pytest.raises(ValueError):
        get_loader("notes.xyz")
