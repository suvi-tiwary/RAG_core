"""Tests for the retriever interface and the shared evidence format (RAG handbook step 0)."""

import pytest

from rag_titan.rag_types import SimpleRAG
from rag_titan.retrieval import BaseRetriever, DenseRetriever, Retriever
from rag_titan.retrieval import retriever as old_module
from rag_titan.vector_stores import MemoryStore


def test_old_import_paths_still_work():
    assert Retriever is DenseRetriever
    assert old_module.Retriever is DenseRetriever
    assert issubclass(DenseRetriever, BaseRetriever)


def test_results_say_which_retriever_found_them(parts):
    results = parts["retriever"].retrieve("leave days", top_k=2)

    assert results[0]["id"] == "chunk_0"
    for result in results:
        assert result["retriever"] == "dense"
        assert result["score_trace"] == {"dense": result["score"]}


def test_results_are_copies_and_the_store_is_not_changed(parts):
    parts["retriever"].retrieve("leave days", top_k=2)
    for record in parts["store"].records:
        assert "score_trace" not in record and "retriever" not in record


@pytest.mark.parametrize("bad", [0, -1, 2.5, "3", None, True])
def test_bad_top_k_raises_a_clear_error(parts, bad):
    with pytest.raises(ValueError, match="top_k"):
        parts["retriever"].retrieve("leave days", top_k=bad)


def test_empty_store_returns_nothing(parts):
    assert DenseRetriever(parts["embedder"], MemoryStore()).retrieve("leave") == []


def test_base_retriever_methods_must_be_written():
    with pytest.raises(NotImplementedError):
        BaseRetriever().retrieve("question")
    with pytest.raises(NotImplementedError):
        BaseRetriever().has_documents()


class FixedRetriever(BaseRetriever):
    """Always returns the same chunk, to show any retriever can be plugged in."""

    name = "fixed"

    def has_documents(self):
        return True

    def retrieve(self, query, top_k=5):
        self._check_top_k(top_k)
        chunk = {"id": "c1", "source": "faq.txt", "text": "leave days are 20", "score": 1.0}
        return self._with_trace([chunk])[:top_k]


def test_any_base_retriever_works_in_simple_rag(parts):
    rag = SimpleRAG(FixedRetriever(), parts["reranker"], parts["generator"])

    assert rag.answer("how many leave days?") == "fake answer"
    prompt = parts["llm"].calls[-1][-1]["content"]
    assert "leave days are 20" in prompt
