"""
Shared test helpers.

FakeLLM and FakeEmbedder replace Ollama, so the tests run anywhere,
fast, and without downloading any model.
"""

import numpy as np
import pytest
from fake_ollama import fake_ollama  # noqa: F401  (shares the fixture with every test)

from rag_titan.chunking import WordChunker
from rag_titan.embeddings import BaseEmbedder
from rag_titan.generation import GroundedGenerator
from rag_titan.llm import BaseLLM
from rag_titan.prompts import EVIDENCE_CHECK_PROMPT, ROUTER_PROMPT
from rag_titan.retrieval import Retriever
from rag_titan.retrieval.rerankers import KeywordReranker
from rag_titan.vector_stores import MemoryStore

# a few words used to build small, easy to understand vectors
VOCABULARY = ["leave", "days", "salary", "remote", "work", "office", "holiday"]


class FakeLLM(BaseLLM):
    """
    LLM that replies from a list of prepared answers, in order.
    Every message it receives is saved in `calls` so tests can check it.
    """

    def __init__(self, replies=None):
        super().__init__("fake-model")
        self.replies = list(replies or ["fake answer"])
        self.calls = []

    def chat(self, messages, tools=None):
        """Return the next prepared reply (the last one repeats)."""
        self.calls.append(messages)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return {"content": reply, "tool_calls": [], "message": {"role": "assistant"}}

    def tool_result_message(self, tool_call, result):
        """Build a simple tool message."""
        return {"role": "tool", "content": str(result)}


class PromptLLM(BaseLLM):
    """Replies by prompt: route "simple", evidence check `enough`, else an answer."""

    def __init__(self, enough="yes"):
        super().__init__("prompt-fake")
        self.enough = enough

    def chat(self, messages, tools=None):
        system = messages[0]["content"]
        if system == ROUTER_PROMPT:
            reply = "simple"
        elif system == EVIDENCE_CHECK_PROMPT:
            reply = self.enough
        else:
            reply = "the answer"
        return {"content": reply, "tool_calls": [], "message": {"role": "assistant"}}


class FakeEmbedder(BaseEmbedder):
    """Counts how often each VOCABULARY word appears in the text."""

    def __init__(self):
        super().__init__("fake-embedder")

    def embed_many(self, texts):
        """Return one word-count vector per text."""
        vectors = []
        for text in texts:
            words = text.lower().split()
            vectors.append(np.array([words.count(word) for word in VOCABULARY], dtype=float))
        return vectors


def make_chunks():
    """Return three small chunks about different topics."""
    return [
        {"id": "chunk_0", "source": "hr.pdf", "text": "you get 20 leave days every year"},
        {"id": "chunk_1", "source": "hr.pdf", "text": "salary is paid every month"},
        {"id": "chunk_2", "source": "it.pdf", "text": "remote work needs a laptop"},
    ]


@pytest.fixture
def parts():
    """Build RAG parts with fakes and three indexed chunks."""
    llm = FakeLLM()
    embedder = FakeEmbedder()
    store = MemoryStore()

    chunks = make_chunks()
    store.add(chunks, embedder.embed_many([chunk["text"] for chunk in chunks]))

    return {
        "llm": llm,
        "embedder": embedder,
        "store": store,
        "chunker": WordChunker(chunk_size=5, overlap=1),
        "retriever": Retriever(embedder, store),
        "reranker": KeywordReranker(),
        "generator": GroundedGenerator(llm),
    }


# four sentences of a small handbook: each one becomes a chunk (c1 .. c4)
HANDBOOK = ["Employees get 25 days of annual leave. ", "Sick pay is full salary for 10 days. ",
            "Travel needs manager approval. ", "Remote work is allowed 3 days a week."]


def make_query(number, gold, retrieved, reranked, answer):
    """Return one fully logged query: gold labels plus what every stage produced."""
    return {
        "id": f"q{number}", "expected_chunk_ids": [gold], "expected_evidence": [answer],
        "retrieved_ids": retrieved, "reranked_ids": reranked, "context_ids": reranked[:2],
        "answer": answer, "expected_answer": answer, "required_facts": ["days"],
        "citations": [gold], "expected_citations": [gold],
        "verified_claims": [{"text": answer, "supported": True}],
        "answerable": True, "abstained": False, "task_success": True,
        "retrieval_latency_ms": 20 * number, "generation_latency_ms": 400 * number,
        "latency_ms": 450 * number, "cost_usd": 0.002, "total_tokens": 800,
    }


@pytest.fixture
def rag_dataset():
    """A small, healthy dataset for RAGEvaluator: every stage has data and scores well."""
    chunks, start = [], 0
    for number, text in enumerate(HANDBOOK, start=1):
        chunks.append({"id": f"c{number}", "text": text, "start": start,
                       "end": start + len(text), "metadata": {"source": "handbook.md", "page": 1}})
        start += len(text)

    return {
        "document_text": "".join(HANDBOOK),
        "chunks": chunks,
        "queries": [
            make_query(1, "c1", ["c2", "c1", "c3"], ["c1", "c2", "c3"],
                       "Employees get 25 days of annual leave."),
            make_query(2, "c4", ["c4", "c3", "c1"], ["c4", "c1", "c3"],
                       "Remote work is allowed 3 days a week."),
        ],
        "expected_chunks": [{"start": chunk["start"], "end": chunk["end"]} for chunk in chunks],
        "expected_evidence": ["25 days of annual leave", "allowed 3 days a week"],
        "required_metadata": ["source", "page"],
        "size_limits": [3, 20],
        "max_context_tokens": 30,
        "embeddings": {
            "vectors": {"c1": [0.95, 0.05], "c2": [0.7, 0.3], "c3": [0.5, 0.5], "c4": [0.1, 0.9]},
            "query_vectors": {"q1": [1.0, 0.0], "q2": [0.0, 1.0]},
            "relevant_ids": {"q1": ["c1"], "q2": ["c4"]},
            "indexed_ids": ["c1", "c2", "c3", "c4"],
            "index_results": {"q1": ["c1", "c2", "c3"], "q2": ["c4", "c3", "c2"]},
        },
    }
