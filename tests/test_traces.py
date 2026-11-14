"""Tests for run(), the step traces and FLOW (they feed the docs playground)."""

import re

import pytest
from conftest import FakeLLM

from rag_titan import RAG_TYPES, AdaptiveRAG, SimpleRAG, build_rag
from rag_titan.generation import GroundedGenerator
from rag_titan.prompts import NO_ANSWER
from rag_titan.rag_types import BaseRAG, Trace, describe_steps
from rag_titan.rag_types.base_rag import NODE_KINDS


def adaptive(parts, replies):
    llm = FakeLLM(replies)
    return AdaptiveRAG(llm, parts["retriever"], parts["reranker"], GroundedGenerator(llm))


def names(result):
    return [step["step"] for step in result["steps"]]


def assert_follows_flow(rag, result):
    """Every step is a FLOW node and two steps in a row are joined by an edge."""
    node_ids = {node["id"] for node in rag.FLOW["nodes"]}
    edges = {tuple(edge) for edge in rag.FLOW["edges"]}
    steps = names(result)

    assert set(steps) <= node_ids
    for pair in zip(steps, steps[1:]):
        assert pair in edges, f"{type(rag).__name__}: no FLOW edge {pair}"


@pytest.mark.parametrize("rag_type", RAG_TYPES)
def test_every_rag_type_has_a_valid_flow(parts, rag_type):
    flow = build_rag(rag_type, parts).FLOW
    ids = [node["id"] for node in flow["nodes"]]

    assert ids, f"{rag_type} has no FLOW: the playground cannot draw it"
    assert len(ids) == len(set(ids))
    for node in flow["nodes"]:
        assert node["label"] and node["kind"] in NODE_KINDS
    for start, end in flow["edges"]:
        assert start in ids and end in ids


def test_simple_rag_trace(parts):
    rag = SimpleRAG(parts["retriever"], parts["reranker"], parts["generator"])
    result = rag.run("how many leave days?")

    assert names(result) == ["retrieve", "rerank", "generate"]
    assert result["answer"] == rag.answer("how many leave days?") == "fake answer"
    assert result["sources"][0]["id"] == "chunk_0"
    assert result["steps"][0]["input"] == {"query": "how many leave days?", "top_k": 5}
    assert all(step["ms"] >= 0 for step in result["steps"])
    assert_follows_flow(rag, result)


def test_adaptive_direct_route_skips_search(parts):
    rag = adaptive(parts, ["direct", "Hello there!"])
    result = rag.run("hi")

    assert names(result) == ["route", "direct_reply"]
    assert result["answer"] == "Hello there!" and result["sources"] == []
    assert_follows_flow(rag, result)


def test_adaptive_enough_evidence_first_time(parts):
    rag = adaptive(parts, ["simple", "yes", "20 days [chunk_0]"])
    result = rag.run("how many leave days?")

    assert names(result) == ["route", "retrieve", "rerank", "check", "generate"]
    assert result["answer"] == "20 days [chunk_0]"
    assert_follows_flow(rag, result)


def test_adaptive_rewrite_and_retry_loop(parts):
    rag = adaptive(parts, ["simple", "no", "annual leave days", "yes", "20 days"])
    result = rag.run("how many leave days?")

    assert names(result) == ["route", "retrieve", "rerank", "check", "rewrite",
                             "retrieve", "rerank", "check", "generate"]
    second_search = result["steps"][5]["input"]
    assert second_search == {"query": "annual leave days", "top_k": 8}   # wider search
    assert_follows_flow(rag, result)


def test_adaptive_gives_up(parts):
    rag = adaptive(parts, ["simple", "no", "leave days", "no"])
    result = rag.run("how many leave days?")

    assert names(result)[-1] == "give_up"
    assert result["answer"] == NO_ANSWER and result["sources"] == []
    assert_follows_flow(rag, result)


def test_empty_question_has_no_steps(parts):
    result = SimpleRAG(parts["retriever"], parts["reranker"], parts["generator"]).run("  ")
    assert result == {"answer": "Please enter a question.", "sources": [], "steps": []}
    assert describe_steps(result["steps"]) == "(no steps: the question was not searched)"


def test_trace_record_returns_the_output_and_saves_the_step():
    trace = Trace()
    assert trace.record("add", lambda a, b: a + b, a=2, b=3) == 5
    assert trace.steps[0]["step"] == "add" and trace.steps[0]["output"] == 5


def test_one_chunk_is_singular():
    steps = [{"step": "retrieve", "input": {"top_k": 1}, "ms": 0.0,
              "output": [{"id": "chunk_2", "score": 1.0}]}]
    assert describe_steps(steps).splitlines()[1].strip() == "-> 1 chunk: chunk_2 (1.000)"


def test_base_rag_run_must_be_written():
    with pytest.raises(NotImplementedError):
        BaseRAG().answer("question")


# what a user sees with print(describe_steps(result["steps"])), timings hidden
EXPECTED_LOG = """\
1. route        0.00 ms  query='how many leave days?'
                         -> 'simple'
2. retrieve     0.00 ms  query='how many leave days?', top_k=4
                         -> 3 chunks: chunk_0 hr.pdf (0.707), chunk_1 hr.pdf (0.000), chunk_2 it.pdf (0.000)
3. rerank       0.00 ms  query='how many leave days?', matches=3 chunks, top_n=2
                         -> 2 chunks: chunk_0 hr.pdf (rerank 0.807), chunk_1 hr.pdf (rerank 0.000)
4. check        0.00 ms  query='how many leave days?', chunks=2 chunks
                         -> False
5. rewrite      0.00 ms  query='how many leave days?'
                         -> 'annual leave days per year'
6. retrieve     0.00 ms  query='annual leave days per year', top_k=8
                         -> 3 chunks: chunk_0 hr.pdf (1.000), chunk_1 hr.pdf (0.000), chunk_2 it.pdf (0.000)
7. rerank       0.00 ms  query='how many leave days?', matches=3 chunks, top_n=4
                         -> 3 chunks: chunk_0 hr.pdf (rerank 1.100), chunk_1 hr.pdf (rerank 0.000), chunk_2 it.pdf (rerank 0.000)
8. check        0.00 ms  query='how many leave days?', chunks=3 chunks
                         -> True
9. generate     0.00 ms  query='how many leave days?', chunks=3 chunks
                         -> 'You get 20 leave days [chunk_0]'"""  # noqa: E501 (the log is wide on purpose)


def test_describe_steps_log(parts):
    rag = adaptive(parts, ["simple", "no", "annual leave days per year", "yes",
                           "You get 20 leave days [chunk_0]"])
    log = describe_steps(rag.run("how many leave days?")["steps"])
    print("\n" + log)   # see it with: pytest tests/test_traces.py -s -k log

    # timings change on every run, so show them all as 0.00 (same width)
    hidden = re.sub(r"\d+\.\d\d(?= ms)", lambda m: "0.00".rjust(len(m.group())), log)
    assert hidden == EXPECTED_LOG
