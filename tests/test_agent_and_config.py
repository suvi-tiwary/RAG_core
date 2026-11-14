"""Tests for the agent loop, the math tools, settings and the factory."""

import sys

import pytest

from rag_titan import get_settings
from rag_titan.agents import BaseAgent
from rag_titan.agents.tools.math_tools import add, divide
from rag_titan.chunking import (
    CharacterChunker,
    HeadingChunker,
    PageChunker,
    ParagraphChunker,
    RecursiveChunker,
    SentenceChunker,
    WordChunker,
)
from rag_titan.factory import build_chunker, build_reranker
from rag_titan.llm import BaseLLM


class ToolCallingLLM(BaseLLM):
    """Asks for the add tool once, then returns the final answer."""

    def __init__(self):
        super().__init__("fake")
        self.step = 0

    def chat(self, messages, tools=None):
        self.step += 1
        if self.step == 1:
            calls = [{"name": "add", "arguments": {"a": "2", "b": "3"}}]
            return {"content": "", "tool_calls": calls, "message": {"role": "assistant"}}
        return {"content": f"answer {messages[-1]['content']}", "tool_calls": [],
                "message": {"role": "assistant"}}

    def tool_result_message(self, tool_call, result):
        return {"role": "tool", "content": str(result)}


def test_agent_runs_tool_and_returns_answer():
    agent = BaseAgent(ToolCallingLLM(), "you add numbers", [add])
    assert agent.run("2 + 3?") == "answer 5.0"


def test_math_tools():
    assert add("1", 2) == 3.0
    assert divide(1, 0) == "undefined (division by zero)"


def test_settings_overrides_win():
    settings = get_settings({"chunk_size": 42})
    assert settings["chunk_size"] == 42


def test_settings_read_environment(monkeypatch):
    monkeypatch.setenv("CHUNK_OVERLAP", "7")
    assert get_settings()["chunk_overlap"] == 7


@pytest.mark.parametrize("name, cls", [
    ("word", WordChunker),
    ("character", CharacterChunker),
    ("recursive", RecursiveChunker),
    ("sentence", SentenceChunker),
    ("paragraph", ParagraphChunker),
    ("heading", HeadingChunker),
    ("page", PageChunker),
])
def test_factory_builds_chunkers(name, cls):
    chunker = build_chunker(get_settings({"chunker": name, "chunk_size": 10, "chunk_overlap": 2}))
    assert isinstance(chunker, cls)
    assert chunker.chunk_size == 10


def test_factory_token_chunker_uses_tiktoken(monkeypatch):
    # without tiktoken the token chunker explains how to install it
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    with pytest.raises(ImportError, match="tokens"):
        build_chunker(get_settings({"chunker": "token"}))


def test_factory_rejects_unknown_names():
    settings = get_settings({"chunker": "magic", "reranker": "magic"})
    with pytest.raises(ValueError):
        build_chunker(settings)
    with pytest.raises(ValueError):
        build_reranker(settings)
