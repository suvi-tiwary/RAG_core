"""Tests for FunctionLLM, FunctionEmbedder and the live playground (no Ollama needed)."""

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import FakeEmbedder, PromptLLM

from rag_titan.embeddings import FunctionEmbedder
from rag_titan.llm import FunctionLLM
from rag_titan.playground import LivePlayground

PRESETS = Path(__file__).parent.parent / "docs" / "assets" / "playground" / "presets.json"
pytestmark = pytest.mark.skipif(not PRESETS.exists(), reason="no recorded presets yet")


def reply(messages):
    """A chat function like the browser's: messages in, text out."""
    return PromptLLM().chat(messages)["content"]


class CountingEmbed:
    """An embed function like the browser's, counting how many texts it embedded."""

    def __init__(self):
        self.texts = 0

    def __call__(self, texts):
        self.texts += len(texts)
        return [vector.tolist() for vector in FakeEmbedder().embed_many(texts)]


@pytest.fixture
def live():
    presets = json.loads(PRESETS.read_text(encoding="utf-8"))
    embed = CountingEmbed()
    return LivePlayground(presets, reply, embed, "prompt-fake", "fake-embed"), embed


def test_function_llm_and_embedder():
    llm = FunctionLLM(lambda messages: f"{len(messages)} messages", model="echo")
    assert llm.ask("system", "hi") == "2 messages" and llm.model == "echo"

    vectors = FunctionEmbedder(lambda texts: [[1.0, 0.0]] * len(texts)).embed_many(["a", "b"])
    assert len(vectors) == 2 and vectors[0].dtype == np.float32


def test_function_model_errors_are_clear():
    with pytest.raises(ValueError, match="must be a function"):
        FunctionLLM("not a function")
    with pytest.raises(ValueError, match="must return text"):
        FunctionLLM(lambda messages: 42).ask("system", "hi")
    with pytest.raises(ValueError, match="must return 2 vectors"):
        FunctionEmbedder(lambda texts: [[1.0]]).embed_many(["a", "b"])
    with pytest.raises(ValueError, match="tool calls"):
        FunctionLLM(reply).chat([{"role": "user", "content": "hi"}], tools=[print])


def test_preset_question_on_sample_documents_is_scored(live):
    playground, _ = live
    question = "How many paid leave days do I get each year?"
    result = playground.run("paragraph", "simple", question)

    assert result["answer"] == "the answer" and result["answered"]
    assert [step["step"] for step in result["steps"]] == ["retrieve", "rerank", "generate"]
    assert result["recall"] is not None


def test_own_question_and_own_text_are_not_scored(live):
    playground, _ = live
    own = [{"source": "notes.md", "text": "Remote work needs a laptop. Leave days are 25."}]
    result = playground.run("word", "adaptive", "leave days?", documents=own)

    assert result["recall"] is None and result["sources"][0] == "notes.md"
    assert result["steps"][0]["step"] == "route"


def test_documents_are_embedded_once_per_chunking(live):
    playground, embed = live
    playground.run("paragraph", "simple", "leave days?")
    after_first = embed.texts
    playground.run("paragraph", "adaptive", "remote work?")

    assert embed.texts - after_first <= 2          # only the questions, not the documents again


@pytest.mark.parametrize("documents, message", [
    ("plain text", "must be a list"),
    ([{"source": "a.md", "text": "   "}], "some 'text'"),
])
def test_bad_live_input(live, documents, message):
    playground, _ = live
    with pytest.raises(ValueError, match=message):
        playground.run("word", "simple", "leave?", documents=documents)
    with pytest.raises(ValueError, match="enter a question"):
        playground.run("word", "simple", "  ")
    with pytest.raises(ValueError, match="Unknown chunking"):
        playground.run("lines", "simple", "leave?")
