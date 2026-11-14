"""
OllamaLLM and OllamaEmbedder must send exactly what the official `ollama`
client sent, and return the same values.

fixtures/ollama_requests.json was recorded from the official client with the
same messages and the same server replies used here.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from rag_titan import get_settings
from rag_titan.agents.tools import MATH_TOOLS
from rag_titan.embeddings import OllamaEmbedder
from rag_titan.factory import build_embedder, build_llm
from rag_titan.llm import OllamaLLM

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "ollama_requests.json")
                      .read_text(encoding="utf-8"))
RECORDED = {case["name"]: case for case in FIXTURES["cases"]}

MESSAGES = [{"role": "system", "content": "Be short."},
            {"role": "user", "content": "How many leave days?"}]
TEXT_REPLY = {"message": {"role": "assistant", "content": "20 days"}}
# the same tool reply the fixtures were recorded with, plus the "id" and
# "index" fields a real Ollama 0.40 adds (the official client dropped them)
TOOL_REPLY = {"message": {"role": "assistant", "content": "", "tool_calls": [
    {"id": "call_1", "function": {"index": 0, "name": "add", "arguments": {"a": 2, "b": "3"}}}]}}


def reply_with(server, body):
    server.replies["/api/chat"] = (200, json.dumps(body))


def check(name, server, result, path="/api/chat"):
    """Compare what the new code sent and returned with the recording."""
    recorded = RECORDED[name]
    sent = [(r["path"], r["authorization"], r["body"]) for r in server.requests]
    expected = [(path, r["authorization"], r["body"]) for r in recorded["requests"]]
    assert sent == expected, name
    assert result == recorded["result"], name
    server.requests.clear()


def short(reply):
    return {"content": reply["content"], "tool_calls": reply["tool_calls"]}


def test_chat_requests_match_the_official_client(fake_ollama, monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    llm = OllamaLLM("gemma3:4b", host=fake_ollama.address)
    reply_with(fake_ollama, TEXT_REPLY)
    check("chat_plain", fake_ollama, short(llm.chat(MESSAGES)))

    thinker = OllamaLLM("qwen3:4b", host=fake_ollama.address, think=True)
    check("chat_think", fake_ollama, short(thinker.chat(MESSAGES)))

    monkeypatch.setenv("OLLAMA_API_KEY", "secret-key")
    cloud = OllamaLLM("gpt-oss:20b-cloud", host=fake_ollama.address)
    check("chat_cloud_key", fake_ollama, short(cloud.chat(MESSAGES)))


def test_tool_calls_and_the_follow_up_match_the_official_client(fake_ollama, monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    llm = OllamaLLM("gemma3:4b", host=fake_ollama.address)
    reply_with(fake_ollama, TOOL_REPLY)
    first = llm.chat(MESSAGES, tools=MATH_TOOLS)
    check("chat_tools", fake_ollama, short(first))

    # every field can be read, like the old reply object allowed
    assert first["message"]["content"] is None and first["message"]["thinking"] is None

    reply_with(fake_ollama, TEXT_REPLY)
    follow = MESSAGES + [first["message"], llm.tool_result_message(first["tool_calls"][0], 5.0)]
    check("chat_tool_followup", fake_ollama, short(llm.chat(follow, tools=MATH_TOOLS)))


def test_embedding_batches_match_the_official_client(fake_ollama, monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    fake_ollama.replies["/api/embed"] = (200, '{"embeddings": [[0.5, 0.0], [1.5, -1.0]]}')
    embedder = OllamaEmbedder("nomic-embed-text", host=fake_ollama.address, batch_size=2)

    # the fake answers 2 vectors each time, so only the first batch (2 texts) is valid
    vectors = embedder.embed_many(["a", "b"])
    assert [v.dtype for v in vectors] == [np.float32, np.float32]
    assert [v.tolist() for v in vectors] == RECORDED["embed_batches"]["result"][:2]
    assert fake_ollama.requests[0]["body"] == RECORDED["embed_batches"]["requests"][0]["body"]

    with pytest.raises(RuntimeError, match="2 embeddings for 1 texts"):
        embedder.embed_many(["c"])


def test_failures_name_the_model(fake_ollama):
    fake_ollama.reply = (500, '{"error": "out of memory"}')
    with pytest.raises(RuntimeError, match="chat failed for model 'gemma3:4b'.*out of memory"):
        OllamaLLM("gemma3:4b", host=fake_ollama.address).chat(MESSAGES)
    with pytest.raises(RuntimeError, match="embedding failed for model 'e'.*out of memory"):
        OllamaEmbedder("e", host=fake_ollama.address).embed("x")

    fake_ollama.reply = (200, '{"done": true}')
    with pytest.raises(RuntimeError, match="has no message"):
        OllamaLLM("gemma3:4b", host=fake_ollama.address).chat(MESSAGES)


def test_settings_host_reaches_both_models(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    settings = get_settings({"ollama_host": "https://ollama.com"})

    assert build_llm(settings).host == "https://ollama.com:443"
    assert build_embedder(settings).host == "https://ollama.com:443"
    assert OllamaLLM("m", host="example.com:9999").host == "http://example.com:9999"
    assert OllamaEmbedder().host == "http://127.0.0.1:11434"
