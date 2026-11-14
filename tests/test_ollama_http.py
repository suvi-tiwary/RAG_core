"""
Tests for the standard-library Ollama client and the tool schemas.

The expected values in fixtures/ollama_requests.json were recorded from the
official `ollama` package, so these tests prove nothing changed for users.
The fake_ollama fixture (tests/fake_ollama.py) stands in for Ollama.
"""

import json
from pathlib import Path

import pytest

from rag_titan.agents.tools import MATH_TOOLS
from rag_titan.llm.ollama_http import clean_message, parse_host, post_json
from rag_titan.llm.tool_schema import function_to_tool, tools_to_schemas

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "ollama_requests.json")
                      .read_text(encoding="utf-8"))


def test_hosts_are_parsed_like_the_official_client(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    for host, expected in FIXTURES["hosts"].items():
        assert parse_host(host) == expected, host

    assert parse_host(None) == "http://127.0.0.1:11434"
    monkeypatch.setenv("OLLAMA_HOST", "example.com:8080")
    assert parse_host(None) == "http://example.com:8080"
    assert parse_host("https://ollama.com") == "https://ollama.com:443"


def test_tool_schemas_match_the_official_client():
    assert tools_to_schemas(MATH_TOOLS) == FIXTURES["tool_schemas"]
    assert tools_to_schemas(None) == []

    ready_made = FIXTURES["tool_schemas"][0]
    assert function_to_tool(ready_made) is ready_made
    with pytest.raises(ValueError, match="function or a dict"):
        function_to_tool("add")


def test_clean_message_keeps_known_non_empty_fields_like_the_official_client():
    # what Ollama 0.40 really sends back: tool calls carry an "id" and an "index"
    message = {"role": "assistant", "content": "", "images": [], "thinking": None,
               "tool_calls": [{"id": "call_1", "function": {"index": 0, "name": "add",
                                                            "arguments": {"a": "2"}}}],
               "unknown": "x"}

    assert clean_message(message) == {"role": "assistant", "tool_calls": [
        {"function": {"name": "add", "arguments": {"a": "2"}}}]}
    assert clean_message({"role": "tool", "content": "5.0", "tool_name": "add"}) == \
        {"role": "tool", "content": "5.0", "tool_name": "add"}
    assert clean_message({"role": "user", "content": "0"})["content"] == "0"
    with pytest.raises(ValueError, match="must be a dict"):
        clean_message("hello")


def test_post_json_sends_json_and_the_api_key(fake_ollama, monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "secret-key")
    fake_ollama.reply = (200, '{"message": {"content": "hi"}}')

    reply = post_json(fake_ollama.address, "/api/chat", {"model": "m", "stream": False})

    assert reply == {"message": {"content": "hi"}}
    assert fake_ollama.requests == [{"path": "/api/chat", "authorization": "Bearer secret-key",
                                "body": {"model": "m", "stream": False}}]


def test_post_json_without_a_key_sends_no_authorization(fake_ollama, monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    post_json(fake_ollama.address, "/api/embed", {"input": ["a"]})
    assert fake_ollama.requests[0]["authorization"] is None


def test_ollama_errors_are_clear(fake_ollama):
    fake_ollama.reply = (404, '{"error": "model \\"nope\\" not found, try pulling it first"}')
    with pytest.raises(RuntimeError, match='HTTP 404: model "nope" not found'):
        post_json(fake_ollama.address, "/api/chat", {})

    fake_ollama.reply = (200, "<html>not json</html>")
    with pytest.raises(RuntimeError, match="not JSON"):
        post_json(fake_ollama.address, "/api/chat", {})


def test_unreachable_server_says_how_to_fix_it(fake_ollama):
    address = fake_ollama.address
    fake_ollama.shutdown()
    fake_ollama.server_close()                       # nothing listens on this port now

    with pytest.raises(RuntimeError, match="Is Ollama running"):
        post_json(address, "/api/chat", {}, timeout=5)
