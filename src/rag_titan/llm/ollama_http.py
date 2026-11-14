"""
A tiny Ollama client built only on the Python standard library.

rag-titan used the `ollama` package before. That package needs pydantic,
whose compiled `_pydantic_core` file is blocked on Windows machines with
Smart App Control or an application-control policy. Plain HTTP with urllib
needs nothing compiled, so it runs everywhere Python runs.

Ollama Cloud: set host="https://ollama.com" and the OLLAMA_API_KEY
environment variable; the key is sent as an "Authorization: Bearer" header.
"""

import ipaddress
import json
import os
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_PORT = 11434

# the chat message fields Ollama understands (same list as the official client)
MESSAGE_FIELDS = ("role", "content", "thinking", "images", "tool_name", "tool_calls")


def parse_host(host=None):
    """
    Turn a short host like "example.com" into a full address.

    Follows the same rules as the official ollama client, so every OLLAMA_HOST
    value that worked before still works.

    Args:
        host: "1.2.3.4", ":8080", "https://ollama.com", ... None or "" means
            the OLLAMA_HOST environment variable, then http://127.0.0.1:11434.
    """
    host = host or os.getenv("OLLAMA_HOST") or ""
    scheme, _, hostport = host.partition("://")
    port = DEFAULT_PORT
    if not hostport:
        scheme, hostport = "http", host
    elif scheme == "http":
        port = 80
    elif scheme == "https":
        port = 443

    split = urllib.parse.urlsplit(f"{scheme}://{hostport}")
    name = split.hostname or "127.0.0.1"
    port = split.port or port

    try:
        if isinstance(ipaddress.ip_address(name), ipaddress.IPv6Address):
            name = f"[{name}]"          # urlsplit drops the brackets IPv6 needs
    except ValueError:
        pass                            # a normal name like "localhost"

    path = split.path.strip("/")
    address = f"{scheme}://{name}:{port}"
    return f"{address}/{path}" if path else address


def clean_message(message):
    """
    Return a copy of a chat message with only the fields Ollama knows.

    Matches the official client: unknown fields (like a tool call's "id") are
    dropped, and so are None, "" and [] values (an assistant tool-call
    message goes back without "content").

    Args:
        message: a dict like {"role": "user", "content": "hi"}.
    """
    if not isinstance(message, dict):
        raise ValueError(f"Each chat message must be a dict, got {type(message).__name__}")

    cleaned = {}
    for key in MESSAGE_FIELDS:
        value = message.get(key)
        if key == "tool_calls" and value:
            value = [_clean_tool_call(call) for call in value]
        if value is not None and value != "" and value != []:
            cleaned[key] = value

    return cleaned


def _clean_tool_call(call):
    """Keep only the function name and arguments of one tool call."""
    function = call.get("function") or {}
    return {"function": {"name": function.get("name"),
                         "arguments": function.get("arguments") or {}}}


def post_json(host, path, payload, timeout=None):
    """
    Send `payload` as JSON to an Ollama server and return the JSON reply.

    Args:
        host: server address, already passed through parse_host().
        path: API path, for example "/api/chat".
        payload: dict to send.
        timeout: seconds to wait, None waits as long as the model needs.
    """
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    api_key = os.getenv("OLLAMA_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(host + path, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Ollama returned HTTP {error.code}: {_error_text(error)}") from error
    except (urllib.error.URLError, OSError) as error:
        raise RuntimeError(
            f"Could not reach Ollama at {host} ({error}). Is Ollama running? "
            "Start it with `ollama serve` or open the Ollama app."
        ) from error

    try:
        return json.loads(text)
    except ValueError as error:
        raise RuntimeError(f"Ollama sent a reply that is not JSON: {text[:200]!r}") from error


def _error_text(error):
    """Read Ollama's own error message (like "model not found") from a failed reply."""
    try:
        body = error.read().decode("utf-8")
    except OSError:
        return str(error.reason)

    try:
        return json.loads(body).get("error", body)
    except (ValueError, AttributeError):
        return body or str(error.reason)
