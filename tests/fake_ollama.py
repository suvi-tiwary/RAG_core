"""
A fake Ollama server for tests: a tiny HTTP server in a background thread.

It saves every request in `server.requests` and answers with `server.reply`,
or with `server.replies[path]` when a reply is set for that API path.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


class FakeOllamaHandler(BaseHTTPRequestHandler):
    """Saves each request and answers with the server's prepared reply."""

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        self.server.requests.append({
            "path": self.path,
            "authorization": self.headers.get("Authorization"),
            "body": json.loads(self.rfile.read(length)),
        })
        status, body = self.server.replies.get(self.path, self.server.reply)
        self.send_response(status)
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, *args):
        pass                                    # keep the test output quiet


@pytest.fixture
def fake_ollama():
    """Start a fake Ollama on a free port and stop it after the test."""
    server = HTTPServer(("127.0.0.1", 0), FakeOllamaHandler)
    server.requests, server.replies = [], {}
    server.reply = (200, '{"ok": true}')
    server.address = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
