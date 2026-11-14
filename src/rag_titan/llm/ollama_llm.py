"""
Chat model that runs locally with Ollama (https://ollama.com).

Before using it, install Ollama and pull a model:
    ollama pull gemma3:4b

Ollama Cloud works too: host="https://ollama.com" and the OLLAMA_API_KEY
environment variable (it is sent for you, see llm/ollama_http.py).
"""

from rag_titan.llm.base_llm import BaseLLM
from rag_titan.llm.ollama_http import MESSAGE_FIELDS, clean_message, parse_host, post_json
from rag_titan.llm.tool_schema import tools_to_schemas
from rag_titan.utils.logger import get_logger

logger = get_logger(__name__)


class OllamaLLM(BaseLLM):
    """
    Talks to an Ollama server.

    Args:
        model: Ollama model name, for example "gemma3:4b".
        host: Ollama server address. None means OLLAMA_HOST, then
            http://localhost:11434.
        think: turn on "thinking" for models that support it.
    """

    def __init__(self, model, host=None, think=False):
        super().__init__(model)
        self.think = think
        self.host = parse_host(host)

    def chat(self, messages, tools=None):
        """Send messages to Ollama. See BaseLLM.chat() for the reply format."""
        logger.debug("Calling Ollama model '%s'", self.model)

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [clean_message(message) for message in messages],
            "tools": tools_to_schemas(tools),
        }
        # only send "think" when it is on, some models do not support it
        if self.think:
            payload["think"] = True

        try:
            reply = post_json(self.host, "/api/chat", payload)
        except RuntimeError as error:
            raise RuntimeError(
                f"Ollama chat failed for model '{self.model}': {error}"
            ) from error

        message = self._reply_message(reply)
        return {
            "content": message["content"] or "",
            "tool_calls": self._read_tool_calls(message),
            "message": message,
        }

    def tool_result_message(self, tool_call, result):
        """Build the "tool" message Ollama expects after a tool call."""
        return {
            "role": "tool",
            "tool_name": tool_call["name"],
            "content": str(result),
        }

    def _reply_message(self, reply):
        """
        Read the assistant message from Ollama's reply.

        Every known field is present (None when missing), so code can read
        message["content"] safely, like it could with the old client.
        """
        raw = reply.get("message") if isinstance(reply, dict) else None
        if not isinstance(raw, dict):
            raise RuntimeError(
                f"Ollama reply for model '{self.model}' has no message: {reply!r:.200}"
            )

        message = {key: None for key in MESSAGE_FIELDS}
        message.update(clean_message(raw))
        return message

    def _read_tool_calls(self, message):
        """Turn Ollama's tool calls into simple dicts."""
        tool_calls = []

        for call in message["tool_calls"] or []:
            tool_calls.append({
                "name": call["function"]["name"],
                "arguments": dict(call["function"]["arguments"] or {}),
            })

        return tool_calls
