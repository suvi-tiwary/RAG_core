"""
Record real LLM replies once, then replay them without the model.

    recorder = RecordingLLM(OllamaLLM("gpt-oss:20b-cloud"))
    ...run a RAG type with recorder...
    replies = recorder.replies                 # save as JSON

    llm = ReplayLLM(replies)                   # later: same answers, no model, no network

The browser playground uses this to show real model answers for its examples.
"""

import hashlib
import json

from rag_titan.llm.base_llm import BaseLLM


def prompt_key(messages):
    """Return a short id for a conversation: only roles and texts count."""
    plain = [{"role": message["role"], "content": message["content"]} for message in messages]
    text = json.dumps(plain, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


class RecordingLLM(BaseLLM):
    """
    Passes every new prompt to a real LLM and remembers the reply.

    A prompt asked again gets the remembered reply, like ReplayLLM would give.
    Models do not always answer the same, so this keeps the recording exactly
    what a replay will do.

    Args:
        llm: any BaseLLM, for example an OllamaLLM.
    """

    def __init__(self, llm):
        super().__init__(llm.model)
        self.llm = llm
        self.replies = {}

    def chat(self, messages, tools=None):
        """Ask the real LLM (only the first time) and save its reply under the prompt's key."""
        if tools:
            raise ValueError("RecordingLLM records plain replies only, not tool calls")

        key = prompt_key(messages)
        if key not in self.replies:
            self.replies[key] = self.llm.chat(messages)["content"]

        return {"content": self.replies[key], "tool_calls": [], "message": {"role": "assistant"}}

    def tool_result_message(self, tool_call, result):
        """Not used: tool calls are not recorded."""
        return self.llm.tool_result_message(tool_call, result)


class ReplayLLM(BaseLLM):
    """
    Answers from recorded replies.

    Args:
        replies: dict of {prompt key: reply text}, e.g. RecordingLLM.replies.
        model: name to show, e.g. the model the replies came from.
    """

    def __init__(self, replies, model="replay"):
        if not isinstance(replies, dict):
            raise ValueError("replies must be a dict of {prompt key: reply text}")

        super().__init__(model)
        self.replies = replies

    def chat(self, messages, tools=None):
        """Return the recorded reply for exactly these messages."""
        key = prompt_key(messages)
        if key not in self.replies:
            raise ValueError(
                f"No recorded reply for this prompt (key {key}). The question, the "
                "documents or the settings differ from the recording; record it again."
            )

        return {"content": self.replies[key], "tool_calls": [], "message": {"role": "assistant"}}

    def tool_result_message(self, tool_call, result):
        """Build a simple tool message (tool calls are not replayed)."""
        return {"role": "tool", "content": str(result)}
