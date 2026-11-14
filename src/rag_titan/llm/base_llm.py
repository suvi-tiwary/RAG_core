"""
Base class for every LLM provider (Ollama, OpenAI, Anthropic, ...).

To add a new provider, create a new file in this folder, inherit from
BaseLLM, and implement chat() and tool_result_message().
"""


class BaseLLM:
    """
    Common interface for chat models.

    Args:
        model: the model name, for example "gemma3:4b".
    """

    def __init__(self, model):
        self.model = model

    def chat(self, messages, tools=None):
        """
        Send a list of messages to the model.

        Args:
            messages: list of {"role": ..., "content": ...} dicts.
            tools: optional list of plain Python functions the model may call.

        Returns:
            A dict:
                {
                    "content": "text reply",
                    "tool_calls": [{"name": "add", "arguments": {"a": 1, "b": 2}}],
                    "message": the assistant message to append to the history,
                }
        """
        raise NotImplementedError

    def tool_result_message(self, tool_call, result):
        """Build the message that sends a tool's result back to the model."""
        raise NotImplementedError

    def ask(self, instructions, user_input):
        """
        Shortcut for a single question with a system instruction.

        Args:
            instructions: the system prompt.
            user_input: the user message.

        Returns:
            The reply text with spaces removed from both ends.
        """
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_input},
        ]
        return self.chat(messages)["content"].strip()
