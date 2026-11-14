"""
Use any Python function as the chat model.

    def my_model(messages):
        return call_my_api(messages)          # return the reply text

    llm = FunctionLLM(my_model, model="my-api")

The browser playground uses it to talk to your local Ollama.
"""

from rag_titan.llm.base_llm import BaseLLM


class FunctionLLM(BaseLLM):
    """
    A chat model made from a function.

    Args:
        reply: function that takes the messages (a list of {"role", "content"}
            dicts) and returns the reply text.
        model: name to show for this model.
    """

    def __init__(self, reply, model="function"):
        if not callable(reply):
            raise ValueError("reply must be a function: messages -> reply text")

        super().__init__(model)
        self.reply = reply

    def chat(self, messages, tools=None):
        """Call the function. Tools are not supported."""
        if tools:
            raise ValueError("FunctionLLM does not support tool calls")

        plain = [{"role": message["role"], "content": message["content"]} for message in messages]
        text = self.reply(plain)
        if not isinstance(text, str):
            raise ValueError(f"The reply function must return text, got {type(text).__name__}")

        message = {"role": "assistant", "content": text}
        return {"content": text, "tool_calls": [], "message": message}

    def tool_result_message(self, tool_call, result):
        """Build a simple tool message (tool calls are not supported)."""
        return {"role": "tool", "content": str(result)}
