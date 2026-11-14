"""
Base class for every tool-calling agent.

The loop:
    think (LLM) -> call tools -> send results back -> repeat
    until the LLM answers without calling a tool, or max_steps is reached.

To build a new agent, inherit from BaseAgent and pass a prompt and tools.
"""

from rag_titan.utils.logger import get_logger

logger = get_logger(__name__)


class BaseAgent:
    """
    A simple agent that lets an LLM call Python functions.

    Args:
        llm: any BaseLLM that supports tool calling.
        system_prompt: instructions that describe the agent's job.
        tools: list of plain Python functions the LLM may call.
        max_steps: stop after this many think/tool rounds.
    """

    def __init__(self, llm, system_prompt, tools, max_steps=5):
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools
        self.tools_by_name = {tool.__name__: tool for tool in tools}
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        """Forget the conversation and start fresh."""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def run(self, query):
        """
        Answer one user message, calling tools as needed.

        Args:
            query: the user's message.

        Returns:
            The agent's final answer text.
        """
        self.messages.append({"role": "user", "content": query})

        for step in range(1, self.max_steps + 1):
            logger.info("Agent step %d: thinking...", step)
            response = self.llm.chat(self.messages, tools=self.tools)
            self.messages.append(response["message"])

            # no tool calls means the LLM has given its final answer
            if not response["tool_calls"]:
                return response["content"]

            for tool_call in response["tool_calls"]:
                result = self._run_tool(tool_call)
                self.messages.append(self.llm.tool_result_message(tool_call, result))

        return "Maximum steps reached without a final answer."

    def _run_tool(self, tool_call):
        """Run one tool safely. Errors are returned as text to the LLM."""
        name = tool_call["name"]
        arguments = tool_call["arguments"]
        tool = self.tools_by_name.get(name)

        if tool is None:
            result = f"Error: unknown tool '{name}'"
        else:
            try:
                result = tool(**arguments)
            except Exception as error:
                result = f"Error while running {name}: {error}"

        logger.info("Tool %s(%s) = %s", name, arguments, result)
        return result
