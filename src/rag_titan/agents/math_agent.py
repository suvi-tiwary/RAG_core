"""
A small example agent that solves math with tools.

    from rag_titan.agents import MathAgent
    agent = MathAgent(llm)
    print(agent.run("What is (12 + 8) * 3?"))
"""

from rag_titan.agents.base_agent import BaseAgent
from rag_titan.agents.prompts import MATH_AGENT_PROMPT
from rag_titan.agents.tools.math_tools import MATH_TOOLS


class MathAgent(BaseAgent):
    """
    Agent that uses add, subtract, multiply and divide tools.

    Args:
        llm: a BaseLLM whose model supports tool calling.
        max_steps: stop after this many think/tool rounds.
    """

    def __init__(self, llm, max_steps=5):
        super().__init__(
            llm=llm,
            system_prompt=MATH_AGENT_PROMPT,
            tools=MATH_TOOLS,
            max_steps=max_steps,
        )
